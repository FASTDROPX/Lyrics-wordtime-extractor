import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alignmentengine import choose_language, prepare_windows, reconcile, retry_lines
from audio import (
    SAMPLE_RATE,
    decode_audio,
    load_stereo_channels,
    read_embedded_lyrics,
    read_samples,
)
from lyrics import parse_lyrics, validate_words, voice_role, acoustic_text

ProgressCallback = Callable[[str, float], None]
CancellationCallback = Callable[[], bool]


def check_cancelled(cancelled: CancellationCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("cancelled")


def report_progress(callback: ProgressCallback | None, phase: str, fraction: float) -> None:
    if callback is not None:
        callback(phase, fraction)


def alignment_options(language: str) -> dict[str, Any]:
    return {
        "language": language,
        "verbose": None,
        "regroup": False,
        "suppress_silence": True,
        "suppress_word_ts": True,
        "min_word_dur": 0.02,
        "vad": False,
    }


def align_line_windows(model, samples, windows, language, progress, cancelled):
    import stable_whisper

    segments = []
    options = alignment_options(language)
    for line_index, window in enumerate(windows):
        check_cancelled(cancelled)
        window_start = window["start"]
        window_end = window["end"]
        clip = samples[round(window_start * SAMPLE_RATE) : round(window_end * SAMPLE_RATE)]
        aligned = model.align_words(
            clip,
            [
                {
                    "start": 0.0,
                    "end": len(clip) / SAMPLE_RATE,
                    "text": acoustic_text(window["text"]),
                }
            ],
            **options,
        )

        if aligned is None or not aligned.segments:
            segments.append(
                {
                    "start": window_start,
                    "end": window_end,
                    "text": window["text"],
                    "words": [],
                }
            )
        else:
            segment = aligned.to_dict()["segments"][0]
            segment["start"] += window_start
            segment["end"] += window_start
            for word in segment.get("words", []):
                word["start"] = round(word["start"] + window_start, 3)
                word["end"] = round(word["end"] + window_start, 3)
            segments.append(segment)

        fraction = 0.08 + 0.75 * (line_index + 1) / len(windows)
        report_progress(progress, "aligning", fraction)

    return stable_whisper.WhisperResult(
        {"segments": segments, "language": language},
        check_sorted=False,
    )


def align_plain_text(model, samples, references, language, progress, cancelled):
    def on_alignment_progress(completed, total):
        check_cancelled(cancelled)
        fraction = min(0.83, 0.08 + 0.75 * completed / max(total, 1))
        report_progress(progress, "aligning", fraction)

    target_text = "\n".join(acoustic_text(line) for line in references)
    return model.align(
        samples,
        target_text,
        **alignment_options(language),
        original_split=True,
        progress_callback=on_alignment_progress,
        max_word_dur=None,
        word_dur_factor=None,
        nonspeech_skip=None,
    )


def refine_boundaries(model, samples, result, language, cancelled):
    import stable_whisper

    segments = []
    for segment in result.to_dict()["segments"]:
        check_cancelled(cancelled)
        line_result = stable_whisper.WhisperResult(
            {"segments": [segment], "language": language},
            check_sorted=False,
        )
        model.refine(samples, line_result, precision=0.04, steps="se", verbose=None)
        segments.extend(line_result.to_dict()["segments"])

    return stable_whisper.WhisperResult(
        {"segments": segments, "language": language},
        check_sorted=False,
    )


def collect_rows(result, references, windows):
    observed_segments = result.to_dict()["segments"]
    rows = []
    for line_index, reference in enumerate(references):
        observed_words = []
        if line_index < len(observed_segments):
            observed_words = observed_segments[line_index].get("words", [])
        anchor = windows[line_index]["start"] if windows else 0.0
        rows.append(reconcile(reference, observed_words, line_index, anchor))
    return rows


def build_word_timings(rows, references, duration):
    words = []
    for line_index, row in enumerate(rows):
        for observed in row:
            start = round(min(duration, max(0.0, float(observed["start"]))), 3)
            end = round(max(start, min(duration, float(observed["end"]))), 3)
            raw_probability = observed.get("probability")
            probability = float(raw_probability) if raw_probability is not None else None
            word_duration = end - start

            if word_duration == 0:
                status = "unresolved"
            elif (
                observed.get("recovery")
                or (probability is not None and probability < 0.4)
                or word_duration > 5
                or word_duration < 0.04
            ):
                status = "review"
            else:
                status = "estimated"

            words.append(
                {
                    "id": len(words),
                    "word": observed["word"].strip(),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "line": line_index,
                    "voice": voice_role(references[line_index]),
                    "score": probability,
                    "status": status,
                    "recovery": observed.get("recovery"),
                }
            )
    return validate_words(words, duration, allow_unresolved=True)


def extract(
    audio,
    lyrics_text: str | None = None,
    *,
    language: str = "auto",
    model_name: str = "base",
    refine: bool = False,
    repair: bool = True,
    model_dir=None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
    threads: int = 4,
) -> dict[str, Any]:
    import stable_whisper
    import torch

    check_cancelled(cancelled)
    audio_path = Path(audio).resolve(strict=True)
    if lyrics_text is None:
        lyrics_text = read_embedded_lyrics(audio_path)
    if not lyrics_text or not lyrics_text.strip():
        raise ValueError("empty_lyrics")

    parsed = parse_lyrics(lyrics_text)
    text = parsed["text"]
    if not text.strip():
        raise ValueError("empty_lyrics")
    source_segments = parsed["segments"]
    references = (
        [segment["text"] for segment in source_segments]
        if source_segments
        else [line for line in text.splitlines() if line.strip()]
    )

    report_progress(progress, "loading_model", 0.01)
    torch.set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
    model_options = {"device": "cpu"}
    if model_dir is not None:
        model_options["download_root"] = str(model_dir)
    model = stable_whisper.load_model(model_name, **model_options)
    check_cancelled(cancelled)

    with tempfile.TemporaryDirectory(prefix="lyrics-extractor-") as directory:
        temporary_directory = Path(directory)
        mono_path = temporary_directory / "mono.wav"
        decode_audio(audio_path, mono_path)
        samples = read_samples(mono_path)
        duration = len(samples) / SAMPLE_RATE
        if duration <= 0:
            raise ValueError("empty_audio")
        windows = prepare_windows(source_segments, duration) if source_segments else []

        report_progress(progress, "detecting_language", 0.06)
        language, language_info = choose_language(
            model,
            samples,
            text,
            language,
            windows[0]["start"] if windows else 0,
        )
        if windows:
            result = align_line_windows(model, samples, windows, language, progress, cancelled)
        else:
            result = align_plain_text(model, samples, references, language, progress, cancelled)
        check_cancelled(cancelled)
        if result is None or not result.segments:
            raise ValueError("alignment_failed")

        if refine:
            report_progress(progress, "refining", 0.86)
            result = refine_boundaries(model, samples, result, language, cancelled)

        rows = collect_rows(result, references, windows)
        repair_stats = None
        if repair:
            report_progress(progress, "repairing", 0.88)
            stereo_channels = None

            def get_channels():
                nonlocal stereo_channels
                if stereo_channels is None:
                    stereo_channels = load_stereo_channels(audio_path, temporary_directory)
                return stereo_channels

            def on_repair_progress(completed, total):
                check_cancelled(cancelled)
                fraction = 0.88 + 0.1 * completed / max(total, 1)
                report_progress(progress, "repairing", fraction)

            rows, repair_stats = retry_lines(
                model,
                samples,
                result,
                references,
                windows,
                language,
                lambda: check_cancelled(cancelled),
                on_repair_progress,
                channel_loader=get_channels,
            )

        check_cancelled(cancelled)
        document = {
            "text": text,
            "language": language,
            "language_info": language_info,
            "duration": duration,
            "words": build_word_timings(rows, references, duration),
            "source_segments": source_segments,
            "refined": refine,
        }
        if repair_stats is not None:
            document["repair_stats"] = repair_stats

    report_progress(progress, "done", 1.0)
    return document
