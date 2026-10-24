import math
from difflib import SequenceMatcher
from lyrics import acoustic_text, voice_role, display_words


def token_key(word):
    return acoustic_text(word).casefold()


def reconcile(text, observed_words, line=0, anchor=0.0):
    expected_words = display_words(text)
    matched_words = {}
    matcher = SequenceMatcher(
        a=[token_key(aligned_word) for aligned_word in expected_words],
        b=[token_key(aligned_word["word"]) for aligned_word in observed_words],
        autojunk=False,
    )
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            matched_words[block.a + offset] = observed_words[block.b + offset]
    result = []
    for word_index, word in enumerate(expected_words):
        if word_index in matched_words:
            aligned_word = dict(matched_words[word_index])
            aligned_word["word"] = word
        else:
            anchor_time = result[-1]["end"] if result else anchor
            next_start = next(
                (
                    float(matched_words[next_index]["start"])
                    for next_index in range(word_index + 1, len(expected_words))
                    if next_index in matched_words
                ),
                anchor_time,
            )
            anchor_time = max(
                result[-1]["start"] if result else anchor, min(anchor_time, next_start)
            )
            aligned_word = {
                "word": word,
                "start": anchor_time,
                "end": anchor_time,
                "probability": None,
                "missing": True,
            }
        aligned_word["line"], aligned_word["voice"] = (line, voice_role(text))
        result.append(aligned_word)
    return result


def quality(words):
    if not words:
        return -1000000.0
    penalties = 0.0
    probabilities = []
    for word in words:
        duration = float(word["end"]) - float(word["start"])
        if duration <= 0.005:
            penalties += 12
        elif duration < 0.04:
            penalties += 3
        if duration > 5:
            penalties += 5 + (duration - 5) * 2
        probability = word.get("probability", word.get("score"))
        if probability is not None:
            probabilities.append(float(probability))
    return -penalties + (sum(probabilities) / len(probabilities) if probabilities else 0)


def needs_retry(words, window=None):
    if any(
        (word["end"] - word["start"] < 0.04 or word["end"] - word["start"] > 5 for word in words)
    ):
        return True
    return bool(
        window
        and words
        and (window.get("voice") == "backing")
        and (window.get("end_kind") != "silence")
        and (abs(float(words[-1]["end"]) - window["end"]) < 0.06)
    )


def prepare_windows(source, duration):
    windows = []
    for segment in source:
        window_start = float(segment["start"])
        window_end = (
            duration if segment.get("end") is None else min(float(segment["end"]), duration)
        )
        if (
            not math.isfinite(window_start)
            or not math.isfinite(window_end)
            or (not 0 <= window_start < window_end)
        ):
            raise ValueError("invalid_line_timing")
        if window_end - window_start > 29:
            window_end = min(
                window_end, window_start + max(8.0, len(segment["text"].split()) * 0.9 + 2)
            )
        windows.append(
            {
                **segment,
                "start": window_start,
                "end": window_end,
                "voice": segment.get("voice", voice_role(segment["text"])),
            }
        )
    return windows


def choose_language(model, samples, text, requested, start=0):
    import whisper

    letters = [character for character in acoustic_text(text) if character.isalpha()]
    latin_count = sum(("a" <= character.lower() <= "z" for character in letters))
    cyrillic_count = sum(("Ѐ" <= character <= "ӿ" for character in letters))
    script_mismatch = requested in ["ru", "uk"] and latin_count > 10 and (cyrillic_count == 0)
    if requested != "auto" and (not script_mismatch):
        return (requested, None)
    offset = max(0, round(start * 16000))
    clip = samples[offset : offset + 30 * 16000]
    mel_spectrogram = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(clip), n_mels=model.dims.n_mels
    ).to(model.device)
    _, language_scores = model.detect_language(mel_spectrogram)
    detected_language = max(language_scores, key=language_scores.get)
    confidence = float(language_scores[detected_language])
    if script_mismatch and confidence < 0.6:
        return (requested, None)
    return (
        detected_language,
        {
            "requested": requested,
            "detected": detected_language,
            "confidence": round(confidence, 3),
            "corrected_mismatch": script_mismatch,
        },
    )


def speech_view(samples):
    import numpy as np

    if len(samples) < 2:
        return samples
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(len(samples), 1 / 16000)
    frequency_gain = np.ones_like(frequencies)
    frequency_gain[frequencies < 150] = 0.25
    frequency_gain[(frequencies >= 350) & (frequencies <= 3500)] = 1.5
    frequency_gain[frequencies > 6500] = 0.4
    filtered_samples = np.fft.irfft(spectrum * frequency_gain, n=len(samples)).astype(np.float32)
    peak_amplitude = np.percentile(np.abs(filtered_samples), 99.5)
    if peak_amplitude > 1e-06:
        filtered_samples *= min(3.0, 0.7 / peak_amplitude)
    return np.clip(filtered_samples, -1, 1)


def retry_lines(
    model, samples, result, references, windows, language, cancelled, progress, channel_loader=None
):
    observed_segments = result.to_dict()["segments"]
    output = []
    for line_index, text in enumerate(references):
        observed_segment = (
            observed_segments[line_index]
            if line_index < len(observed_segments)
            else {"words": [], "start": 0, "end": 0}
        )
        anchor = (
            windows[line_index]["start"] if windows else float(observed_segment.get("start", 0))
        )
        output.append(reconcile(text, observed_segment.get("words", []), line_index, anchor))
    suspects = [
        line_index
        for line_index, line in enumerate(output)
        if needs_retry(line, windows[line_index] if windows else None)
    ]
    stats = {
        "retried_lines": 0,
        "replaced_lines": 0,
        "initial_unresolved": sum(
            (int(word["end"] <= word["start"]) for line in output for word in line)
        ),
    }
    for count, line_index in enumerate(suspects):
        cancelled()
        progress(count, len(suspects))
        original_words = output[line_index]
        if windows:
            window = windows[line_index]
            window_start, window_end = (window["start"], window["end"])
            hard_end = window.get("end_kind") == "silence"
        else:
            if line_index >= len(observed_segments):
                continue
            window_start, window_end = (
                float(observed_segments[line_index]["start"]),
                float(observed_segments[line_index]["end"]),
            )
            hard_end = False
        max_span = max(8.0, len(references[line_index].split()) * 0.9 + 2)
        local_end = min(window_end, window_start + max_span)
        context_start = max(
            0.0, window_start - (0.3 if voice_role(references[line_index]) == "backing" else 0.0)
        )
        context_end = min(len(samples) / 16000, local_end + (0 if hard_end else 0.65))
        if context_end - context_start < 0.04 or context_end - context_start > 29:
            continue
        stats["retried_lines"] += 1
        best_words = original_words
        for method in ["context", "voice_eq", "dynamic"] + (
            ["left", "right"] if channel_loader else []
        ):
            cancelled()
            clip = samples[round(context_start * 16000) : round(context_end * 16000)]
            if method == "voice_eq":
                clip = speech_view(clip)
            if method in ["left", "right"]:
                channels = channel_loader()
                clip = channels[method][round(context_start * 16000) : round(context_end * 16000)]
            options = dict(language=language, verbose=None, regroup=False, suppress_silence=False)
            if method == "dynamic":
                options["dynamic_heads"] = True
            candidate = model.align_words(
                clip,
                [
                    {
                        "start": 0.0,
                        "end": len(clip) / 16000,
                        "text": acoustic_text(references[line_index]),
                    }
                ],
                **options
            )
            if candidate is None or not candidate.segments:
                continue
            candidate_words = candidate.to_dict()["segments"][0].get("words", [])
            for word in candidate_words:
                word["start"] = round(float(word["start"]) + context_start, 3)
                word["end"] = round(float(word["end"]) + context_start, 3)
            candidate_words = reconcile(
                references[line_index], candidate_words, line_index, window_start
            )
            if quality(candidate_words) > quality(best_words) + 0.05:
                best_words = candidate_words
                for word in best_words:
                    word["recovery"] = method
            if not needs_retry(best_words, windows[line_index] if windows else None):
                break
        if best_words is not original_words:
            stats["replaced_lines"] += 1
            output[line_index] = best_words
    stats["remaining_unresolved"] = sum(
        (int(word["end"] <= word["start"]) for line in output for word in line)
    )
    return (output, stats)
