import os
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import mutagen
import numpy as np

SAMPLE_RATE = 16000


def read_embedded_lyrics(audio_path: Path) -> str:
    audio = mutagen.File(audio_path)
    if audio is None or not audio.tags:
        return ""

    plain_candidates = []
    timed_candidates = []
    for tag_name, tag_value in audio.tags.items():
        tag_name = str(tag_name)
        if tag_name.startswith("USLT"):
            plain_candidates.insert(0, tag_value.text)
        elif tag_name.startswith("SYLT") and getattr(tag_value, "format", None) == 2:
            lines = []
            for text, milliseconds in tag_value.text:
                minutes = int(milliseconds // 60000)
                seconds = milliseconds / 1000 % 60
                lines.append(f"[{minutes:02}:{seconds:05.2f}]{text.strip()}")
            timed_candidates.append("\n".join(lines))
        elif "lyrics" in tag_name.lower() or tag_name == "©lyr":
            values = tag_value if isinstance(tag_value, list) else [tag_value]
            plain_candidates.extend(str(value) for value in values)

    return next((text for text in plain_candidates + timed_candidates if text.strip()), "")


def decode_audio(source: Path, destination: Path, channels: int = 1) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.run(command, capture_output=True, creationflags=creation_flags)
    if process.returncode:
        raise ValueError("audio_decode_failed", process.stderr.decode(errors="replace").strip())


def read_samples(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path)) as audio:
        channel_count = audio.getnchannels()
        if audio.getsampwidth() != 2 or audio.getframerate() != SAMPLE_RATE:
            raise ValueError("invalid_pcm_format")
        frames = audio.readframes(audio.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    return samples if channel_count == 1 else samples.reshape(-1, channel_count)


def load_stereo_channels(source: Path, directory: Path) -> dict[str, np.ndarray]:
    stereo_path = directory / "stereo.wav"
    decode_audio(source, stereo_path, channels=2)
    stereo = read_samples(stereo_path)
    return {"left": stereo[:, 0], "right": stereo[:, 1]}
