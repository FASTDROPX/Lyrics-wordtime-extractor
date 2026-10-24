import json
import html
import math
import re
import xml.etree.ElementTree as ET

STAMP = r"(\d+):(\d{2}(?:[.,]\d+)?)"


def seconds(minutes, second_part):
    return round(int(minutes) * 60 + float(second_part.replace(",", ".")), 6)


def display_text(text):
    return html.unescape(str(text)).replace("**", "").replace("__", "").strip()


def voice_role(text):
    return "backing" if display_text(text).startswith(("(", "（", "[", "［", "{")) else "lead"


def acoustic_text(text):
    text = display_text(text).replace("’", "'").replace("‘", "'")
    text = re.sub("<[^>]*>", " ", text)
    return " ".join(re.sub("[^\\w\\s'\\-]", " ", text, flags=re.UNICODE).split())


def display_words(text):
    words, leading_symbols = ([], [])
    for part in display_text(text).split():
        if any((character.isalnum() for character in acoustic_text(part))):
            words.append(" ".join(leading_symbols + [part]))
            leading_symbols = []
        elif part in ["(", "[", "{", "（", "［"] or not words:
            leading_symbols.append(part)
        else:
            words[-1] += " " + part
    if leading_symbols and words:
        words[-1] += " " + " ".join(leading_symbols)
    return words


def parse_json_lyrics(text):
    try:
        document = json.loads(text)
        if isinstance(document, dict) and "words" in document:
            return {
                "text": document.get("text", ""),
                "segments": document.get("source_segments", []),
                "words": validate_words(document["words"], allow_unresolved=True),
                "source": "json",
                "project": document,
            }
    except (ValueError, TypeError):
        pass
    return None


def parse_ttml_lyrics(text):
    try:
        root = ET.fromstring(text)
        words, lines = ([], [])

        def timing(seconds_part):
            if seconds_part.endswith("s"):
                return float(seconds_part[:-1])
            parts = seconds_part.split(":")
            return sum(
                (
                    float(component) * 60**position
                    for position, component in enumerate(reversed(parts))
                )
            )

        for line, paragraph in enumerate(root.findall(".//{*}p")):
            line_words = []
            line_begin = timing(paragraph.get("begin", "0s"))
            for span in paragraph.findall(".//{*}span"):
                if span.get("begin") and span.get("end"):
                    word_text = "".join(span.itertext()).strip()
                    if word_text:
                        words.append(
                            {
                                "word": word_text,
                                "start": line_begin + timing(span.get("begin")),
                                "end": line_begin + timing(span.get("end")),
                                "line": line,
                                "status": "imported",
                            }
                        )
                        line_words.append(word_text)
            if line_words:
                lines.append(" ".join(line_words))
        if words:
            return {
                "text": "\n".join(lines),
                "segments": [],
                "words": validate_words(words),
                "source": "ttml",
            }
    except (ET.ParseError, ValueError):
        pass
    return None


def parse_lrc_lyrics(text):
    text = display_text(text)
    segments, plain, events = ([], [], [])
    offset = 0
    offset_match = re.search("\\[offset:([+-]?\\d+)\\]", text, re.I)
    if offset_match:
        offset = int(offset_match[1]) / 1000
    for raw_line in text.splitlines():
        timestamps = re.findall("\\[" + STAMP + "\\]", raw_line)
        line = re.sub("\\[" + STAMP + "\\]", "", raw_line)
        line = re.sub("\\[[A-Za-z][\\w-]*:[^\\]]*\\]", "", line)
        line = re.sub("<" + STAMP + ">", "", line).strip()
        if re.fullmatch(
            "[\\[({（](?:chorus|verse\\s*\\d*|intro|outro|bridge|instrumental|background vocals?|backing vocals?|бэк(?:-вокал)?|куплет\\s*\\d*|припев)[\\])}）]",
            line,
            re.I,
        ):
            line = ""
        if not any((c.isalnum() for c in acoustic_text(line))):
            line = ""
        if line:
            plain.append(line)
        for minutes, seconds_part in timestamps:
            events.append(
                {
                    "start": max(0, seconds(minutes, seconds_part) + offset),
                    "text": line,
                    "voice": voice_role(line),
                }
            )
    events.sort(key=lambda event: event["start"])
    for event_index, event in enumerate(events):
        if event["text"]:
            boundary = next(
                (
                    next_event
                    for next_event in events[event_index + 1 :]
                    if next_event["start"] > event["start"]
                    and (not next_event["text"] or next_event["voice"] == event["voice"])
                ),
                None,
            )
            segments.append(
                {
                    **event,
                    "end": boundary["start"] if boundary else None,
                    "end_kind": "silence" if boundary and (not boundary["text"]) else "next_line",
                }
            )
    if segments:
        plain = [segment["text"] for segment in segments]
    return {
        "text": "\n".join(plain),
        "segments": segments,
        "words": [],
        "source": "lrc" if segments else "text",
    }


def parse_lyrics(text):
    text = text.lstrip("\ufeff").strip()
    if text.startswith(("{", "[")):
        project = parse_json_lyrics(text)
        if project is not None:
            return project
    if text.startswith("<"):
        timed_lyrics = parse_ttml_lyrics(text)
        if timed_lyrics is not None:
            return timed_lyrics
    return parse_lrc_lyrics(text)


def validate_words(words, duration=None, allow_unresolved=False):
    if not isinstance(words, list) or len(words) > 50000:
        raise ValueError("invalid_words")
    result = []
    previous_by_line = {}
    for word_index, word_data in enumerate(words):
        start, end = (float(word_data["start"]), float(word_data["end"]))
        unresolved_allowed = (
            allow_unresolved and word_data.get("status") == "unresolved" and (end == start)
        )
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or (end < start)
            or (end == start and (not unresolved_allowed))
        ):
            raise ValueError("invalid_word_timing", word_index)
        line = int(word_data.get("line", 0))
        if start < previous_by_line.get(line, -1):
            raise ValueError("word_order_invalid", word_index)
        if duration and end > duration + 0.05:
            raise ValueError("word_outside_audio", word_index)
        word = str(word_data["word"]).strip()
        if not word:
            raise ValueError("empty_word", word_index)
        result.append(
            {
                **word_data,
                "id": word_index,
                "word": word,
                "start": round(start, 3),
                "end": round(end, 3),
                "line": int(word_data.get("line", 0)),
            }
        )
        previous_by_line[line] = start
    return result


def stamp(time_seconds, srt=False):
    milliseconds = round(time_seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{(',' if srt else '.')}{milliseconds:03}"


def export_srt(words):
    words.sort(key=lambda word: word["start"])
    return (
        "\n\n".join(
            (
                f"{word_index + 1}\n{stamp(word['start'], True)} --> {stamp(word['end'], True)}\n{word['word']}"
                for word_index, word in enumerate(words)
            )
        )
        + "\n",
        "text/plain",
        "srt",
    )


def export_ttml(words, document):
    groups = {}
    for word in words:
        groups.setdefault(word["line"], []).append(word)
    namespace = "http://www.w3.org/ns/ttml"
    ET.register_namespace("", namespace)
    root = ET.Element(
        f"{{{namespace}}}tt",
        {"{http://www.w3.org/XML/1998/namespace}lang": document.get("language", "ru")},
    )
    container = ET.SubElement(ET.SubElement(root, f"{{{namespace}}}body"), f"{{{namespace}}}div")
    for group in groups.values():
        line_start = group[0]["start"]
        paragraph = ET.SubElement(
            container,
            f"{{{namespace}}}p",
            {"begin": f"{line_start:.3f}s", "end": f"{max((word['end'] for word in group)):.3f}s"},
        )
        for word in group:
            span = ET.SubElement(
                paragraph,
                f"{{{namespace}}}span",
                {
                    "begin": f"{word['start'] - line_start:.3f}s",
                    "end": f"{word['end'] - line_start:.3f}s",
                },
            )
            span.text, span.tail = (word["word"], " ")
    return (
        ET.tostring(root, encoding="unicode", xml_declaration=True),
        "application/ttml+xml",
        "ttml",
    )


def export_lrc(words):
    groups = {}
    for word in words:
        groups.setdefault(word["line"], []).append(word)
    lines = []
    for group in groups.values():
        start_time = group[0]["start"]
        minutes = int(start_time // 60)
        seconds = start_time % 60
        lines.append(f"[{minutes:02}:{seconds:05.2f}]" + " ".join((word["word"] for word in group)))
    return ("\n".join(lines), "text/plain", "lrc")


def export_document(document, format_name):
    words = validate_words(
        document.get("words", []), document.get("duration"), allow_unresolved=format_name == "json"
    )
    if not words:
        raise ValueError("empty_words")
    if format_name == "json":
        return (
            json.dumps({**document, "words": words}, ensure_ascii=False, indent=2),
            "application/json",
            "json",
        )
    if format_name == "srt":
        return export_srt(words)
    if format_name == "ttml":
        return export_ttml(words, document)
    if format_name == "lrc":
        return export_lrc(words)
    raise ValueError("unsupported_export_format")
