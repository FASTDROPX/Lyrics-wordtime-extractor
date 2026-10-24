# Lyrics Wordtime Extractor
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/AI-Whisper-412991)
![stable-ts](https://img.shields.io/badge/stable--ts-2.19.1-orange)
![Local](https://img.shields.io/badge/Processing-Local_CPU-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

Python-модуль для определения начала и конца каждого слова в песне по аудио и готовому тексту lyrics.

- Поддерживает обычный текст, LRC и lyrics из метаданных.
- Учитывает паузы и пересечение основного и фонового вокала.
- Повторно проверяет проблемные строки.
- Экспортирует JSON, TTML, SRT и построчный LRC.

Использует Whisper и stable-ts. После загрузки модели работает локально.

## Проверка на треке

- Длительность: 3 минуты 32 секунды.
- Всего слов: 363.
- Получены интервалы: 337 слов.
- Остались неопределёнными: 26 слов.
- Обработка на тестовом ПК: около 1–1,5 минуты на CPU.
