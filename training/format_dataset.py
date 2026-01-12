"""
Форматирование датасета для обучения модели

Улучшенная версия v2:
- Расширенный системный промпт с примерами
- Фильтрация некачественных данных
- Больше разнообразных промптов
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


# Улучшенный системный промпт с примерами (few-shot)
SYSTEM_PROMPT = """Ту шоири тоҷикӣ ҳастӣ. Ту метавонӣ шеърҳои классикӣ ва муосир бинависӣ.

Шаклҳои шеър:
- Рубоӣ: чор мисраъ, қофияи ААБА
- Ғазал: байтҳо бо радифу қофия
- Қасида: шеъри дароз дар мадҳ ё васф
- Маснавӣ: ҷуфти мисраъҳо бо қофияи АА БА СА
- Шеъри озод: бе қофияи муайян

Услуби шоирони бузург: Рӯдакӣ, Ҳофиз, Саъдӣ, Хайём, Мавлавӣ, Ҷомӣ.

Ба забони тоҷикӣ (кириллӣ) менависӣ. Шеърҳои ту зебо, маънидор ва пурэҳсос ҳастанд."""


# Примеры хороших стихов для reference (не включаются в промпт, для валидации)
QUALITY_EXAMPLES = {
    "rubaiyat": [
        "Бӯи ҷӯи Мӯлиён ояд ҳаме,\nЁди ёри меҳрубон ояд ҳаме,\nРеги Омуву дурушти роҳи ӯ,\nЗери поям парниён ояд ҳаме.",
    ],
    "ghazal": [
        "Дил меравад зи дастам, соҳибдилон Худо ро,\nДардо ки раҳ намедонам, ёрон Худо ро.",
    ],
}


@dataclass
class FormattedExample:
    """Пример для обучения"""
    messages: List[Dict[str, str]]
    text: str
    quality_score: float = 1.0


class QualityFilter:
    """Фильтр качества стихов"""

    # Минимальная длина текста
    MIN_LENGTH = 30

    # Максимальная длина
    MAX_LENGTH = 2000

    # Минимальное количество строк
    MIN_LINES = 2

    # Паттерны для обнаружения мусора
    GARBAGE_PATTERNS = [
        r'^[\s\d\.\,\:\;]+$',  # Только пробелы, цифры, пунктуация
        r'[\u0600-\u06FF]{10,}',  # Много арабских символов подряд (не транслитерировано)
        r'^[a-zA-Z\s]+$',  # Только латиница
    ]

    # Обязательные таджикские буквы (хотя бы некоторые должны быть)
    TAJIK_LETTERS = set('ғӣқӯҳҷ')

    # Частые таджикские слова
    COMMON_WORDS = {'ва', 'ки', 'аз', 'ба', 'дар', 'бо', 'ман', 'ту', 'ӯ', 'мо',
                    'дил', 'ҷон', 'ишқ', 'ёр', 'гул', 'шаб', 'рӯз', 'моҳ', 'об',
                    'аст', 'буд', 'шуд', 'нест', 'ҳаст', 'бошад', 'кунад'}

    def is_quality_poem(self, text: str) -> tuple:
        """
        Проверка качества стиха.

        Returns:
            (is_valid, quality_score, reason)
        """
        if not text:
            return False, 0.0, "empty"

        # Длина
        if len(text) < self.MIN_LENGTH:
            return False, 0.0, "too_short"
        if len(text) > self.MAX_LENGTH:
            return False, 0.0, "too_long"

        # Количество строк
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < self.MIN_LINES:
            return False, 0.0, "too_few_lines"

        # Проверка на мусор
        for pattern in self.GARBAGE_PATTERNS:
            if re.match(pattern, text):
                return False, 0.0, "garbage_pattern"

        # Проверка на арабские символы (не полностью транслитерировано)
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(text)
        if total_chars > 0 and arabic_chars / total_chars > 0.3:
            return False, 0.0, "too_much_arabic"

        # Подсчёт качества
        score = 1.0

        # Бонус за таджикские буквы
        text_lower = text.lower()
        tajik_letter_count = sum(1 for c in text_lower if c in self.TAJIK_LETTERS)
        if tajik_letter_count > 0:
            score += min(0.2, tajik_letter_count * 0.02)

        # Бонус за частые таджикские слова
        words = set(re.findall(r'\b\w+\b', text_lower))
        common_word_count = len(words & self.COMMON_WORDS)
        if common_word_count > 0:
            score += min(0.3, common_word_count * 0.05)

        # Штраф за слишком короткие строки
        short_lines = sum(1 for l in lines if len(l) < 10)
        if short_lines > len(lines) / 2:
            score -= 0.2

        # Штраф за повторы
        if len(set(lines)) < len(lines) * 0.7:
            score -= 0.3

        return True, max(0.1, min(1.5, score)), "ok"


class PromptGenerator:
    """Генератор разнообразных промптов для обучения"""

    FORM_PROMPTS = {
        "rubaiyat": [
            "Рубоӣ бинавис",
            "Як рубоӣ эҷод кун",
            "Чор мисраъ дар шакли рубоӣ бинавис",
            "Рубоӣ дар бораи {theme} бинавис",
            "Рубоии зебо эҷод кун",
            "Рубоӣ бо қофияи ААБА бинавис",
            "Як рубоии кӯтоҳ бисоз",
            "Рубоӣ дар мавзӯи {theme} эҷод кун",
        ],
        "ghazal": [
            "Ғазал бинавис",
            "Ғазали ошиқона эҷод кун",
            "Ғазал дар мавзӯи {theme} бисоз",
            "Ғазали зебо бинавис",
            "Ғазал бо радиф эҷод кун",
            "Ғазали пурсӯз бинавис",
            "Як ғазали дилкаш эҷод кун",
        ],
        "qasida": [
            "Қасида бинавис",
            "Қасида дар васфи {theme} эҷод кун",
            "Қасидаи мадҳия бинавис",
            "Қасидаи кӯтоҳ эҷод кун",
        ],
        "masnavi": [
            "Маснавӣ бинавис",
            "Маснавии кӯтоҳ эҷод кун",
            "Достон дар шакли маснавӣ бинавис",
            "Маснавӣ дар бораи {theme} бисоз",
        ],
        "free": [
            "Шеъри озод бинавис",
            "Шеъри муосир эҷод кун",
            "Шеър дар бораи {theme} бинавис",
            "Шеъри озод дар мавзӯи {theme}",
            "Шеър бинавис",
            "Як шеъри зебо эҷод кун",
        ],
        "fragment": [
            "Қитъа бинавис",
            "Чанд байт эҷод кун",
            "Абёт дар бораи {theme} бинавис",
        ],
        "other": [
            "Шеър бинавис",
            "Шеъри зебо эҷод кун",
            "Абёти зебо бинавис",
            "Як шеър бисоз",
            "Шеър дар бораи {theme} бинавис",
        ],
    }

    THEMES = {
        "love": "ишқ", "nature": "табиат", "spring": "баҳор",
        "life": "зиндагӣ", "death": "марг", "wisdom": "ҳикмат",
        "wine": "шароб", "beauty": "зебоӣ", "homeland": "ватан",
        "friendship": "дӯстӣ", "night": "шаб", "moon": "моҳ",
        "sun": "офтоб", "flower": "гул", "garden": "боғ",
        "heart": "дил", "soul": "ҷон", "time": "замон",
        "fate": "тақдир", "god": "Худо", "morning": "субҳ",
        "autumn": "тирамоҳ", "winter": "зимистон", "youth": "ҷавонӣ",
        "old_age": "пирӣ", "separation": "ҷудоӣ", "reunion": "висол",
        "pain": "дард", "joy": "шодӣ", "tears": "ашк",
        "candle": "шамъ", "butterfly": "парвона", "nightingale": "булбул",
    }

    POET_STYLE_PROMPTS = [
        "Шеър дар услуби {poet} бинавис",
        "Ба тарзи {poet} шеър эҷод кун",
        "Мисли {poet} {form} бинавис",
    ]

    def generate_prompt(
        self,
        form: str,
        poet: Optional[str] = None,
        theme: Optional[str] = None,
    ) -> str:
        prompts = self.FORM_PROMPTS.get(form, self.FORM_PROMPTS["other"])
        prompt = random.choice(prompts)

        if "{theme}" in prompt:
            if theme and theme in self.THEMES:
                theme_tj = self.THEMES[theme]
            else:
                theme_tj = random.choice(list(self.THEMES.values()))
            prompt = prompt.replace("{theme}", theme_tj)

        # Иногда добавляем стиль поэта
        if poet and random.random() < 0.2:
            style_prompt = random.choice(self.POET_STYLE_PROMPTS)
            form_name = {"rubaiyat": "рубоӣ", "ghazal": "ғазал"}.get(form, "шеър")
            prompt = style_prompt.replace("{poet}", poet).replace("{form}", form_name)

        return prompt


class DatasetFormatter:
    """Форматирование датасета для обучения"""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.prompt_generator = PromptGenerator()
        self.quality_filter = QualityFilter()

    def format_example(
        self,
        poem_text: str,
        form: str,
        poet: Optional[str] = None,
        themes: Optional[List[str]] = None,
    ) -> Optional[FormattedExample]:
        """Форматирование одного примера с проверкой качества."""

        # Проверка качества
        is_valid, quality_score, reason = self.quality_filter.is_quality_poem(poem_text)
        if not is_valid:
            return None

        # Очистка текста
        poem_text = self._clean_text(poem_text)

        # Генерируем промпт
        theme = random.choice(themes) if themes else None
        user_prompt = self.prompt_generator.generate_prompt(form, poet, theme)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": poem_text},
        ]

        text = self._format_chatml(messages)

        return FormattedExample(messages=messages, text=text, quality_score=quality_score)

    def _clean_text(self, text: str) -> str:
        """Очистка текста стиха."""
        # Убираем лишние пробелы
        text = re.sub(r' +', ' ', text)
        # Убираем пустые строки в начале/конце
        text = text.strip()
        # Убираем множественные переносы строк
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Убираем пробелы в конце строк
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        return text

    def _format_chatml(self, messages: List[Dict[str, str]]) -> str:
        """Форматирование в ChatML"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        source_type: str = "ganjoor",
        min_quality: float = 0.5,
    ) -> int:
        """Обработка JSONL файла с фильтрацией по качеству."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        skipped = 0
        stats = {"too_short": 0, "too_much_arabic": 0, "garbage": 0, "low_quality": 0}

        with open(input_path, 'r', encoding='utf-8') as fin:
            with open(output_path, 'w', encoding='utf-8') as fout:
                for line in fin:
                    data = json.loads(line)

                    if source_type == "ganjoor":
                        poem_text = data.get("text_tajik", "")
                    else:
                        poem_text = data.get("text", "")

                    form = data.get("form", "other")
                    poet = data.get("poet")
                    themes = data.get("themes", [])

                    example = self.format_example(poem_text, form, poet, themes)

                    if example is None:
                        skipped += 1
                        continue

                    if example.quality_score < min_quality:
                        stats["low_quality"] += 1
                        skipped += 1
                        continue

                    output_data = {
                        "text": example.text,
                        "messages": example.messages,
                        "quality_score": example.quality_score,
                    }
                    json.dump(output_data, fout, ensure_ascii=False)
                    fout.write("\n")
                    count += 1

        print(f"Обработано: {count}, пропущено: {skipped}")
        print(f"  -> {output_path}")
        return count

    def create_train_val_split(
        self,
        input_path: str,
        output_dir: str,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple:
        """Разбиение на train/val."""
        random.seed(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(line)

        random.shuffle(examples)

        val_size = int(len(examples) * val_ratio)
        val_examples = examples[:val_size]
        train_examples = examples[val_size:]

        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"

        with open(train_path, 'w', encoding='utf-8') as f:
            f.writelines(train_examples)

        with open(val_path, 'w', encoding='utf-8') as f:
            f.writelines(val_examples)

        print(f"Train: {len(train_examples)} -> {train_path}")
        print(f"Val: {len(val_examples)} -> {val_path}")

        return len(train_examples), len(val_examples)

    def merge_datasets(
        self,
        input_paths: List[str],
        output_path: str,
    ) -> int:
        """Объединение нескольких датасетов."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, 'w', encoding='utf-8') as fout:
            for input_path in input_paths:
                if Path(input_path).exists():
                    with open(input_path, 'r', encoding='utf-8') as fin:
                        for line in fin:
                            fout.write(line)
                            count += 1

        print(f"Объединено {count} примеров -> {output_path}")
        return count


def prepare_full_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    training_dir: str = "data/training",
) -> dict:
    """Полная подготовка датасета."""
    formatter = DatasetFormatter()
    stats = {"classical": 0, "modern": 0, "manual": 0, "total": 0}

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    training_dir = Path(training_dir)

    processed_files = []

    # Классика (Ganjoor) - локальный скрапинг
    ganjoor_file = raw_dir / "ganjoor" / "all_classical.jsonl"
    if ganjoor_file.exists():
        output = processed_dir / "classical.jsonl"
        stats["classical"] = formatter.process_jsonl(
            str(ganjoor_file), str(output), source_type="ganjoor"
        )
        processed_files.append(str(output))

    # Классика (Ganjoor HuggingFace) - 119K стихов
    ganjoor_hf_file = raw_dir / "ganjoor_hf" / "all_poems.jsonl"
    if ganjoor_hf_file.exists():
        output = processed_dir / "ganjoor_hf.jsonl"
        count = formatter.process_jsonl(
            str(ganjoor_hf_file), str(output), source_type="ganjoor"
        )
        stats["classical"] += count
        processed_files.append(str(output))

    # Wikisource
    wikisource_file = raw_dir / "wikisource" / "wikisource.jsonl"
    if wikisource_file.exists():
        output = processed_dir / "wikisource.jsonl"
        count = formatter.process_jsonl(
            str(wikisource_file), str(output), source_type="wikisource"
        )
        if count > 0:
            processed_files.append(str(output))

    # Ручные примеры
    manual_file = raw_dir / "manual" / "poems.jsonl"
    if manual_file.exists():
        output = processed_dir / "manual.jsonl"
        stats["manual"] = formatter.process_jsonl(
            str(manual_file), str(output), source_type="manual"
        )
        if stats["manual"] > 0:
            processed_files.append(str(output))

    # Объединяем
    if processed_files:
        combined = processed_dir / "combined.jsonl"
        stats["total"] = formatter.merge_datasets(processed_files, str(combined))
        formatter.create_train_val_split(str(combined), str(training_dir))

    return stats


if __name__ == "__main__":
    formatter = DatasetFormatter()

    test_poem = """Бӯи ҷӯи Мӯлиён ояд ҳаме,
Ёди ёри меҳрубон ояд ҳаме,
Реги Омуву дурушти роҳи ӯ,
Зери поям парниён ояд ҳаме."""

    example = formatter.format_example(
        poem_text=test_poem,
        form="rubaiyat",
        poet="Рӯдакӣ",
        themes=["homeland", "love"],
    )

    if example:
        print("Пример форматирования:\n")
        print(example.text)
        print(f"\nQuality score: {example.quality_score}")
