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


# Системный промпт на персидском (арабица)
SYSTEM_PROMPT = """تو شاعر فارسی هستی. تو می‌توانی شعرهای کلاسیک و معاصر بنویسی.

انواع شعر:
- رباعی: چهار مصرع، قافیه AABA
- غزل: بیت‌ها با ردیف و قافیه
- قصیده: شعر بلند در مدح یا وصف
- مثنوی: جفت مصرع‌ها با قافیه AA BB CC

سبک شاعران بزرگ: رودکی، حافظ، سعدی، خیام، مولوی، جامی.

به زبان فارسی می‌نویسی. شعرهای تو زیبا، معنادار و پرحس هستند."""


# Примеры хороших стихов (персидский) - для возможного использования в будущем
QUALITY_EXAMPLES = {
    "rubaiyat": [
        "از آمدن و رفتن ما سودی نیست\nوز تار و پود هستی جز بادی نیست",  # Хайям
    ],
    "ghazal": [
        "دل می‌رود ز دستم صاحب‌دلان خدا را\nدردا که راز پنهان خواهد شد آشکارا",  # Хафиз
    ],
}


@dataclass
class FormattedExample:
    """Пример для обучения"""
    messages: List[Dict[str, str]]
    text: str
    quality_score: float = 1.0


class QualityFilter:
    """Фильтр качества стихов (персидский/арабица)"""

    # Минимальная длина текста
    MIN_LENGTH = 30

    # Максимальная длина
    MAX_LENGTH = 2000

    # Минимальное количество строк
    MIN_LINES = 2

    # Паттерны для обнаружения мусора
    GARBAGE_PATTERNS = [
        r'^[\s\d\.\,\:\;]+$',  # Только пробелы, цифры, пунктуация
        r'^[a-zA-Z\s]+$',  # Только латиница
        r'^[\u0400-\u04FF\s]+$',  # Только кириллица (нам нужна арабица)
    ]

    # Персидские буквы (должны присутствовать)
    PERSIAN_LETTERS = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')

    # Частые персидские слова
    COMMON_WORDS = {'و', 'که', 'از', 'به', 'در', 'با', 'من', 'تو', 'او', 'ما',
                    'دل', 'جان', 'عشق', 'یار', 'گل', 'شب', 'روز', 'ماه', 'آب',
                    'است', 'بود', 'شد', 'نیست', 'هست', 'باشد', 'کند', 'را', 'این', 'آن'}

    def is_quality_poem(self, text: str) -> tuple:
        """
        Проверка качества стиха (персидский/арабица).

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

        # Проверка что текст на арабице (персидский)
        persian_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(text)
        if total_chars > 0 and persian_chars / total_chars < 0.5:
            return False, 0.0, "not_persian"  # Слишком мало арабских букв

        # Подсчёт качества
        score = 1.0

        # Бонус за персидские буквы
        persian_letter_count = sum(1 for c in text if c in self.PERSIAN_LETTERS)
        if persian_letter_count > 0:
            score += min(0.2, persian_letter_count * 0.005)

        # Бонус за частые персидские слова
        words = set(re.findall(r'[\u0600-\u06FF]+', text))
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
    """Генератор разнообразных промптов для обучения (персидский)"""

    FORM_PROMPTS = {
        "rubaiyat": [
            "رباعی بنویس",
            "یک رباعی بساز",
            "چهار مصرع به شکل رباعی بنویس",
            "رباعی درباره {theme} بنویس",
            "رباعی زیبا بساز",
        ],
        "ghazal": [
            "غزل بنویس",
            "غزل عاشقانه بساز",
            "غزل درباره {theme} بنویس",
            "غزل زیبا بنویس",
            "یک غزل دلکش بساز",
        ],
        "qasida": [
            "قصیده بنویس",
            "قصیده در وصف {theme} بساز",
            "قصیده کوتاه بنویس",
        ],
        "masnavi": [
            "مثنوی بنویس",
            "مثنوی کوتاه بساز",
            "داستان به شکل مثنوی بنویس",
        ],
        "free": [
            "شعر بنویس",
            "شعر زیبا بساز",
            "شعر درباره {theme} بنویس",
        ],
        "fragment": [
            "قطعه بنویس",
            "چند بیت بساز",
        ],
        "other": [
            "شعر بنویس",
            "شعر زیبا بساز",
            "ابیات زیبا بنویس",
            "شعر درباره {theme} بنویس",
        ],
    }

    THEMES = {
        "love": "عشق", "nature": "طبیعت", "spring": "بهار",
        "life": "زندگی", "death": "مرگ", "wisdom": "حکمت",
        "wine": "شراب", "beauty": "زیبایی", "homeland": "وطن",
        "friendship": "دوستی", "night": "شب", "moon": "ماه",
        "sun": "آفتاب", "flower": "گل", "garden": "باغ",
        "heart": "دل", "soul": "جان", "time": "زمان",
        "fate": "تقدیر", "god": "خدا", "morning": "صبح",
        "autumn": "پاییز", "winter": "زمستان", "youth": "جوانی",
        "old_age": "پیری", "separation": "جدایی", "reunion": "وصال",
        "pain": "درد", "joy": "شادی", "tears": "اشک",
        "candle": "شمع", "butterfly": "پروانه", "nightingale": "بلبل",
    }

    POET_STYLE_PROMPTS = [
        "شعر به سبک {poet} بنویس",
        "به شیوه {poet} شعر بساز",
        "مثل {poet} {form} بنویس",
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
            form_name = {"rubaiyat": "رباعی", "ghazal": "غزل"}.get(form, "شعر")
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
        # Нормализация арабских символов в персидские
        # ي (U+064A Arabic) → ی (U+06CC Persian)
        # ك (U+0643 Arabic) → ک (U+06A9 Persian)
        # ە (U+06D5 Arabic) → ه (U+0647 Persian)
        text = text.replace('\u064a', '\u06cc')  # ي → ی
        text = text.replace('\u0643', '\u06a9')  # ك → ک
        text = text.replace('\u06d5', '\u0647')  # ە → ه
        # Также числа: ٤ → ۴ и т.д. (арабские → персидские)
        arabic_nums = '٠١٢٣٤٥٦٧٨٩'
        persian_nums = '۰۱۲۳۴۵۶۷۸۹'
        for ar, fa in zip(arabic_nums, persian_nums):
            text = text.replace(ar, fa)

        # Убираем невидимые Unicode символы (кроме ZWNJ который важен для персидского)
        # U+200B (Zero-Width Space), U+200D (ZWJ), U+FEFF (BOM), U+00AD (Soft Hyphen)
        text = re.sub(r'[\u200b\u200d\ufeff\u00ad]', '', text)
        # ZWNJ (U+200C) - оставляем, он важен для персидской типографики
        # Но убираем множественные ZWNJ подряд
        text = re.sub(r'\u200c+', '\u200c', text)
        # Нормализуем разные виды пробелов в обычный пробел
        text = re.sub(r'[\u00a0\u2000-\u200a\u202f\u205f]', ' ', text)
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
        deduplicate: bool = True,
    ) -> int:
        """Обработка JSONL файла с фильтрацией по качеству."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        total_lines = 0
        stats = {
            "duplicates": 0,
            "too_short": 0,
            "too_long": 0,
            "too_few_lines": 0,
            "not_persian": 0,
            "garbage_pattern": 0,
            "low_quality": 0,
            "empty": 0,
        }

        # Для дедупликации храним хеши текстов
        seen_hashes = set() if deduplicate else None

        # Считаем общее количество строк для прогресса
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        with open(input_path, 'r', encoding='utf-8') as fin:
            with open(output_path, 'w', encoding='utf-8') as fout:
                for i, line in enumerate(fin):
                    # Прогресс каждые 10000 строк
                    if (i + 1) % 10000 == 0:
                        print(f"  Прогресс: {i + 1}/{total_lines} ({100 * (i + 1) // total_lines}%)")

                    data = json.loads(line)

                    # Пробуем text_persian, затем text (для совместимости)
                    poem_text = data.get("text_persian") or data.get("text", "")

                    # Дедупликация по хешу текста
                    if deduplicate:
                        text_hash = hash(poem_text.strip())
                        if text_hash in seen_hashes:
                            stats["duplicates"] += 1
                            continue
                        seen_hashes.add(text_hash)

                    form = data.get("form", "other")
                    poet = data.get("poet")
                    themes = data.get("themes", [])

                    # Проверяем качество и получаем причину отказа
                    is_valid, quality_score, reason = self.quality_filter.is_quality_poem(poem_text)
                    if not is_valid:
                        if reason in stats:
                            stats[reason] += 1
                        continue

                    if quality_score < min_quality:
                        stats["low_quality"] += 1
                        continue

                    # Форматируем пример
                    example = self.format_example(poem_text, form, poet, themes)
                    if example is None:
                        continue

                    output_data = {
                        "text": example.text,
                        "messages": example.messages,
                        "quality_score": example.quality_score,
                    }
                    json.dump(output_data, fout, ensure_ascii=False)
                    fout.write("\n")
                    count += 1

        # Итоговая статистика
        print(f"\nОбработано: {count} из {total_lines}")
        print(f"Статистика отфильтрованных:")
        for reason, cnt in stats.items():
            if cnt > 0:
                print(f"  - {reason}: {cnt}")
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

    test_poem = """بوی جوی مولیان آید همی
یاد یار مهربان آید همی
ریگ آموی و درشتی راه او
زیر پایم پرنیان آید همی"""

    example = formatter.format_example(
        poem_text=test_poem,
        form="rubaiyat",
        poet="رودکی",
        themes=["homeland", "love"],
    )

    if example:
        print("Пример форматирования:\n")
        print(example.text)
        print(f"\nQuality score: {example.quality_score}")
