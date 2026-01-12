#!/usr/bin/env python3
"""
Загрузка датасета Ganjoor с Hugging Face и транслитерация в таджикскую кириллицу.

Датасет: https://huggingface.co/datasets/mabidan/ganjoor
Содержит 119,061 стихов от 203 поэтов.

Запуск:
    python scripts/download_ganjoor_hf.py
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.utils.transliterate import PersianToTajikTransliterator


def load_ganjoor_dataset():
    """Lazy import with a clear message for common Colab NumPy ABI issues."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        error_text = str(exc)
        if "numpy.dtype size changed" in error_text:
            raise RuntimeError(
                "Обнаружена бинарная несовместимость NumPy.\n"
                "В Colab обнови зависимости и перезапусти Runtime:\n"
                "  %pip install -U \"numpy>=1.26,<2.0\" \"pyarrow>=15,<18\" \"datasets>=2.18,<3.0\""
            ) from exc
        raise RuntimeError(f"Не удалось импортировать datasets: {error_text}") from exc
    return load_dataset("mabidan/ganjoor", split="train")


# Маппинг персидских имён поэтов в таджикские
POET_NAMES = {
    "حافظ": "Ҳофиз",
    "سعدی": "Саъдӣ",
    "مولوی": "Мавлавӣ",
    "فردوسی": "Фирдавсӣ",
    "خیام": "Хайём",
    "عطار": "Аттор",
    "نظامی": "Низомӣ",
    "جامی": "Ҷомӣ",
    "رودکی": "Рӯдакӣ",
    "سنایی": "Саноӣ",
    "انوری": "Анварӣ",
    "خاقانی": "Хоқонӣ",
    "ناصرخسرو": "Носирхусрав",
    "باباطاهر": "Боботоҳир",
    "صائب": "Соиб",
    "بیدل": "Бедил",
    "فرخی سیستانی": "Фаррухӣ",
    "منوچهری": "Манучеҳрӣ",
    "عنصری": "Унсурӣ",
    "ابوسعید ابوالخیر": "Абусаид",
    "شهریار": "Шаҳриёр",
    "پروین اعتصامی": "Парвин",
    "اقبال لاهوری": "Иқбол",
    "فروغ فرخزاد": "Фурӯғ",
    "سهراب سپهری": "Суҳроб",
    "احمد شاملو": "Шомлӯ",
}

# Категории для определения формы стиха
FORM_MAPPING = {
    "رباعیات": "rubaiyat",
    "رباعی": "rubaiyat",
    "غزلیات": "ghazal",
    "غزل": "ghazal",
    "قصاید": "qasida",
    "قصیده": "qasida",
    "مثنوی": "masnavi",
    "قطعات": "fragment",
    "دوبیتی": "rubaiyat",
}


def detect_form(category: str) -> str:
    """Определение формы стиха по категории"""
    if not category:
        return "other"
    for pattern, form in FORM_MAPPING.items():
        if pattern in category:
            return form
    return "other"


def transliterate_poet_name(persian_name: str, transliterator) -> str:
    """Транслитерация имени поэта"""
    # Сначала проверяем известные имена
    if persian_name in POET_NAMES:
        return POET_NAMES[persian_name]
    # Иначе транслитерируем
    return transliterator.transliterate(persian_name)


def main():
    output_dir = Path("data/raw/ganjoor_hf")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "all_poems.jsonl"

    print("=" * 60)
    print("Загрузка датасета Ganjoor с Hugging Face")
    print("=" * 60)

    # Загружаем датасет
    print("\nЗагрузка датасета (может занять несколько минут)...")
    dataset = load_ganjoor_dataset()
    print(f"Загружено: {len(dataset)} стихов")

    # Инициализируем транслитератор
    transliterator = PersianToTajikTransliterator()

    # Обрабатываем
    print("\nТранслитерация в таджикскую кириллицу...")

    valid_count = 0
    skipped_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc="Обработка"):
            # Пропускаем если нет текста
            text = item.get("text", "")
            if not text or not text.strip():
                skipped_count += 1
                continue

            # Транслитерируем текст
            text_tajik = transliterator.transliterate_poem(text)

            # Пропускаем слишком короткие
            if len(text_tajik) < 30:
                skipped_count += 1
                continue

            # Определяем форму
            form = detect_form(item.get("cat", ""))

            # Транслитерируем имя поэта
            poet_persian = item.get("poet", "")
            poet_tajik = transliterate_poet_name(poet_persian, transliterator)

            poem_data = {
                "id": f"hf_ganjoor_{item.get('id', valid_count)}",
                "poet": poet_tajik,
                "poet_persian": poet_persian,
                "title": item.get("poem", ""),
                "text_persian": text,
                "text_tajik": text_tajik,
                "form": form,
                "category": item.get("cat", ""),
                "source": "huggingface/mabidan/ganjoor",
            }

            json.dump(poem_data, f, ensure_ascii=False)
            f.write('\n')
            valid_count += 1

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ:")
    print(f"  Обработано: {valid_count} стихов")
    print(f"  Пропущено:  {skipped_count} (пустые или слишком короткие)")
    print(f"  Сохранено:  {output_file}")
    print("=" * 60)

    print("\nСледующий шаг:")
    print("  python scripts/prepare_dataset.py")

    return valid_count


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
