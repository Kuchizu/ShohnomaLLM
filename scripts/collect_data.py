#!/usr/bin/env python3
"""
Скрипт для локального сбора данных

Запуск:
    python scripts/collect_data.py --source ganjoor
    python scripts/collect_data.py --source wikisource
    python scripts/collect_data.py --source all
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.sources import GanjoorScraper, WikisourceTajikScraper, ManualDatasetBuilder


def collect_ganjoor(poets: list = None):
    """Сбор классической поэзии с ganjoor.net"""
    print("=" * 50)
    print("Сбор данных с ganjoor.net")
    print("=" * 50)

    scraper = GanjoorScraper("data/raw/ganjoor")

    # Используем все поэты из POETS (25 классиков) если не указаны конкретные
    if poets is None:
        poets = list(scraper.POETS.keys())
        print(f"Сбор от {len(poets)} поэтов...")

    total = scraper.collect_and_save(poets)
    print(f"\nВсего собрано: {total} стихов")
    return total


def collect_wikisource():
    """Сбор с таджикского Wikisource"""
    print("=" * 50)
    print("Сбор данных с tg.wikisource.org")
    print("=" * 50)

    scraper = WikisourceTajikScraper("data/raw/wikisource")
    total = scraper.collect_and_save()
    print(f"\nВсего собрано: {total} текстов")
    return total


def add_sample_poems():
    """Добавление примеров стихов вручную"""
    print("=" * 50)
    print("Добавление примеров стихов")
    print("=" * 50)

    builder = ManualDatasetBuilder("data/raw/manual/poems.jsonl")

    # Рубаи Рудаки
    builder.add_rubaiyat(
        """Бӯи ҷӯи Мӯлиён ояд ҳаме,
Ёди ёри меҳрубон ояд ҳаме,
Реги Омуву дурушти роҳи ӯ,
Зери поям парниён ояд ҳаме.""",
        poet="Рӯдакӣ"
    )

    # Рубаи Хайяма
    builder.add_rubaiyat(
        """Май хӯр, ки умр ҷовидона ин аст,
Мояи ҳосили замона ин аст,
Вақти гулу муле мусаффо биншин,
Хуш бошу бидон, ки вақт ин аст.""",
        poet="Хайём"
    )

    builder.add_rubaiyat(
        """Гар бар фалакам даст будӣ чун Яздон,
Бардоштаме ман ин фалакро зи миён,
Аз нав фалаке дигар чунон сохтаме,
К-озода ба коми дил расидӣ осон.""",
        poet="Хайём"
    )

    builder.save()
    print(f"Добавлено {len(builder.poems)} стихов")
    return len(builder.poems)


def main():
    parser = argparse.ArgumentParser(description="Сбор таджикской поэзии")
    parser.add_argument(
        "--source",
        choices=["ganjoor", "wikisource", "manual", "all"],
        default="all",
        help="Источник данных"
    )
    parser.add_argument(
        "--poets",
        nargs="+",
        type=int,
        default=None,
        help="ID поэтов для ganjoor (2=Рудаки, 5=Хайям, 7=Хафиз, 22=Саади)"
    )

    args = parser.parse_args()

    total = 0

    if args.source in ["ganjoor", "all"]:
        total += collect_ganjoor(args.poets)

    if args.source in ["wikisource", "all"]:
        total += collect_wikisource()

    if args.source in ["manual", "all"]:
        total += add_sample_poems()

    print("\n" + "=" * 50)
    print(f"ИТОГО СОБРАНО: {total}")
    print("=" * 50)
    print("\nДанные сохранены в data/raw/")
    print("Следующий шаг: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
