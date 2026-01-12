#!/usr/bin/env python3
"""
Подготовка датасета для обучения

Запуск:
    python scripts/prepare_dataset.py

После этого загрузите data/training/ в Google Drive для обучения в Colab.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.format_dataset import DatasetFormatter, prepare_full_dataset


def main():
    print("=" * 50)
    print("Подготовка датасета для обучения")
    print("=" * 50)

    stats = prepare_full_dataset(
        raw_dir="data/raw",
        processed_dir="data/processed",
        training_dir="data/training",
    )

    print("\n" + "=" * 50)
    print("СТАТИСТИКА:")
    print(f"  Классические стихи: {stats.get('classical', 0)}")
    print(f"  Современные стихи:  {stats.get('modern', 0)}")
    print(f"  ВСЕГО:              {stats.get('total', 0)}")
    print("=" * 50)

    print("\nГотово! Файлы для обучения:")
    print("  data/training/train.jsonl")
    print("  data/training/val.jsonl")
    print("\nЗагрузите папку data/training/ в Google Drive")
    print("и откройте notebooks/03_training.ipynb в Colab")


if __name__ == "__main__":
    main()
