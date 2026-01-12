"""
Форматирование датасета для обучения модели

Конвертирует собранные стихи в формат ChatML для Qwen2.5
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass


# Системный промпт на таджикском
SYSTEM_PROMPT = """Ту шоири тоҷикӣ ҳастӣ. Ту метавонӣ шеърҳои классикӣ (рубоӣ, ғазал, қасида, маснавӣ) ва шеърҳои озод бинависӣ. Ту услуби шоирони бузурги тоҷик ва форсро медонӣ: Рӯдакӣ, Ҳофиз, Саъдӣ, Хайём, Фирдавсӣ."""

# Перевод: "Ты таджикский поэт. Ты умеешь писать классические стихи
# (рубаи, газель, касыда, маснави) и свободные стихи. Ты знаешь стиль
# великих таджикских и персидских поэтов: Рудаки, Хафиз, Саади, Хайям, Фирдавси."


@dataclass
class FormattedExample:
    """Пример для обучения"""
    messages: List[Dict[str, str]]
    text: str  # Форматированный текст для SFT


class PromptGenerator:
    """Генератор разнообразных промптов для обучения"""

    # Промпты по формам стихов
    FORM_PROMPTS = {
        "rubaiyat": [
            "Рубоӣ бинавис",
            "Як рубоӣ эҷод кун",
            "Чор мисраъ дар шакли рубоӣ бинавис",
            "Рубоӣ дар бораи {theme} бинавис",
            "Рубоии зебо эҷод кун",
            "Рубоӣ бо қофияи ААБА бинавис",
        ],
        "ghazal": [
            "Ғазал бинавис",
            "Ғазали ошиқона эҷод кун",
            "Ғазал дар мавзӯи {theme} бисоз",
            "Ғазали зебо бинавис",
            "Ғазал бо радиф эҷод кун",
        ],
        "qasida": [
            "Қасида бинавис",
            "Қасида дар васфи {theme} эҷод кун",
            "Қасидаи мадҳия бинавис",
        ],
        "masnavi": [
            "Маснавӣ бинавис",
            "Маснавии кӯтоҳ эҷод кун",
            "Достон дар шакли маснавӣ бинавис",
        ],
        "free": [
            "Шеъри озод бинавис",
            "Шеъри муосир эҷод кун",
            "Шеър дар бораи {theme} бинавис",
            "Шеъри озод дар мавзӯи {theme}",
            "Шеър бинавис",
        ],
        "other": [
            "Шеър бинавис",
            "Шеъри зебо эҷод кун",
            "Абёти зебо бинавис",
        ],
    }

    # Темы на таджикском
    THEMES = {
        "love": "ишқ",
        "nature": "табиат",
        "spring": "баҳор",
        "life": "зиндагӣ",
        "death": "марг",
        "wisdom": "ҳикмат",
        "wine": "шароб",
        "beauty": "зебоӣ",
        "homeland": "ватан",
        "friendship": "дӯстӣ",
        "night": "шаб",
        "moon": "моҳ",
        "sun": "офтоб",
        "flower": "гул",
        "garden": "боғ",
        "heart": "дил",
        "soul": "ҷон",
        "time": "замон",
        "fate": "тақдир",
        "god": "Худо",
    }

    # Промпты в стиле конкретного поэта
    POET_STYLE_PROMPTS = [
        "Шеър дар услуби {poet} бинавис",
        "Ба тарзи {poet} шеър эҷод кун",
        "Мисли {poet} шеър бинавис",
    ]

    def generate_prompt(
        self,
        form: str,
        poet: Optional[str] = None,
        theme: Optional[str] = None,
    ) -> str:
        """
        Генерация случайного промпта.

        Args:
            form: Форма стиха (rubaiyat, ghazal, etc.)
            poet: Имя поэта (опционально)
            theme: Тема (опционально)

        Returns:
            Промпт на таджикском
        """
        prompts = self.FORM_PROMPTS.get(form, self.FORM_PROMPTS["other"])
        prompt = random.choice(prompts)

        # Подставляем тему
        if "{theme}" in prompt:
            if theme and theme in self.THEMES:
                theme_tj = self.THEMES[theme]
            else:
                # Случайная тема
                theme_tj = random.choice(list(self.THEMES.values()))
            prompt = prompt.replace("{theme}", theme_tj)

        # Иногда добавляем стиль поэта
        if poet and random.random() < 0.3:
            style_prompt = random.choice(self.POET_STYLE_PROMPTS)
            prompt = style_prompt.replace("{poet}", poet)

        return prompt


class DatasetFormatter:
    """Форматирование датасета для обучения"""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.prompt_generator = PromptGenerator()

    def format_example(
        self,
        poem_text: str,
        form: str,
        poet: Optional[str] = None,
        themes: Optional[List[str]] = None,
    ) -> FormattedExample:
        """
        Форматирование одного примера.

        Args:
            poem_text: Текст стиха
            form: Форма стиха
            poet: Автор
            themes: Темы

        Returns:
            FormattedExample
        """
        # Генерируем промпт
        theme = random.choice(themes) if themes else None
        user_prompt = self.prompt_generator.generate_prompt(form, poet, theme)

        # Формируем сообщения
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": poem_text},
        ]

        # Форматируем в ChatML (Qwen формат)
        text = self._format_chatml(messages)

        return FormattedExample(messages=messages, text=text)

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
    ) -> int:
        """
        Обработка JSONL файла с стихами.

        Args:
            input_path: Путь к входному файлу
            output_path: Путь к выходному файлу
            source_type: Тип источника (ganjoor/adabiyot)

        Returns:
            Количество обработанных примеров
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0

        with open(input_path, 'r', encoding='utf-8') as fin:
            with open(output_path, 'w', encoding='utf-8') as fout:
                for line in fin:
                    data = json.loads(line)

                    # Извлекаем текст в зависимости от источника
                    if source_type == "ganjoor":
                        poem_text = data.get("text_tajik", "")
                    else:
                        poem_text = data.get("text", "")

                    if not poem_text or len(poem_text) < 20:
                        continue

                    form = data.get("form", "other")
                    poet = data.get("poet")
                    themes = data.get("themes", [])

                    example = self.format_example(poem_text, form, poet, themes)

                    # Сохраняем
                    output_data = {
                        "text": example.text,
                        "messages": example.messages,
                    }
                    json.dump(output_data, fout, ensure_ascii=False)
                    fout.write("\n")
                    count += 1

        print(f"Обработано {count} примеров -> {output_path}")
        return count

    def create_train_val_split(
        self,
        input_path: str,
        output_dir: str,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple:
        """
        Разбиение на train/val.

        Args:
            input_path: Путь к JSONL файлу
            output_dir: Директория для сохранения
            val_ratio: Доля валидации
            seed: Сид для воспроизводимости

        Returns:
            (train_count, val_count)
        """
        random.seed(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Читаем все примеры
        examples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(line)

        # Перемешиваем
        random.shuffle(examples)

        # Разбиваем
        val_size = int(len(examples) * val_ratio)
        val_examples = examples[:val_size]
        train_examples = examples[val_size:]

        # Сохраняем
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"

        with open(train_path, 'w', encoding='utf-8') as f:
            f.writelines(train_examples)

        with open(val_path, 'w', encoding='utf-8') as f:
            f.writelines(val_examples)

        print(f"Train: {len(train_examples)} примеров -> {train_path}")
        print(f"Val: {len(val_examples)} примеров -> {val_path}")

        return len(train_examples), len(val_examples)

    def merge_datasets(
        self,
        input_paths: List[str],
        output_path: str,
    ) -> int:
        """
        Объединение нескольких датасетов.

        Args:
            input_paths: Список путей к JSONL файлам
            output_path: Путь к выходному файлу

        Returns:
            Общее количество примеров
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, 'w', encoding='utf-8') as fout:
            for input_path in input_paths:
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
    """
    Полная подготовка датасета.

    Args:
        raw_dir: Директория с сырыми данными
        processed_dir: Директория для обработанных данных
        training_dir: Директория для обучающих данных

    Returns:
        Статистика
    """
    formatter = DatasetFormatter()
    stats = {"classical": 0, "modern": 0, "total": 0}

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    training_dir = Path(training_dir)

    processed_files = []

    # Обрабатываем классику (Ganjoor)
    ganjoor_file = raw_dir / "ganjoor" / "all_classical.jsonl"
    if ganjoor_file.exists():
        output = processed_dir / "classical.jsonl"
        stats["classical"] = formatter.process_jsonl(
            str(ganjoor_file),
            str(output),
            source_type="ganjoor"
        )
        processed_files.append(str(output))

    # Обрабатываем современное (Adabiyot)
    adabiyot_file = raw_dir / "adabiyot" / "modern_poetry.jsonl"
    if adabiyot_file.exists():
        output = processed_dir / "modern.jsonl"
        stats["modern"] = formatter.process_jsonl(
            str(adabiyot_file),
            str(output),
            source_type="adabiyot"
        )
        processed_files.append(str(output))

    # Объединяем
    if processed_files:
        combined = processed_dir / "combined.jsonl"
        stats["total"] = formatter.merge_datasets(processed_files, str(combined))

        # Разбиваем на train/val
        formatter.create_train_val_split(str(combined), str(training_dir))

    return stats


if __name__ == "__main__":
    # Пример использования
    formatter = DatasetFormatter()

    # Тестовый пример
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

    print("Пример форматирования:\n")
    print(example.text)
