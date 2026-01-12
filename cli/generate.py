#!/usr/bin/env python3
"""
CLI интерфейс для генерации таджикских стихов

Примеры использования:
    python -m cli.generate "Рубоӣ бинавис"
    python -m cli.generate --form rubaiyat
    python -m cli.generate --interactive
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.generator import TajikPoetryGenerator, GenerationConfig


def main():
    parser = argparse.ArgumentParser(
        description="ShohnomaLLM - Генератор таджикских стихов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s "Рубоӣ бинавис"
  %(prog)s --form rubaiyat --theme "ишқ"
  %(prog)s -i  # интерактивный режим
        """
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Запрос для генерации (на таджикском)"
    )

    parser.add_argument(
        "-m", "--model",
        default="models/tajik-poetry-1.5b",
        help="Путь к модели (default: models/tajik-poetry-1.5b)"
    )

    parser.add_argument(
        "-f", "--form",
        choices=["rubaiyat", "ghazal", "qasida", "masnavi", "free"],
        help="Форма стиха"
    )

    parser.add_argument(
        "--theme",
        help="Тема стиха (на таджикском)"
    )

    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.8,
        help="Температура генерации (0.0-1.0, default: 0.8)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Максимальное количество токенов (default: 256)"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Интерактивный режим"
    )

    parser.add_argument(
        "--4bit",
        dest="load_4bit",
        action="store_true",
        help="Загрузить модель в 4-bit (экономия памяти)"
    )

    parser.add_argument(
        "-n", "--num",
        type=int,
        default=1,
        help="Количество стихов для генерации (default: 1)"
    )

    args = parser.parse_args()

    # Проверка аргументов
    if not args.interactive and not args.prompt and not args.form:
        parser.error("Укажите prompt, --form или используйте --interactive")

    # Загрузка модели
    print("Загрузка модели...")
    try:
        generator = TajikPoetryGenerator(
            args.model,
            load_in_4bit=args.load_4bit,
        )
        print("Модель загружена!\n")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("\nУбедитесь, что модель обучена и находится по пути:")
        print(f"  {args.model}")
        sys.exit(1)

    # Конфигурация генерации
    config = GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    # Интерактивный режим
    if args.interactive:
        run_interactive(generator, config)
        return

    # Формируем промпт
    if args.prompt:
        prompt = args.prompt
    else:
        # Генерируем промпт по форме и теме
        form_prompts = {
            "rubaiyat": "Рубоӣ бинавис",
            "ghazal": "Ғазал бинавис",
            "qasida": "Қасида бинавис",
            "masnavi": "Маснавӣ бинавис",
            "free": "Шеъри озод бинавис",
        }
        prompt = form_prompts.get(args.form, "Шеър бинавис")

        if args.theme:
            prompt = f"{prompt} дар бораи {args.theme}"

    # Генерация
    for i in range(args.num):
        if args.num > 1:
            print(f"\n{'='*50}")
            print(f"Стих {i+1}/{args.num}")
            print(f"{'='*50}")

        poem = generator.generate(prompt, form=args.form, config=config)
        print(poem)

        if args.num > 1:
            print()


def run_interactive(generator, config):
    """Интерактивный режим"""
    print("Интерактивный режим ShohnomaLLM")
    print("=" * 50)
    print("Команды:")
    print("  /rubaiyat - генерация рубаи")
    print("  /ghazal   - генерация газели")
    print("  /free     - свободный стих")
    print("  /help     - помощь")
    print("  /exit     - выход")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue

        # Команды
        if user_input.lower() in ['/exit', '/quit', '/q', 'exit', 'quit']:
            print("До свидания!")
            break

        if user_input.lower() == '/help':
            print_help()
            continue

        # Быстрые команды для форм
        form = None
        prompt = user_input

        if user_input.lower().startswith('/rubaiyat'):
            form = "rubaiyat"
            prompt = user_input[9:].strip() or "Рубоӣ бинавис"
        elif user_input.lower().startswith('/ghazal'):
            form = "ghazal"
            prompt = user_input[7:].strip() or "Ғазал бинавис"
        elif user_input.lower().startswith('/free'):
            form = "free"
            prompt = user_input[5:].strip() or "Шеъри озод бинавис"

        # Генерация
        print()
        poem = generator.generate(prompt, form=form, config=config)
        print(poem)
        print()


def print_help():
    """Вывод справки"""
    print("""
Справка ShohnomaLLM
==================

Просто введите запрос на таджикском языке, например:
  "Рубоӣ дар бораи баҳор бинавис"
  "Ғазали ошиқона эҷод кун"
  "Шеър дар бораи зиндагӣ"

Быстрые команды:
  /rubaiyat [тема] - генерация рубаи
  /ghazal [тема]   - генерация газели
  /free [тема]     - свободный стих
  /exit            - выход

Примеры промптов:
  "Рубоӣ бинавис" - напиши рубаи
  "Ғазал бинавис" - напиши газель
  "Шеъри озод бинавис" - напиши свободный стих
  "Шеър дар бораи ишқ бинавис" - напиши стих о любви

Темы (на таджикском):
  ишқ - любовь
  баҳор - весна
  табиат - природа
  зиндагӣ - жизнь
  ватан - родина
  дӯстӣ - дружба
""")


if __name__ == "__main__":
    main()
