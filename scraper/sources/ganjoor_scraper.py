"""
Парсер для ganjoor.net - библиотеки персидской поэзии

Ganjoor содержит произведения классических поэтов:
- Рудаки, Хафиз, Саади, Хайям, Фирдавси, Руми, Джами и др.

Тексты на арабице, требуют транслитерации в кириллицу.

Особенности:
- Защита от дубликатов при перезапусках
- Автосохранение прогресса
- Возможность продолжить с места остановки
"""

import requests
import json
import time
from typing import Optional, Generator, Set, Dict
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm

from ..utils.transliterate import PersianToTajikTransliterator


@dataclass
class Poem:
    """Структура стихотворения"""
    id: str
    poet: str
    poet_id: int
    title: str
    text_persian: str           # Оригинал на арабице
    text_tajik: str             # Транслитерация в кириллицу
    form: str                   # rubaiyat, ghazal, qasida, masnavi, free
    url: str
    source: str = "ganjoor"


class GanjoorScraper:
    """
    Скрапер для Ganjoor API с защитой от дубликатов.

    API документация: https://api.ganjoor.net/swagger/index.html
    """

    BASE_URL = "https://api.ganjoor.net/api"

    # Поэты, чьи произведения относятся к таджикской/персидской традиции
    # Расширенный список для большего объёма данных
    POETS = {
        # Основные классики
        2: {"name": "رودکی", "tajik_name": "Рӯдакӣ", "era": "classical"},
        3: {"name": "فردوسی", "tajik_name": "Фирдавсӣ", "era": "classical"},
        5: {"name": "خیام", "tajik_name": "Хайём", "era": "classical"},
        7: {"name": "حافظ", "tajik_name": "Ҳофиз", "era": "classical"},
        22: {"name": "سعدی", "tajik_name": "Саъдӣ", "era": "classical"},
        26: {"name": "مولوی", "tajik_name": "Мавлавӣ", "era": "classical"},
        28: {"name": "عطار", "tajik_name": "Аттор", "era": "classical"},
        29: {"name": "نظامی", "tajik_name": "Низомӣ", "era": "classical"},
        35: {"name": "جامی", "tajik_name": "Ҷомӣ", "era": "classical"},

        # Дополнительные классики
        19: {"name": "سنایی", "tajik_name": "Саноӣ", "era": "classical"},
        20: {"name": "انوری", "tajik_name": "Анварӣ", "era": "classical"},
        21: {"name": "خاقانی", "tajik_name": "Хоқонӣ", "era": "classical"},
        24: {"name": "فرخی", "tajik_name": "Фаррухӣ", "era": "classical"},
        25: {"name": "عنصری", "tajik_name": "Унсурӣ", "era": "classical"},
        27: {"name": "منوچهری", "tajik_name": "Манучеҳрӣ", "era": "classical"},
        32: {"name": "ناصرخسرو", "tajik_name": "Носирхусрав", "era": "classical"},
        33: {"name": "باباطاهر", "tajik_name": "Боботоҳир", "era": "classical"},
        34: {"name": "ابوسعید", "tajik_name": "Абусаид", "era": "classical"},
        36: {"name": "هلالی", "tajik_name": "Ҳилолӣ", "era": "classical"},
        37: {"name": "کاتبی", "tajik_name": "Котибӣ", "era": "classical"},
        38: {"name": "اهلی", "tajik_name": "Аҳлӣ", "era": "classical"},
        39: {"name": "وحشی", "tajik_name": "Ваҳшӣ", "era": "classical"},
        40: {"name": "صائب", "tajik_name": "Соиб", "era": "classical"},
        41: {"name": "بیدل", "tajik_name": "Бедил", "era": "classical"},
        42: {"name": "کلیم", "tajik_name": "Калим", "era": "classical"},
    }

    # Определение формы стиха по категории
    FORM_MAPPING = {
        "رباعیات": "rubaiyat",
        "رباعی": "rubaiyat",
        "غزلیات": "ghazal",
        "غزل": "ghazal",
        "قصاید": "qasida",
        "قصیده": "qasida",
        "مثنوی": "masnavi",
        "قطعات": "fragment",
        "ترکیبات": "tarkib",
    }

    def __init__(self, output_dir: str = "data/raw/ganjoor"):
        """
        Args:
            output_dir: Директория для сохранения данных
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transliterator = PersianToTajikTransliterator()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ShohnomaLLM-DataCollector/1.0"
        })

        # Загружаем существующие ID для предотвращения дубликатов
        self.existing_ids: Set[str] = set()
        self.existing_poems: Dict[str, Poem] = {}
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """Загрузка существующих данных для предотвращения дубликатов"""
        all_file = self.output_dir / "all_classical.jsonl"

        if all_file.exists():
            print(f"Загрузка существующих данных из {all_file}...")
            try:
                with open(all_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            poem_id = data.get("id", "")
                            self.existing_ids.add(poem_id)
                            self.existing_poems[poem_id] = Poem(**data)
                print(f"Загружено {len(self.existing_ids)} существующих стихов")
            except Exception as e:
                print(f"Ошибка загрузки существующих данных: {e}")

    def _request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Выполнение запроса к API"""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка запроса {url}: {e}")
            return None

    def get_poet_info(self, poet_id: int) -> Optional[dict]:
        """Получение информации о поэте"""
        return self._request(f"ganjoor/poet/{poet_id}")

    def get_poet_categories(self, poet_id: int) -> Optional[list]:
        """Получение категорий (сборников) поэта"""
        data = self._request(f"ganjoor/poet/{poet_id}")
        if data and "cat" in data:
            return self._extract_categories(data["cat"])
        return None

    def _extract_categories(self, cat: dict, categories: list = None) -> list:
        """Рекурсивное извлечение категорий"""
        if categories is None:
            categories = []

        children = cat.get("children")
        if children:
            for child in children:
                categories.append({
                    "id": child["id"],
                    "title": child["title"],
                    "url": child.get("fullUrl", ""),
                })
                self._extract_categories(child, categories)

        return categories

    def get_poems_in_category(self, cat_id: int) -> Optional[list]:
        """Получение стихов в категории"""
        data = self._request(f"ganjoor/cat/{cat_id}?poems=true&mainSections=true")
        if data and "cat" in data and "poems" in data["cat"]:
            return data["cat"]["poems"]
        return None

    def get_poem_full(self, poem_id: int) -> Optional[dict]:
        """Получение полного текста стиха"""
        return self._request(f"ganjoor/poem/{poem_id}?verseDetails=true")

    def _detect_form(self, category_title: str, poem_title: str) -> str:
        """Определение формы стиха"""
        for pattern, form in self.FORM_MAPPING.items():
            if pattern in category_title:
                return form

        for pattern, form in self.FORM_MAPPING.items():
            if pattern in poem_title:
                return form

        return "other"

    def _extract_poem_text(self, verses: list) -> str:
        """Извлечение текста из списка строф"""
        lines = []
        for verse in verses:
            if "text" in verse:
                lines.append(verse["text"])
        return "\n".join(lines)

    def scrape_poet(self, poet_id: int, skip_existing: bool = True) -> Generator[Poem, None, None]:
        """
        Сбор всех стихов одного поэта.

        Args:
            poet_id: ID поэта в Ganjoor
            skip_existing: Пропускать уже скачанные стихи

        Yields:
            Poem объекты
        """
        poet_info = self.POETS.get(poet_id, {})
        poet_name = poet_info.get("tajik_name", f"Poet_{poet_id}")

        print(f"\nСбор стихов: {poet_name}")

        categories = self.get_poet_categories(poet_id)
        if not categories:
            print(f"Не найдены категории для поэта {poet_id}")
            return

        skipped = 0
        collected = 0

        # Прогресс-бар по категориям
        pbar = tqdm(categories, desc=f"{poet_name[:10]}", leave=False)

        for cat in pbar:
            # Показываем текущую категорию (коротко)
            cat_short = cat["title"][:20] if len(cat["title"]) > 20 else cat["title"]
            pbar.set_postfix_str(f"{cat_short} | +{collected}")

            poems = self.get_poems_in_category(cat["id"])
            if not poems:
                continue

            for poem_data in poems:
                poem_id = f"ganjoor_{poet_id}_{poem_data['id']}"

                # Пропускаем если уже есть
                if skip_existing and poem_id in self.existing_ids:
                    skipped += 1
                    continue

                # Получаем полный текст
                full_poem = self.get_poem_full(poem_data["id"])
                if not full_poem or "verses" not in full_poem:
                    continue

                text_persian = self._extract_poem_text(full_poem["verses"])
                if not text_persian.strip():
                    continue

                # Транслитерация
                text_tajik = self.transliterator.transliterate_poem(text_persian)

                # Определяем форму
                form = self._detect_form(cat["title"], poem_data.get("title", ""))

                poem = Poem(
                    id=poem_id,
                    poet=poet_name,
                    poet_id=poet_id,
                    title=poem_data.get("title", ""),
                    text_persian=text_persian,
                    text_tajik=text_tajik,
                    form=form,
                    url=poem_data.get("fullUrl", ""),
                )

                # Добавляем в существующие
                self.existing_ids.add(poem_id)
                self.existing_poems[poem_id] = poem
                collected += 1

                # Обновляем счётчик в прогресс-баре
                pbar.set_postfix_str(f"{cat_short} | +{collected}")

                yield poem

                # Небольшая задержка чтобы не нагружать API
                time.sleep(0.1)

        pbar.close()
        print(f"  {poet_name}: +{collected} новых, {skipped} пропущено")

    def scrape_all_poets(self) -> Generator[Poem, None, None]:
        """Сбор стихов всех поэтов"""
        for poet_id in self.POETS:
            yield from self.scrape_poet(poet_id)

    def save_poems(self, poems: list, filename: str = "poems.jsonl", append: bool = False) -> None:
        """
        Сохранение стихов в JSONL формате.

        Args:
            poems: Список Poem объектов
            filename: Имя файла
            append: Добавить к существующему файлу
        """
        filepath = self.output_dir / filename
        mode = 'a' if append else 'w'

        # Дедупликация по ID
        seen_ids = set()
        unique_poems = []
        for poem in poems:
            if poem.id not in seen_ids:
                seen_ids.add(poem.id)
                unique_poems.append(poem)

        with open(filepath, mode, encoding='utf-8') as f:
            for poem in unique_poems:
                json.dump(asdict(poem), f, ensure_ascii=False)
                f.write('\n')

        print(f"Сохранено {len(unique_poems)} стихов в {filepath}")

    def save_all(self, filename: str = "all_classical.jsonl") -> None:
        """
        Сохранение всех стихов (существующих + новых) без дубликатов.
        """
        filepath = self.output_dir / filename

        # Собираем все уникальные стихи
        all_poems = list(self.existing_poems.values())

        # Сортируем по ID для консистентности
        all_poems.sort(key=lambda p: p.id)

        with open(filepath, 'w', encoding='utf-8') as f:
            for poem in all_poems:
                json.dump(asdict(poem), f, ensure_ascii=False)
                f.write('\n')

        print(f"Сохранено всего {len(all_poems)} уникальных стихов в {filepath}")

    def collect_and_save(self, poet_ids: list = None, save_interval: int = 50) -> int:
        """
        Основной метод: сбор и сохранение стихов.

        Args:
            poet_ids: Список ID поэтов (None = все)
            save_interval: Сохранять каждые N новых стихов

        Returns:
            Количество новых собранных стихов
        """
        if poet_ids is None:
            poet_ids = list(self.POETS.keys())

        initial_count = len(self.existing_ids)
        new_count = 0
        unsaved_poems = []

        for poet_id in poet_ids:
            poet_name = self.POETS.get(poet_id, {}).get("tajik_name", f"poet_{poet_id}")

            for poem in self.scrape_poet(poet_id):
                new_count += 1
                unsaved_poems.append(poem)

                # Периодическое сохранение
                if len(unsaved_poems) >= save_interval:
                    print(f"\n  Автосохранение ({new_count} новых стихов)...")
                    self.save_all()
                    unsaved_poems = []

        # Финальное сохранение
        self.save_all()

        # Статистика
        final_count = len(self.existing_ids)
        print(f"\n{'='*50}")
        print(f"Было стихов:     {initial_count}")
        print(f"Новых стихов:    {new_count}")
        print(f"Всего стихов:    {final_count}")
        print(f"{'='*50}")

        return new_count

    def get_stats(self) -> dict:
        """Статистика по собранным данным"""
        stats = {
            "total": len(self.existing_poems),
            "by_poet": {},
            "by_form": {},
        }

        for poem in self.existing_poems.values():
            # По поэтам
            stats["by_poet"][poem.poet] = stats["by_poet"].get(poem.poet, 0) + 1
            # По формам
            stats["by_form"][poem.form] = stats["by_form"].get(poem.form, 0) + 1

        return stats

    def print_stats(self) -> None:
        """Вывод статистики"""
        stats = self.get_stats()

        print(f"\nСтатистика:")
        print(f"  Всего стихов: {stats['total']}")

        print(f"\n  По поэтам:")
        for poet, count in sorted(stats["by_poet"].items(), key=lambda x: -x[1]):
            print(f"    {poet}: {count}")

        print(f"\n  По формам:")
        for form, count in sorted(stats["by_form"].items(), key=lambda x: -x[1]):
            print(f"    {form}: {count}")


def main():
    """Пример использования"""
    scraper = GanjoorScraper()

    # Показываем существующую статистику
    if scraper.existing_ids:
        scraper.print_stats()

    # Собираем только рубаи Хайяма для теста
    print("\nТестовый сбор (Хайям)...")
    new_count = scraper.collect_and_save([5])  # ID Хайяма = 5

    print(f"\nСобрано новых: {new_count}")
    scraper.print_stats()


if __name__ == "__main__":
    main()
