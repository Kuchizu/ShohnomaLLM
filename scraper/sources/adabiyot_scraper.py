"""
Парсеры для таджикской поэзии из различных источников

Источники:
1. tg.wikisource.org — таджикский Wikisource
2. Локальные файлы (TXT, PDF)
3. Ручной ввод
"""

import requests
import json
import time
import re
from typing import Optional, Generator, List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm


@dataclass
class Poem:
    """Структура стихотворения"""
    id: str
    poet: str
    title: str
    text: str
    form: str           # rubaiyat, ghazal, free
    source: str
    url: str = ""


class WikisourceTajikScraper:
    """
    Парсер для таджикского Wikisource (tg.wikisource.org)

    Wikisource содержит тексты в общественном достоянии,
    включая произведения таджикских поэтов.
    """

    BASE_URL = "https://tg.wikisource.org"
    API_URL = "https://tg.wikisource.org/w/api.php"

    def __init__(self, output_dir: str = "data/raw/wikisource"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ShohnomaLLM-DataCollector/1.0"
        })

    def search(self, query: str, limit: int = 100) -> List[dict]:
        """
        Поиск страниц по запросу.

        Args:
            query: Поисковый запрос (например, "шеър", "ғазал", "рубоӣ")
            limit: Максимальное количество результатов
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }

        try:
            response = self.session.get(self.API_URL, params=params, timeout=30)
            data = response.json()
            return data.get("query", {}).get("search", [])
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

    def get_page_content(self, title: str) -> Optional[str]:
        """Получение содержимого страницы"""
        params = {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
        }

        try:
            response = self.session.get(self.API_URL, params=params, timeout=30)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if page_id != "-1":
                    revisions = page_data.get("revisions", [])
                    if revisions:
                        slots = revisions[0].get("slots", {})
                        main = slots.get("main", {})
                        return main.get("*", "")
        except Exception as e:
            print(f"Ошибка получения страницы {title}: {e}")

        return None

    def get_category_pages(self, category: str) -> List[str]:
        """Получение страниц из категории"""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": 500,
            "format": "json",
        }

        try:
            response = self.session.get(self.API_URL, params=params, timeout=30)
            data = response.json()
            members = data.get("query", {}).get("categorymembers", [])
            return [m["title"] for m in members]
        except Exception as e:
            print(f"Ошибка получения категории: {e}")
            return []

    def _clean_wikitext(self, text: str) -> str:
        """Очистка wiki-разметки"""
        # Убираем шаблоны {{...}}
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        # Убираем ссылки [[...|text]] -> text
        text = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', text)
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        # Убираем HTML теги
        text = re.sub(r'<[^>]+>', '', text)
        # Убираем категории
        text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
        # Убираем лишние пробелы
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _detect_form(self, text: str, title: str) -> str:
        """Определение формы стиха"""
        lines = [l for l in text.split('\n') if l.strip()]

        # Рубаи - 4 строки
        if len(lines) == 4:
            return "rubaiyat"

        # Проверка по заголовку
        title_lower = title.lower()
        if 'рубоӣ' in title_lower or 'ruboiy' in title_lower:
            return "rubaiyat"
        if 'ғазал' in title_lower or 'ghazal' in title_lower:
            return "ghazal"
        if 'қасида' in title_lower:
            return "qasida"

        # По количеству строк
        if len(lines) >= 8 and len(lines) <= 30:
            return "ghazal"

        return "other"

    def scrape_poems(self, search_queries: List[str] = None) -> Generator[Poem, None, None]:
        """
        Сбор стихов из Wikisource.

        Args:
            search_queries: Список поисковых запросов
        """
        if search_queries is None:
            search_queries = ["шеър", "ғазал", "рубоӣ", "қасида"]

        seen_titles = set()

        for query in search_queries:
            print(f"\nПоиск: {query}")
            results = self.search(query)

            for result in tqdm(results, desc=query):
                title = result.get("title", "")

                if title in seen_titles:
                    continue
                seen_titles.add(title)

                content = self.get_page_content(title)
                if not content:
                    continue

                text = self._clean_wikitext(content)
                if len(text) < 20:
                    continue

                poem = Poem(
                    id=f"wikisource_{len(seen_titles)}",
                    poet="",  # Может быть извлечено из title
                    title=title,
                    text=text,
                    form=self._detect_form(text, title),
                    source="wikisource",
                    url=f"{self.BASE_URL}/wiki/{title.replace(' ', '_')}",
                )

                yield poem
                time.sleep(0.2)

    def save_poems(self, poems: List[Poem], filename: str = "wikisource.jsonl"):
        """Сохранение в JSONL"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for poem in poems:
                json.dump(asdict(poem), f, ensure_ascii=False)
                f.write('\n')
        print(f"Сохранено {len(poems)} текстов -> {filepath}")

    def collect_and_save(self) -> int:
        """Полный сбор и сохранение"""
        poems = list(self.scrape_poems())
        self.save_poems(poems)
        return len(poems)


class LocalFileScraper:
    """
    Загрузчик стихов из локальных файлов.

    Поддерживает:
    - TXT файлы (один стих на файл или разделённые ---)
    - JSON/JSONL файлы
    """

    def __init__(self, output_dir: str = "data/raw/local"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_txt_file(self, filepath: str, poet: str = "") -> List[Poem]:
        """
        Загрузка из TXT файла.

        Формат: стихи разделены строкой "---"
        """
        poems = []
        path = Path(filepath)

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Разделяем по ---
        parts = content.split('---')

        for i, part in enumerate(parts):
            text = part.strip()
            if len(text) < 10:
                continue

            poems.append(Poem(
                id=f"local_{path.stem}_{i}",
                poet=poet,
                title=f"{path.stem}_{i}",
                text=text,
                form=self._detect_form(text),
                source="local",
            ))

        return poems

    def load_directory(self, directory: str, poet: str = "") -> List[Poem]:
        """Загрузка всех TXT файлов из директории"""
        poems = []
        dir_path = Path(directory)

        for txt_file in dir_path.glob("*.txt"):
            poems.extend(self.load_txt_file(str(txt_file), poet))

        return poems

    def load_jsonl(self, filepath: str) -> List[Poem]:
        """Загрузка из JSONL файла"""
        poems = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                poems.append(Poem(
                    id=data.get("id", f"jsonl_{len(poems)}"),
                    poet=data.get("poet", ""),
                    title=data.get("title", ""),
                    text=data.get("text", ""),
                    form=data.get("form", "other"),
                    source=data.get("source", "jsonl"),
                    url=data.get("url", ""),
                ))

        return poems

    def _detect_form(self, text: str) -> str:
        """Определение формы по тексту"""
        lines = [l for l in text.split('\n') if l.strip()]
        if len(lines) == 4:
            return "rubaiyat"
        if 8 <= len(lines) <= 30:
            return "ghazal"
        return "other"

    def save_poems(self, poems: List[Poem], filename: str = "local.jsonl"):
        """Сохранение в JSONL"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for poem in poems:
                json.dump(asdict(poem), f, ensure_ascii=False)
                f.write('\n')
        print(f"Сохранено {len(poems)} -> {filepath}")


class ManualDatasetBuilder:
    """
    Ручное создание датасета.

    Позволяет добавлять стихи программно.
    """

    def __init__(self, output_path: str = "data/raw/manual/poems.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.poems: List[Poem] = []

    def add_poem(
        self,
        text: str,
        poet: str = "",
        title: str = "",
        form: str = "other",
    ) -> None:
        """Добавление стиха"""
        poem = Poem(
            id=f"manual_{len(self.poems)}",
            poet=poet,
            title=title,
            text=text.strip(),
            form=form,
            source="manual",
        )
        self.poems.append(poem)

    def add_rubaiyat(self, text: str, poet: str = "") -> None:
        """Добавление рубаи"""
        self.add_poem(text, poet=poet, form="rubaiyat")

    def add_ghazal(self, text: str, poet: str = "") -> None:
        """Добавление газели"""
        self.add_poem(text, poet=poet, form="ghazal")

    def save(self) -> None:
        """Сохранение датасета"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for poem in self.poems:
                json.dump(asdict(poem), f, ensure_ascii=False)
                f.write('\n')
        print(f"Сохранено {len(self.poems)} стихов -> {self.output_path}")

    def load_existing(self) -> None:
        """Загрузка существующего датасета для дополнения"""
        if self.output_path.exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.poems.append(Poem(**data))
            print(f"Загружено {len(self.poems)} существующих стихов")


# Пример использования
if __name__ == "__main__":
    # Пример ручного добавления стихов
    builder = ManualDatasetBuilder()

    # Добавляем примеры (классические рубаи)
    builder.add_rubaiyat(
        """Бӯи ҷӯи Мӯлиён ояд ҳаме,
Ёди ёри меҳрубон ояд ҳаме,
Реги Омуву дурушти роҳи ӯ,
Зери поям парниён ояд ҳаме.""",
        poet="Рӯдакӣ"
    )

    builder.add_rubaiyat(
        """Ин ҷаҳон кӯҳест ва кирдори мо
Нидо, ҳар чи гӯем бозояд садо,
Некӣ кунад некӣ расад бар ҷояш,
Бадӣ кунад бадӣ расад бар ҷояш.""",
        poet="Халқӣ"
    )

    builder.save()

    print("\n" + "="*50)
    print("Для сбора из Wikisource:")
    print("  scraper = WikisourceTajikScraper()")
    print("  scraper.collect_and_save()")
