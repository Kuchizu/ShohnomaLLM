"""
Парсеры для различных источников поэзии
"""
from .ganjoor_scraper import GanjoorScraper
from .adabiyot_scraper import WikisourceTajikScraper, LocalFileScraper, ManualDatasetBuilder

__all__ = [
    "GanjoorScraper",           # Классика с ganjoor.net (арабица -> кириллица)
    "WikisourceTajikScraper",   # tg.wikisource.org
    "LocalFileScraper",         # Локальные TXT/JSONL файлы
    "ManualDatasetBuilder",     # Ручное добавление стихов
]
