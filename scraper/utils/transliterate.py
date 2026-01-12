"""
Транслитератор персидской арабицы в таджикскую кириллицу

Этот модуль конвертирует тексты с персидского алфавита (арабица)
в таджикский кириллический алфавит.

Таджикский алфавит содержит специфичные буквы:
- Ғ ғ (гайн)
- Ӣ ӣ (и с макроном)
- Қ қ (каф)
- Ӯ ӯ (у с макроном)
- Ҳ ҳ (ха)
- Ҷ ҷ (джим)
"""

import re
from typing import Optional
import json
from pathlib import Path


class PersianToTajikTransliterator:
    """
    Контекстно-зависимая транслитерация персидского текста в таджикский.

    Использует правила транслитерации и словарь известных слов
    для точной конвертации.
    """

    # Базовая таблица транслитерации согласных
    CONSONANTS = {
        'ب': 'б',
        'پ': 'п',
        'ت': 'т',
        'ث': 'с',
        'ج': 'ҷ',
        'چ': 'ч',
        'ح': 'ҳ',
        'خ': 'х',
        'د': 'д',
        'ذ': 'з',
        'ر': 'р',
        'ز': 'з',
        'ژ': 'ж',
        'س': 'с',
        'ش': 'ш',
        'ص': 'с',
        'ض': 'з',
        'ط': 'т',
        'ظ': 'з',
        'ع': 'ъ',
        'غ': 'ғ',
        'ف': 'ф',
        'ق': 'қ',
        'ک': 'к',
        'ك': 'к',  # арабская каф
        'گ': 'г',
        'ل': 'л',
        'م': 'м',
        'ن': 'н',
        'ه': 'ҳ',  # будет обработано контекстно
        'ی': 'й',  # будет обработано контекстно
        'ي': 'й',  # арабская йа
    }

    # Гласные и их варианты
    VOWELS = {
        'ا': 'о',   # алиф - контекстно-зависимо
        'آ': 'о',   # алиф с мадда
        'و': 'в',   # вав - контекстно-зависимо (в/у/ӯ)
        'ی': 'и',   # йа - контекстно-зависимо (и/й)
        'ي': 'и',
        'ے': 'е',
        'ئ': 'ъ',
    }

    # Диакритические знаки (харакат)
    DIACRITICS = {
        'َ': 'а',   # фатха
        'ِ': 'и',   # касра
        'ُ': 'у',   # дамма
        'ّ': '',    # шадда (удвоение) - обрабатывается отдельно
        'ْ': '',    # сукун
        'ً': 'ан',  # танвин фатха
        'ٍ': 'ин',  # танвин касра
        'ٌ': 'ун',  # танвин дамма
    }

    # Специальные комбинации
    SPECIAL_COMBINATIONS = {
        'خو': 'хв',
        'خوا': 'хо',
        'وا': 'во',
        'ای': 'ои',
        'یا': 'ё',
    }

    # Частые слова с правильной транслитерацией
    WORD_DICTIONARY = {
        # Местоимения
        'من': 'ман',
        'تو': 'ту',
        'او': 'ӯ',
        'ما': 'мо',
        'شما': 'шумо',
        'ایشان': 'эшон',

        # Глаголы
        'است': 'аст',
        'بود': 'буд',
        'شد': 'шуд',
        'کرد': 'кард',
        'گفت': 'гуфт',
        'آمد': 'омад',
        'رفت': 'рафт',

        # Существительные
        'دل': 'дил',
        'جان': 'ҷон',
        'عشق': 'ишқ',
        'یار': 'ёр',
        'گل': 'гул',
        'باغ': 'боғ',
        'آب': 'об',
        'نان': 'нон',
        'شب': 'шаб',
        'روز': 'рӯз',
        'سر': 'сар',
        'چشم': 'чашм',
        'دست': 'даст',
        'پا': 'по',

        # Поэтические слова
        'بلبل': 'булбул',
        'صبا': 'сабо',
        'شراب': 'шароб',
        'ساقی': 'соқӣ',
        'مستی': 'мастӣ',
        'رباعی': 'рубоӣ',
        'غزل': 'ғазал',
        'قصیده': 'қасида',

        # Союзы и частицы
        'و': 'ва',
        'که': 'ки',
        'از': 'аз',
        'به': 'ба',
        'در': 'дар',
        'با': 'бо',
        'بر': 'бар',
        'تا': 'то',
        'چون': 'чун',
        'اگر': 'агар',
        'هم': 'ҳам',
        'نه': 'на',
        'یا': 'ё',

        # Числительные
        'یک': 'як',
        'دو': 'ду',
        'سه': 'се',
        'چهار': 'чор',
        'پنج': 'панҷ',
        'شش': 'шаш',
        'هفت': 'ҳафт',
        'هشت': 'ҳашт',
        'نه': 'нӯҳ',
        'ده': 'даҳ',
        'صد': 'сад',
        'هزار': 'ҳазор',

        # Прилагательные
        'خوب': 'хуб',
        'بد': 'бад',
        'زیبا': 'зебо',
        'بزرگ': 'бузург',
        'کوچک': 'хурд',
        'نو': 'нав',
        'کهن': 'куҳан',
    }

    # Имена известных поэтов
    POET_NAMES = {
        'رودکی': 'Рӯдакӣ',
        'فردوسی': 'Фирдавсӣ',
        'حافظ': 'Ҳофиз',
        'سعدی': 'Саъдӣ',
        'خیام': 'Хайём',
        'مولوی': 'Мавлавӣ',
        'رومی': 'Румӣ',
        'جامی': 'Ҷомӣ',
        'بیدل': 'Бедил',
        'عطار': 'Аттор',
        'نظامی': 'Низомӣ',
    }

    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Инициализация транслитератора.

        Args:
            dictionary_path: Путь к JSON-файлу с дополнительным словарём
        """
        self.custom_dictionary = {}
        if dictionary_path:
            self._load_dictionary(dictionary_path)

    def _load_dictionary(self, path: str) -> None:
        """Загрузка пользовательского словаря"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.custom_dictionary = json.load(f)
        except FileNotFoundError:
            pass

    def transliterate(self, text: str) -> str:
        """
        Основной метод транслитерации.

        Args:
            text: Текст на персидском (арабица)

        Returns:
            Текст на таджикском (кириллица)
        """
        if not text:
            return ""

        # Нормализация текста
        text = self._normalize(text)

        # Разбиение на слова с сохранением пунктуации
        tokens = self._tokenize(text)

        result = []
        for token in tokens:
            if self._is_persian_word(token):
                transliterated = self._transliterate_word(token)
                result.append(transliterated)
            else:
                # Пунктуация и пробелы остаются как есть
                result.append(token)

        return ''.join(result)

    def _normalize(self, text: str) -> str:
        """Нормализация Unicode символов"""
        # Замена арабских вариантов на персидские
        replacements = {
            'ي': 'ی',  # арабская йа → персидская
            'ك': 'ک',  # арабская каф → персидская
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _tokenize(self, text: str) -> list:
        """Разбиение текста на токены (слова и пунктуацию)"""
        # Регулярка для персидских слов
        pattern = r'([\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+|[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+)'
        return re.findall(pattern, text)

    def _is_persian_word(self, token: str) -> bool:
        """Проверка, является ли токен персидским словом"""
        return bool(re.match(r'^[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+$', token))

    def _transliterate_word(self, word: str) -> str:
        """Транслитерация одного слова"""
        # Сначала проверяем словари
        if word in self.custom_dictionary:
            return self.custom_dictionary[word]
        if word in self.POET_NAMES:
            return self.POET_NAMES[word]
        if word in self.WORD_DICTIONARY:
            return self.WORD_DICTIONARY[word]

        # Применяем правила транслитерации
        return self._apply_rules(word)

    def _apply_rules(self, word: str) -> str:
        """Применение правил транслитерации"""
        result = []
        i = 0
        word_len = len(word)

        while i < word_len:
            char = word[i]
            next_char = word[i + 1] if i + 1 < word_len else ''
            prev_char = word[i - 1] if i > 0 else ''

            # Проверка специальных комбинаций
            two_chars = char + next_char
            if two_chars in self.SPECIAL_COMBINATIONS:
                result.append(self.SPECIAL_COMBINATIONS[two_chars])
                i += 2
                continue

            # Обработка шадды (удвоение)
            if char == 'ّ' and result:
                last = result[-1]
                result.append(last)
                i += 1
                continue

            # Алиф в начале слова
            if char == 'ا':
                if i == 0:
                    # В начале слова - часто не произносится
                    result.append('')
                elif prev_char in self.CONSONANTS:
                    result.append('о')
                else:
                    result.append('а')
                i += 1
                continue

            # Алиф с мадда
            if char == 'آ':
                result.append('о')
                i += 1
                continue

            # Вав (в/у/ӯ)
            if char == 'و':
                if i == 0:
                    result.append('в')
                elif prev_char in self.CONSONANTS:
                    # После согласной - обычно 'у' или 'ӯ'
                    if next_char in self.CONSONANTS or next_char == '':
                        result.append('у')
                    else:
                        result.append('в')
                else:
                    result.append('в')
                i += 1
                continue

            # Йа (и/й/е)
            if char == 'ی' or char == 'ي':
                if i == word_len - 1:
                    # В конце слова - часто 'ӣ'
                    result.append('ӣ')
                elif prev_char in self.CONSONANTS:
                    result.append('и')
                else:
                    result.append('й')
                i += 1
                continue

            # Ха в конце слова
            if char == 'ه':
                if i == word_len - 1:
                    # В конце слова обычно '-а' или немое
                    if prev_char in self.CONSONANTS:
                        result.append('а')
                    else:
                        result.append('')
                else:
                    result.append('ҳ')
                i += 1
                continue

            # Айн
            if char == 'ع':
                if i == 0:
                    result.append('')  # В начале часто опускается
                else:
                    result.append('ъ')
                i += 1
                continue

            # Диакритические знаки
            if char in self.DIACRITICS:
                result.append(self.DIACRITICS[char])
                i += 1
                continue

            # Обычные согласные
            if char in self.CONSONANTS:
                result.append(self.CONSONANTS[char])
                i += 1
                continue

            # Если символ не распознан - оставляем как есть
            result.append(char)
            i += 1

        return ''.join(result)

    def transliterate_poem(self, poem: str) -> str:
        """
        Транслитерация стихотворения с сохранением форматирования.

        Args:
            poem: Стихотворение на персидском

        Returns:
            Стихотворение на таджикском
        """
        lines = poem.split('\n')
        transliterated_lines = [self.transliterate(line) for line in lines]
        return '\n'.join(transliterated_lines)

    def add_to_dictionary(self, persian: str, tajik: str) -> None:
        """
        Добавление слова в пользовательский словарь.

        Args:
            persian: Слово на персидском
            tajik: Правильная транслитерация
        """
        self.custom_dictionary[persian] = tajik

    def save_dictionary(self, path: str) -> None:
        """Сохранение пользовательского словаря"""
        all_words = {**self.WORD_DICTIONARY, **self.custom_dictionary}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_words, f, ensure_ascii=False, indent=2)


def transliterate_text(text: str) -> str:
    """
    Удобная функция для быстрой транслитерации.

    Args:
        text: Текст на персидском

    Returns:
        Текст на таджикском
    """
    transliterator = PersianToTajikTransliterator()
    return transliterator.transliterate(text)


# Примеры использования
if __name__ == "__main__":
    t = PersianToTajikTransliterator()

    # Тестовые примеры
    examples = [
        ("بوی جوی مولیان آید همی", "Бӯи ҷӯи Мӯлиён ояд ҳаме"),
        ("دل من همچو مرغ بی پر است", "Дили ман ҳамчу мурғи бепар аст"),
        ("عشق آمد و شد خون جگرم", "Ишқ омад ва шуд хуну ҷигарам"),
    ]

    print("Тестирование транслитератора:\n")
    for persian, expected in examples:
        result = t.transliterate(persian)
        print(f"Вход:     {persian}")
        print(f"Выход:    {result}")
        print(f"Ожидали:  {expected}")
        print("-" * 50)
