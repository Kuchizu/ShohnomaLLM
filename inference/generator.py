"""
Генератор таджикских стихов

Основной модуль для inference обученной модели.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, List
from dataclasses import dataclass


# Системный промпт
SYSTEM_PROMPT = """Ту шоири тоҷикӣ ҳастӣ. Ту метавонӣ шеърҳои классикӣ (рубоӣ, ғазал, қасида, маснавӣ) ва шеърҳои озод бинависӣ. Ту услуби шоирони бузурги тоҷик ва форсро медонӣ: Рӯдакӣ, Ҳофиз, Саъдӣ, Хайём, Фирдавсӣ."""


@dataclass
class GenerationConfig:
    """Параметры генерации"""
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class TajikPoetryGenerator:
    """
    Генератор таджикских стихов на основе fine-tuned Qwen модели.

    Пример использования:
    ```python
    generator = TajikPoetryGenerator("models/tajik-poetry-1.5b")
    poem = generator.generate("Рубоӣ бинавис")
    print(poem)
    ```
    """

    # Инструкции по формам на таджикском
    FORM_INSTRUCTIONS = {
        'rubaiyat': 'Бинавис дар шакли рубоӣ (чор мисраъ, қофияи ААБА)',
        'ghazal': 'Бинавис дар шакли ғазал (байтҳо бо радиф)',
        'qasida': 'Бинавис дар шакли қасида',
        'masnavi': 'Бинавис дар шакли маснавӣ (ҷуфт-ҷуфт қофия)',
        'free': 'Бинавис дар шакли шеъри озод',
    }

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        """
        Инициализация генератора.

        Args:
            model_path: Путь к модели (локальный или HuggingFace Hub)
            device: Устройство ("auto", "cuda", "cpu")
            load_in_4bit: Загрузить в 4-bit для экономии памяти
            system_prompt: Системный промпт
        """
        self.system_prompt = system_prompt
        self.device = device

        # Конфигурация квантизации
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Загрузка модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map=device,
            quantization_config=quantization_config,
        )

        # Устанавливаем pad token если нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def generate(
        self,
        prompt: str,
        form: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Генерация стиха по запросу.

        Args:
            prompt: Запрос на таджикском (например, "Рубоӣ бинавис")
            form: Форма стиха (rubaiyat, ghazal, qasida, masnavi, free)
            config: Параметры генерации

        Returns:
            Сгенерированный стих
        """
        if config is None:
            config = GenerationConfig()

        # Добавляем инструкцию по форме если указана
        if form and form in self.FORM_INSTRUCTIONS:
            prompt = f"{prompt}. {self.FORM_INSTRUCTIONS[form]}"

        # Формируем сообщения
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Применяем chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Декодирование (только новые токены)
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return self._postprocess(response)

    def _postprocess(self, text: str) -> str:
        """Пост-обработка сгенерированного текста"""
        # Убираем лишние пробелы
        lines = [line.strip() for line in text.split('\n')]
        # Убираем пустые строки в начале и конце
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return '\n'.join(lines)

    def generate_rubaiyat(self, theme: Optional[str] = None) -> str:
        """Генерация рубаи"""
        prompt = "Рубоӣ бинавис"
        if theme:
            prompt = f"Рубоӣ дар бораи {theme} бинавис"
        return self.generate(prompt, form="rubaiyat")

    def generate_ghazal(self, theme: Optional[str] = None) -> str:
        """Генерация газели"""
        prompt = "Ғазал бинавис"
        if theme:
            prompt = f"Ғазал дар мавзӯи {theme} эҷод кун"
        return self.generate(prompt, form="ghazal")

    def generate_free_verse(self, theme: Optional[str] = None) -> str:
        """Генерация свободного стиха"""
        prompt = "Шеъри озод бинавис"
        if theme:
            prompt = f"Шеъри озод дар бораи {theme} бинавис"
        return self.generate(prompt, form="free")

    def generate_batch(
        self,
        prompts: List[str],
        form: Optional[str] = None,
    ) -> List[str]:
        """
        Пакетная генерация (для эффективности).

        Args:
            prompts: Список промптов
            form: Форма стиха (одна для всех)

        Returns:
            Список сгенерированных стихов
        """
        return [self.generate(prompt, form) for prompt in prompts]


def load_generator(
    model_path: str = "models/tajik-poetry-1.5b",
    **kwargs
) -> TajikPoetryGenerator:
    """
    Удобная функция для загрузки генератора.

    Args:
        model_path: Путь к модели
        **kwargs: Дополнительные аргументы для TajikPoetryGenerator

    Returns:
        TajikPoetryGenerator
    """
    return TajikPoetryGenerator(model_path, **kwargs)


# Пример использования
if __name__ == "__main__":
    print("ShohnomaLLM - Генератор таджикских стихов")
    print("=" * 50)

    # Для теста используем базовую модель (без fine-tuning)
    # В реальности используйте обученную модель
    try:
        generator = TajikPoetryGenerator(
            "Qwen/Qwen2.5-0.5B-Instruct",
            load_in_4bit=True,
        )

        prompts = [
            "Рубоӣ бинавис",
            "Ғазали ошиқона эҷод кун",
            "Шеър дар бораи баҳор бинавис",
        ]

        for prompt in prompts:
            print(f"\nЗапрос: {prompt}")
            print("-" * 40)
            poem = generator.generate(prompt)
            print(poem)
            print()

    except Exception as e:
        print(f"Ошибка: {e}")
        print("\nДля использования обученной модели:")
        print("  generator = TajikPoetryGenerator('models/tajik-poetry-1.5b')")
