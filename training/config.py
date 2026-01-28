"""
Конфигурация обучения модели

Оптимизировано для Google Colab T4 GPU (15GB VRAM)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Конфигурация модели"""

    # Базовая модель
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Альтернативы:
    # - "Qwen/Qwen2.5-0.5B-Instruct"  # Меньше, быстрее
    # - "Qwen/Qwen2.5-3B-Instruct"    # Больше, качественнее (нужно A100)
    # - "google/gemma-2-2b-it"         # Альтернатива

    # Квантизация для обучения
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Максимальная длина последовательности
    max_seq_length: int = 1024  # Стихи обычно короткие


@dataclass
class LoRAConfig:
    """Конфигурация LoRA"""

    # Ранг LoRA (больше = больше параметров, лучше качество)
    r: int = 64  # Увеличено для лучшего качества

    # Alpha (обычно alpha = 2 * r)
    lora_alpha: int = 128

    # Dropout для регуляризации
    lora_dropout: float = 0.05

    # Целевые модули для LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])

    # Bias
    bias: str = "none"

    # Task type
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""

    # Пути
    output_dir: str = "outputs"
    train_data: str = "data/training/train.jsonl"
    val_data: str = "data/training/val.jsonl"

    # Гиперпараметры
    num_train_epochs: int = 5  # Увеличено для лучшего качества
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Эффективный batch size = 32

    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # Регуляризация
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Оптимизация памяти
    bf16: bool = True  # T4 поддерживает bfloat16
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"  # 8-bit Adam экономит память

    # Логирование и сохранение
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100  # Совпадает с eval_steps
    eval_strategy: str = "steps"  # evaluation_strategy deprecated
    save_strategy: str = "steps"  # Должен совпадать с eval для load_best_model
    save_total_limit: int = 3
    load_best_model_at_end: bool = True

    # Разное
    seed: int = 42
    dataloader_num_workers: int = 2
    report_to: str = "none"  # Или "wandb" для логирования


@dataclass
class InferenceConfig:
    """Конфигурация для inference"""

    # Генерация
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Остановка генерации
    do_sample: bool = True


@dataclass
class FullConfig:
    """Полная конфигурация"""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Название проекта
    project_name: str = "ShohnomaLLM"
    model_name: str = "tajik-poetry-1.5b"


# Предустановленные конфигурации

def get_colab_t4_config() -> FullConfig:
    """Конфигурация для Colab T4 (15GB VRAM)"""
    config = FullConfig()
    config.model.base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    config.model.load_in_4bit = True
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 8
    return config


def get_colab_a100_config() -> FullConfig:
    """Конфигурация для Colab A100 (40GB VRAM) - БЫСТРАЯ"""
    config = FullConfig()
    config.model.base_model = "Qwen/Qwen3-4B"  # Qwen3 лучше Qwen2.5!
    config.model.load_in_4bit = True  # 4-bit для большего batch size
    config.training.per_device_train_batch_size = 16  # Большой batch
    config.training.gradient_accumulation_steps = 2   # Меньше накопления = быстрее
    config.training.dataloader_num_workers = 4        # Параллельная загрузка
    config.training.gradient_checkpointing = False    # Выключаем - хватает VRAM
    config.lora.r = 64
    config.lora.lora_alpha = 128
    return config


def get_small_model_config() -> FullConfig:
    """Конфигурация для маленькой модели (экономия ресурсов)"""
    config = FullConfig()
    config.model.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    config.model.load_in_4bit = True
    config.training.per_device_train_batch_size = 8
    config.training.gradient_accumulation_steps = 4
    config.lora.r = 16
    config.lora.lora_alpha = 32
    config.model_name = "tajik-poetry-0.5b"
    return config


def get_config(preset: str = "colab_t4") -> FullConfig:
    """
    Получение конфигурации по имени пресета.

    Args:
        preset: "colab_t4", "colab_a100", "small"

    Returns:
        FullConfig
    """
    presets = {
        "colab_t4": get_colab_t4_config,
        "colab_a100": get_colab_a100_config,
        "small": get_small_model_config,
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    return presets[preset]()


if __name__ == "__main__":
    # Выводим конфигурацию
    config = get_colab_t4_config()

    print("ShohnomaLLM Training Configuration")
    print("=" * 50)
    print(f"\nModel: {config.model.base_model}")
    print(f"4-bit quantization: {config.model.load_in_4bit}")
    print(f"Max sequence length: {config.model.max_seq_length}")
    print(f"\nLoRA rank: {config.lora.r}")
    print(f"LoRA alpha: {config.lora.lora_alpha}")
    print(f"Target modules: {config.lora.target_modules}")
    print(f"\nBatch size: {config.training.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_train_epochs}")
