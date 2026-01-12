"""
REST API для генерации таджикских стихов

Запуск:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Документация API:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.generator import TajikPoetryGenerator, GenerationConfig


# Глобальная переменная для модели
generator: Optional[TajikPoetryGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте"""
    global generator

    model_path = "models/tajik-poetry-1.5b"
    print(f"Загрузка модели: {model_path}")

    try:
        generator = TajikPoetryGenerator(
            model_path,
            load_in_4bit=True,  # Экономим память
        )
        print("Модель загружена!")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        print("API запустится без модели")

    yield

    # Cleanup
    generator = None


app = FastAPI(
    title="ShohnomaLLM API",
    description="API для генерации стихов на таджикском языке",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модели данных

class GenerationRequest(BaseModel):
    """Запрос на генерацию"""
    prompt: str = Field(
        ...,
        description="Запрос на таджикском (например, 'Рубоӣ бинавис')",
        example="Рубоӣ дар бораи баҳор бинавис"
    )
    form: Optional[str] = Field(
        None,
        description="Форма стиха",
        example="rubaiyat"
    )
    temperature: Optional[float] = Field(
        0.8,
        ge=0.0,
        le=2.0,
        description="Температура генерации (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        256,
        ge=1,
        le=1024,
        description="Максимальное количество токенов"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Рубоӣ дар бораи баҳор бинавис",
                "form": "rubaiyat",
                "temperature": 0.8,
                "max_tokens": 256
            }
        }


class GenerationResponse(BaseModel):
    """Ответ с сгенерированным стихом"""
    poem: str = Field(..., description="Сгенерированный стих")
    form: Optional[str] = Field(None, description="Форма стиха")
    prompt: str = Field(..., description="Исходный запрос")


class BatchRequest(BaseModel):
    """Пакетный запрос"""
    prompts: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Список запросов (1-10)"
    )
    form: Optional[str] = None
    temperature: Optional[float] = 0.8


class BatchResponse(BaseModel):
    """Пакетный ответ"""
    poems: List[str]
    count: int


class HealthResponse(BaseModel):
    """Статус здоровья"""
    status: str
    model_loaded: bool
    version: str = "1.0.0"


# Эндпоинты

@app.get("/", tags=["Info"])
async def root():
    """Информация об API"""
    return {
        "name": "ShohnomaLLM API",
        "description": "API для генерации таджикских стихов",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Проверка здоровья сервиса"""
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None,
    )


@app.post("/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate_poem(request: GenerationRequest):
    """
    Генерация стиха по запросу.

    **Формы стихов:**
    - `rubaiyat` - рубаи (4 строки, схема AABA)
    - `ghazal` - газель (байты с радифом)
    - `qasida` - касыда
    - `masnavi` - маснави
    - `free` - свободный стих

    **Примеры запросов:**
    - "Рубоӣ бинавис" - напиши рубаи
    - "Ғазал бинавис" - напиши газель
    - "Шеър дар бораи ишқ бинавис" - напиши стих о любви
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Перезапустите сервер."
        )

    # Валидация формы
    valid_forms = ["rubaiyat", "ghazal", "qasida", "masnavi", "free", None]
    if request.form not in valid_forms:
        raise HTTPException(
            status_code=400,
            detail=f"Неверная форма. Допустимые: {valid_forms[:-1]}"
        )

    config = GenerationConfig(
        temperature=request.temperature,
        max_new_tokens=request.max_tokens,
    )

    try:
        poem = generator.generate(
            request.prompt,
            form=request.form,
            config=config,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации: {str(e)}"
        )

    return GenerationResponse(
        poem=poem,
        form=request.form,
        prompt=request.prompt,
    )


@app.post("/generate/batch", response_model=BatchResponse, tags=["Generation"])
async def generate_batch(request: BatchRequest):
    """
    Пакетная генерация стихов.

    Позволяет сгенерировать до 10 стихов за один запрос.
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    config = GenerationConfig(temperature=request.temperature)

    poems = []
    for prompt in request.prompts:
        try:
            poem = generator.generate(prompt, form=request.form, config=config)
            poems.append(poem)
        except Exception as e:
            poems.append(f"[Ошибка: {str(e)}]")

    return BatchResponse(poems=poems, count=len(poems))


@app.post("/generate/rubaiyat", response_model=GenerationResponse, tags=["Quick"])
async def generate_rubaiyat(theme: Optional[str] = None):
    """Быстрая генерация рубаи"""
    if generator is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    prompt = "Рубоӣ бинавис"
    if theme:
        prompt = f"Рубоӣ дар бораи {theme} бинавис"

    poem = generator.generate(prompt, form="rubaiyat")

    return GenerationResponse(poem=poem, form="rubaiyat", prompt=prompt)


@app.post("/generate/ghazal", response_model=GenerationResponse, tags=["Quick"])
async def generate_ghazal(theme: Optional[str] = None):
    """Быстрая генерация газели"""
    if generator is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    prompt = "Ғазал бинавис"
    if theme:
        prompt = f"Ғазал дар мавзӯи {theme} эҷод кун"

    poem = generator.generate(prompt, form="ghazal")

    return GenerationResponse(poem=poem, form="ghazal", prompt=prompt)


@app.get("/forms", tags=["Info"])
async def list_forms():
    """Список доступных форм стихов"""
    return {
        "forms": [
            {
                "id": "rubaiyat",
                "name": "Рубоӣ",
                "description": "Четверостишие с рифмой AABA",
                "example_prompt": "Рубоӣ бинавис"
            },
            {
                "id": "ghazal",
                "name": "Ғазал",
                "description": "Газель - лирическая форма с радифом",
                "example_prompt": "Ғазал бинавис"
            },
            {
                "id": "qasida",
                "name": "Қасида",
                "description": "Касыда - длинная ода",
                "example_prompt": "Қасида бинавис"
            },
            {
                "id": "masnavi",
                "name": "Маснавӣ",
                "description": "Маснави - двустишия с парной рифмой",
                "example_prompt": "Маснавӣ бинавис"
            },
            {
                "id": "free",
                "name": "Шеъри озод",
                "description": "Свободный стих без строгой формы",
                "example_prompt": "Шеъри озод бинавис"
            },
        ]
    }


@app.get("/themes", tags=["Info"])
async def list_themes():
    """Список популярных тем для стихов"""
    return {
        "themes": [
            {"tajik": "ишқ", "russian": "любовь", "english": "love"},
            {"tajik": "баҳор", "russian": "весна", "english": "spring"},
            {"tajik": "табиат", "russian": "природа", "english": "nature"},
            {"tajik": "зиндагӣ", "russian": "жизнь", "english": "life"},
            {"tajik": "ватан", "russian": "родина", "english": "homeland"},
            {"tajik": "дӯстӣ", "russian": "дружба", "english": "friendship"},
            {"tajik": "шаб", "russian": "ночь", "english": "night"},
            {"tajik": "моҳ", "russian": "луна", "english": "moon"},
            {"tajik": "гул", "russian": "цветок", "english": "flower"},
            {"tajik": "боғ", "russian": "сад", "english": "garden"},
            {"tajik": "дил", "russian": "сердце", "english": "heart"},
            {"tajik": "ҷон", "russian": "душа", "english": "soul"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
