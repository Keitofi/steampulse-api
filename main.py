"""API de SteamPulse con FastAPI."""

from dotenv import load_dotenv

load_dotenv()

import os

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import steampulse
from limits import PREGUNTAS_POR_MINUTO, estado_llm, permitir_pregunta
from steam_tools import (
    buscar_appid,
    compare_games,
    get_best_deals,
    get_steam_news,
    get_trending_games,
)

app = FastAPI(
    title="SteamPulse API",
    description="Agente de inteligencia gaming en tiempo real con datos de Steam.",
)


class AskRequest(BaseModel):
    pregunta: str
    contexto: dict | None = None  # pregunta anterior: {pregunta, tool, params}


def _steam(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando Steam: {type(exc).__name__}")


INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_configurada": bool(os.getenv("API_KEY")),
        "llm": estado_llm(),
        "preguntas_por_minuto": PREGUNTAS_POR_MINUTO,
    }


def _cliente(request: Request) -> str:
    # Render pasa la IP real del usuario en X-Forwarded-For
    reenviada = request.headers.get("x-forwarded-for", "")
    return reenviada.split(",")[0].strip() or (request.client.host if request.client else "desconocido")


@app.get("/trending")
def trending():
    return _steam(get_trending_games)


@app.get("/deals")
def deals():
    return _steam(get_best_deals)


@app.get("/compare")
def compare(juego1: str = Query(..., examples=["Palworld"]), juego2: str = Query(..., examples=["Elden Ring"])):
    j1 = _steam(buscar_appid, juego1)
    j2 = _steam(buscar_appid, juego2)
    if not j1 or not j2:
        raise HTTPException(status_code=404, detail="No encontré uno de los juegos")
    return _steam(compare_games, j1["appid"], j2["appid"])


@app.get("/news")
def news(juego: str | None = None):
    appid = None
    if juego:
        encontrado = _steam(buscar_appid, juego)
        if not encontrado:
            raise HTTPException(status_code=404, detail="No encontré ese juego")
        appid = encontrado["appid"]
    return _steam(get_steam_news, appid=appid)


@app.post("/ask")
def ask(request: AskRequest, http: Request):
    if not request.pregunta.strip():
        raise HTTPException(status_code=422, detail="La pregunta está vacía")
    if len(request.pregunta) > 500:
        raise HTTPException(status_code=422, detail="La pregunta es muy larga (máx. 500 caracteres)")
    permitido, espera = permitir_pregunta(_cliente(http))
    if not permitido:
        raise HTTPException(
            status_code=429,
            detail=f"Hiciste muchas preguntas seguidas. Espera {espera} segundos e intenta de nuevo.",
        )
    return steampulse(request.pregunta, request.contexto)
