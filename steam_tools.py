"""Herramientas de SteamPulse: consultas a la API pública de Steam (sin API key)."""

import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

TIMEOUT = 10


def limpiar_texto(texto: str | None, limite: int = 300) -> str:
    """Quita etiquetas HTML/BBCode y espacios extra de las noticias."""
    if not texto:
        return ""
    texto = html.unescape(texto)
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"\[/?[a-zA-Z0-9_=\"' ]*\]", " ", texto)
    texto = texto.replace("\\", " ")
    # Las notas de parche vienen pegadas ("kitsReduced"): separa las frases
    texto = re.sub(r"([a-z\)])([A-Z])", r"\1. \2", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:limite]


def buscar_appid(nombre_juego: str) -> dict | None:
    """Busca el AppID de un juego por nombre."""
    resp = requests.get(
        "https://store.steampowered.com/api/storesearch/",
        params={"term": nombre_juego, "l": "spanish", "cc": "mx"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if items:
        return {"appid": items[0]["id"], "nombre": items[0]["name"]}
    return None


def get_game_details(appid: int) -> dict | None:
    """Detalles de un juego desde la tienda de Steam."""
    resp = requests.get(
        "https://store.steampowered.com/api/appdetails",
        params={"appids": appid, "l": "spanish", "cc": "mx"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json().get(str(appid), {})
    if not data.get("success"):
        return None
    d = data["data"]
    return {
        "appid": appid,
        "nombre": d.get("name"),
        "generos": [g["description"] for g in d.get("genres", [])],
        "descripcion": d.get("short_description", "")[:300],
        "desarrollador": (d.get("developers") or ["N/A"])[0],
        "precio": d.get("price_overview", {}).get("final_formatted", "Gratis"),
        "fecha_lanzamiento": d.get("release_date", {}).get("date", "N/A"),
        "categorias": [c["description"] for c in d.get("categories", [])[:5]],
    }


def get_trending_games(limit: int = 10) -> list[dict]:
    """Juegos más jugados en Steam ahora mismo."""
    resp = requests.get(
        "https://api.steampowered.com/ISteamChartsService/GetMostPlayedGames/v1/",
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    ranks = resp.json()["response"]["ranks"][:limit]

    juegos = []
    for j in ranks:
        detalle = get_game_details(j["appid"])
        if detalle:
            juegos.append({
                "rank": j["rank"],
                "appid": j["appid"],
                "nombre": detalle["nombre"],
                "generos": detalle["generos"],
                "precio": detalle["precio"],
                "peak_jugadores": j.get("peak_in_game"),
            })
        time.sleep(0.3)
    return juegos


def get_best_deals(max_results: int = 10) -> list[dict]:
    """Mejores ofertas activas en Steam (precios en MXN)."""
    resp = requests.get(
        "https://store.steampowered.com/api/featuredcategories",
        params={"cc": "mx", "l": "spanish"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    specials = resp.json().get("specials", {}).get("items", [])

    ofertas = []
    for item in specials[:max_results]:
        descuento = item.get("discount_percent", 0)
        if descuento > 0:
            original = item.get("original_price", 0) / 100
            final = item.get("final_price", 0) / 100
            ofertas.append({
                "appid": item.get("id"),
                "nombre": item.get("name"),
                "precio_original": f"${original:.2f}",
                "precio_final": f"${final:.2f}",
                "descuento": f"{descuento}%",
                "ahorro": f"${original - final:.2f}",
            })
    return ofertas


def compare_games(appid1: int, appid2: int) -> dict:
    """Compara dos juegos por sus características en la tienda."""
    juego1 = get_game_details(appid1)
    time.sleep(0.5)
    juego2 = get_game_details(appid2)
    return {"juego1": juego1, "juego2": juego2}


def get_steam_news(appid: int | None = None, max_results: int = 5) -> list[dict]:
    """Noticias recientes de un juego, o de Steam en general si no hay appid."""
    if appid:
        resp = requests.get(
            "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/",
            params={
                "appid": appid,
                "count": max_results,
                "maxlength": 600,
                "feeds": "steam_community_announcements",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        noticias = []
        for item in resp.json().get("appnews", {}).get("newsitems", []):
            fecha = item.get("date")
            noticias.append({
                "titulo": item.get("title"),
                "contenido": limpiar_texto(item.get("contents")),
                "fecha": (
                    datetime.fromtimestamp(fecha, tz=timezone.utc).strftime("%Y-%m-%d")
                    if fecha else None
                ),
                "url": item.get("url"),
            })
        return noticias

    resp = requests.get(
        "https://store.steampowered.com/feeds/news/",
        params={"l": "spanish"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    return [
        {
            "titulo": item.findtext("title"),
            "contenido": limpiar_texto(item.findtext("description")),
            "fecha": item.findtext("pubDate"),
            "url": item.findtext("link"),
        }
        for item in root.findall(".//item")[:max_results]
    ]