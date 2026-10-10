"""Herramientas de SteamPulse: consultas a la API pública de Steam (sin API key)."""

import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

TIMEOUT = 10
ZONA_MX = ZoneInfo("America/Mexico_City")


def _fecha_mx(timestamp: int | None) -> str | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=ZONA_MX).strftime("%d/%m/%Y %H:%M")

# Etiquetas (tags) de Steam por género. En Steam "terror" no es género sino etiqueta.
GENEROS = {
    "terror": ("Terror", 1667), "horror": ("Terror", 1667), "miedo": ("Terror", 1667),
    "rpg": ("RPG", 122), "rol": ("RPG", 122),
    "acción": ("Acción", 19), "accion": ("Acción", 19),
    "aventura": ("Aventura", 21),
    "estrategia": ("Estrategia", 9),
    "simulación": ("Simulación", 599), "simulacion": ("Simulación", 599), "simulador": ("Simulación", 599),
    "carreras": ("Carreras", 699),
    "deportes": ("Deportes", 701),
    "indie": ("Indie", 492),
    "casual": ("Casual", 597),
    "puzzle": ("Puzzle", 1664), "rompecabezas": ("Puzzle", 1664),
    "shooter": ("Shooter", 1774), "disparos": ("Shooter", 1774),
    "fps": ("FPS", 1663),
    "supervivencia": ("Supervivencia", 1662),
    "mundo abierto": ("Mundo abierto", 1695),
    "cooperativo": ("Cooperativo", 1685), "co-op": ("Cooperativo", 1685),
    "plataformas": ("Plataformas", 1625),
    "roguelike": ("Roguelike", 1716),
    "sandbox": ("Sandbox", 3810),
    "anime": ("Anime", 4085),
    "pelea": ("Pelea", 1743), "peleas": ("Pelea", 1743),
    "zombies": ("Zombis", 1659), "zombis": ("Zombis", 1659),
    "farmear": ("Granja", 87918), "granja": ("Granja", 87918), "farming": ("Granja", 87918),
    "cultivar": ("Granja", 87918), "sembrar": ("Granja", 87918),
    "construir": ("Construcción", 1643), "construcción": ("Construcción", 1643), "construccion": ("Construcción", 1643),
    "multijugador": ("Multijugador", 3859), "multiplayer": ("Multijugador", 3859),
    "espacio": ("Espacio", 1755), "espacial": ("Espacio", 1755),
    "buena historia": ("Buena historia", 1742), "con historia": ("Buena historia", 1742), "narrativos": ("Buena historia", 1742),
    "battle royale": ("Battle Royale", 176981),
    "souls": ("Souls-like", 29482), "soulslike": ("Souls-like", 29482), "souls-like": ("Souls-like", 29482),
    "medieval": ("Medieval", 4172),
    "carros": ("Carreras", 699), "autos": ("Carreras", 699), "coches": ("Carreras", 699),
    "sobrevivir": ("Supervivencia", 1662),
}


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
        "tipo": d.get("type"),
        "nombre": d.get("name"),
        "generos": [g["description"] for g in d.get("genres", [])],
        "descripcion": d.get("short_description", "")[:300],
        "desarrollador": (d.get("developers") or ["N/A"])[0],
        "precio": d.get("price_overview", {}).get("final_formatted", "Gratis"),
        "precio_original": d.get("price_overview", {}).get("initial_formatted") or None,
        "descuento": d.get("price_overview", {}).get("discount_percent", 0),
        "fecha_lanzamiento": d.get("release_date", {}).get("date", "N/A"),
        "categorias": [c["description"] for c in d.get("categories", [])],
        "idiomas": limpiar_texto(d.get("supported_languages"), 400).replace("*", "").replace(" idiomas con localización de audio", ""),
        "plataformas": [p for p, ok in (d.get("platforms") or {}).items() if ok],
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
        if detalle and detalle.get("tipo") in (None, "game"):
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
                "termina": _fecha_mx(item.get("discount_expiration")),
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


def get_review_summary(appid: int) -> dict | None:
    """Resumen de reseñas de Steam: % positivas y total."""
    resp = requests.get(
        f"https://store.steampowered.com/appreviews/{appid}",
        params={"json": 1, "language": "all", "purchase_type": "all", "num_per_page": 0},
        timeout=TIMEOUT,
    )
    if not resp.ok:
        return None
    q = resp.json().get("query_summary", {})
    total = q.get("total_reviews") or 0
    if not total:
        return None
    return {
        "porcentaje_positivas": round(q.get("total_positive", 0) * 100 / total),
        "total_resenas": total,
        "calificacion": q.get("review_score_desc"),
    }


def buscar_juegos(
    genero: str | None = None,
    solo_ofertas: bool = False,
    cantidad: int = 6,
    por_resenas: bool = False,
) -> dict:
    """Juegos más vendidos ahora (opcional: de un género, solo en oferta, o ordenados por reseñas)."""
    params = {"json": 1, "filter": "topsellers", "cc": "mx", "l": "spanish"}
    nombre_genero = None
    if genero:
        nombre_genero, tag_id = GENEROS[genero.lower()]
        params["tags"] = tag_id
    if solo_ofertas:
        params["specials"] = 1
    resp = requests.get("https://store.steampowered.com/search/results/", params=params, timeout=TIMEOUT)
    resp.raise_for_status()

    # Para ordenar por reseñas revisamos más candidatos y nos quedamos con los mejores
    candidatos = 12 if por_resenas else cantidad
    juegos = []
    for item in resp.json().get("items", []):
        m = re.search(r"/apps/(\d+)/", item.get("logo", ""))
        if not m:
            continue
        appid = int(m.group(1))
        detalle = get_game_details(appid)
        if detalle and detalle.get("tipo") in (None, "game"):  # fuera hardware, DLC y soundtracks
            juegos.append({
                "nombre": detalle["nombre"],
                "precio": detalle["precio"],
                "precio_original": detalle["precio_original"],
                "descuento": detalle["descuento"],
                "generos": detalle["generos"],
                "resenas": get_review_summary(appid),
            })
        if len(juegos) >= candidatos:
            break
        time.sleep(0.2)

    if por_resenas:
        juegos = [j for j in juegos if j["resenas"] and j["resenas"]["total_resenas"] >= 500]
        juegos.sort(key=lambda j: j["resenas"]["porcentaje_positivas"], reverse=True)

    return {
        "genero": nombre_genero,
        "solo_ofertas": solo_ofertas,
        "por_resenas": por_resenas,
        "juegos": juegos[:cantidad],
    }


def buscar_por_genero(genero: str, solo_ofertas: bool = False, max_results: int = 6) -> dict:
    """Compatibilidad: juegos más vendidos de un género."""
    return buscar_juegos(genero, solo_ofertas, max_results)


def get_current_players(appid: int) -> int | None:
    """Jugadores conectados ahora mismo a un juego."""
    resp = requests.get(
        "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/",
        params={"appid": appid},
        timeout=TIMEOUT,
    )
    if not resp.ok:
        return None
    return resp.json().get("response", {}).get("player_count")


def get_game_info(appid: int) -> dict | None:
    """Ficha de un juego: detalles de la tienda + jugadores conectados ahora."""
    detalle = get_game_details(appid)
    if not detalle:
        return None
    detalle["jugadores_ahora"] = get_current_players(appid)
    detalle["resenas"] = get_review_summary(appid)
    return detalle


def buscar_precios(nombre: str, max_results: int = 8) -> dict:
    """Precios de los juegos que coinciden con un nombre o saga (ej. Five Nights at Freddy's)."""
    resp = requests.get(
        "https://store.steampowered.com/api/storesearch/",
        params={"term": nombre, "l": "spanish", "cc": "mx"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    juegos = []
    for item in resp.json().get("items", [])[:max_results]:
        precio = item.get("price")
        if precio:
            final, inicial = precio.get("final", 0) / 100, precio.get("initial", 0) / 100
            juegos.append({
                "nombre": item.get("name"),
                "precio": f"${final:.2f} {precio.get('currency', '')}".strip(),
                "precio_original": f"${inicial:.2f}" if inicial > final else None,
                "descuento": round((1 - final / inicial) * 100) if inicial > final else 0,
            })
        else:
            juegos.append({"nombre": item.get("name"), "precio": "Gratis o sin precio", "precio_original": None, "descuento": 0})
    return {"buscado": nombre, "resultados": juegos}
