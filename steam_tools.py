"""Herramientas de SteamPulse: consultas a la API pública de Steam (sin API key)."""

import html
import json
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
    "gratis": ("Gratis", 113),
    "gratuito": ("Gratis", 113),
    "free to play": ("Gratis", 113),
}


def limpiar_texto(texto: str | None, limite: int = 300) -> str:
    """Quita etiquetas HTML/BBCode y espacios extra de las noticias."""
    if not texto:
        return ""
    texto = html.unescape(texto)
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"\[/?[a-zA-Z0-9_=\"' ]*\]", " ", texto)
    # Imágenes y videos incrustados en los anuncios: {STEAM_CLAN_IMAGE}/123/abc.png, links a .png/.jpg/.gif
    texto = re.sub(r"\{STEAM_[A-Z_]+\}\S*", " ", texto)
    texto = re.sub(r"https?://\S+?\.(?:png|jpe?g|gif|webp|mp4|webm)\S*", " ", texto, flags=re.IGNORECASE)
    texto = texto.replace("\\", " ")
    # Las notas de parche vienen pegadas ("kitsReduced"): separa las frases
    texto = re.sub(r"([a-z\)])([A-Z])", r"\1. \2", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:limite]


def buscar_candidatos(nombre_juego: str) -> list[dict]:
    """Todos los resultados de la búsqueda de la tienda (para elegir el que mejor coincide)."""
    resp = requests.get(
        "https://store.steampowered.com/api/storesearch/",
        params={"term": nombre_juego, "l": "spanish", "cc": "mx"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return [{"appid": i["id"], "nombre": i["name"]} for i in resp.json().get("items", [])]


def _requisitos(d: dict, tipo: str) -> str:
    req = d.get("pc_requirements")
    if not isinstance(req, dict):
        return ""
    crudo = re.sub(r"<li>", "\n", req.get(tipo) or "")
    lineas = [limpiar_texto(x, 200) for x in crudo.split("\n")]
    lineas = [x for x in lineas if x and not re.fullmatch(r"(?:M[ií]nimo|Recomendado|Minimum|Recommended)\s*:?", x, re.IGNORECASE)]
    return "\n".join("• " + x for x in lineas[:10])


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
        "imagen": d.get("header_image"),
        "url": f"https://store.steampowered.com/app/{appid}/",
        "requisitos_minimos": _requisitos(d, "minimum"),
        "requisitos_recomendados": _requisitos(d, "recommended"),
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


def get_best_deals(max_results: int = 10, min_descuento: int = 0) -> list[dict]:
    """Mejores ofertas activas en Steam (precios en MXN). Con min_descuento solo las de ese % o más."""
    resp = requests.get(
        "https://store.steampowered.com/api/featuredcategories",
        params={"cc": "mx", "l": "spanish"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    specials = resp.json().get("specials", {}).get("items", [])

    ofertas, vistos = [], set()
    for item in specials:
        descuento = item.get("discount_percent", 0)
        if len(ofertas) >= max_results:
            break
        if descuento > 0 and descuento >= min_descuento and item.get("id") not in vistos:
            vistos.add(item.get("id"))
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
                "minimo": min_descuento,
            })

    # Si piden un descuento mínimo y las destacadas no alcanzan, se buscan más entre las ofertas más vendidas
    if min_descuento and len(ofertas) < 5:
        extra = buscar_juegos(solo_ofertas=True, cantidad=max_results - len(ofertas), min_descuento=min_descuento,
                              excluir_appids=vistos)
        for j in extra["juegos"]:
            ofertas.append({
                "appid": j.get("appid"),
                "nombre": j["nombre"],
                "precio_original": j["precio_original"],
                "precio_final": j["precio"],
                "descuento": f"{j['descuento']}%",
                "termina": None,
                "minimo": min_descuento,
            })
    return ofertas


def compare_games(appid1: int, appid2: int) -> dict:
    """Compara dos juegos por sus características en la tienda."""
    juego1 = get_game_details(appid1)
    time.sleep(0.5)
    juego2 = get_game_details(appid2)
    return {"juego1": juego1, "juego2": juego2}


URL_IMAGENES_CLAN = "https://clan.akamai.steamstatic.com/images"


def primera_imagen(texto: str | None) -> str | None:
    """Saca la primera imagen de un anuncio: {STEAM_CLAN_IMAGE}/…png, <img src=…> o un link directo a imagen."""
    if not texto:
        return None
    m = re.search(r"\{STEAM_CLAN_IMAGE\}(/\S+?\.(?:png|jpe?g|gif|webp))", texto, re.IGNORECASE)
    if m:
        return URL_IMAGENES_CLAN + m.group(1)
    m = re.search(r"<img[^>]+src=[\"']([^\"']+)", texto, re.IGNORECASE)
    if m and m.group(1).startswith("http"):
        return m.group(1)
    m = re.search(r"https?://\S+?\.(?:png|jpe?g|gif|webp)", texto, re.IGNORECASE)
    return m.group(0) if m else None


# ---------------- Traducción (sin gastar el LLM) ----------------
# MyMemory: API gratuita de traducción. Sin correo da ~5,000 caracteres al día;
# con MYMEMORY_EMAIL en las variables de Render sube a ~50,000.
_cache_traducciones: dict[str, str] = {}
_INGLES = {"the", "and", "you", "of", "to", "is", "are", "for", "with", "this", "our", "we", "new", "now",
           "out", "update", "your", "will", "has", "have", "on", "in", "from", "be", "it", "at", "all"}
_ESPANOL = {"el", "la", "los", "las", "de", "del", "que", "y", "en", "para", "con", "una", "por", "ya", "nuevo", "es"}


def parece_ingles(texto: str) -> bool:
    palabras = re.findall(r"[a-záéíóúñ]+", texto.lower())
    ingles = sum(w in _INGLES for w in palabras)
    espanol = sum(w in _ESPANOL for w in palabras)
    return ingles >= 2 and ingles > espanol


ERRORES_TRADUCCION: list[str] = []   # motivo del último fallo, para diagnosticar


def _mymemory(texto: str) -> str | None:
    params = {"q": texto[:480], "langpair": "en|es"}
    if os.getenv("MYMEMORY_EMAIL"):
        params["de"] = os.getenv("MYMEMORY_EMAIL")
    resp = requests.get("https://api.mymemory.translated.net/get", params=params, timeout=6)
    resp.raise_for_status()
    datos = resp.json()
    traducido = html.unescape((datos.get("responseData") or {}).get("translatedText") or "")
    if not traducido or "MYMEMORY WARNING" in traducido.upper() or str(datos.get("responseStatus")) not in ("200", ""):
        raise RuntimeError(f"MyMemory: {traducido[:80] or datos.get('responseDetails')}")
    return traducido


def _google(texto: str) -> str | None:
    """Endpoint público de Google Translate (el mismo que usa la extensión del navegador)."""
    resp = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "en", "tl": "es", "dt": "t", "q": texto[:1500]},
        timeout=6,
    )
    resp.raise_for_status()
    partes = resp.json()[0] or []
    traducido = "".join(p[0] for p in partes if p and p[0])
    if not traducido:
        raise RuntimeError("Google: respuesta vacía")
    return traducido


def _traducir_uno(texto: str) -> str:
    if not texto or not parece_ingles(texto):
        return texto
    if texto in _cache_traducciones:
        return _cache_traducciones[texto]
    for traductor in (_google, _mymemory):
        try:
            traducido = traductor(texto)
            _cache_traducciones[texto] = traducido
            return traducido
        except Exception as exc:
            ERRORES_TRADUCCION.append(f"{type(exc).__name__}: {str(exc)[:120]}")
            del ERRORES_TRADUCCION[:-5]
    return texto   # si ninguno funciona, se deja en inglés


def traducir(textos: list[str]) -> list[str]:
    """Traduce al español los textos que estén en inglés (en paralelo). Si falla, regresa el original."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(_traducir_uno, textos))


def _traducir_noticias(noticias: list[dict]) -> list[dict]:
    textos = [n["titulo"] or "" for n in noticias] + [n["contenido"] or "" for n in noticias]
    traducidos = traducir(textos)
    k = len(noticias)
    for i, n in enumerate(noticias):
        n["traducido"] = traducidos[i] != (n["titulo"] or "") or traducidos[k + i] != (n["contenido"] or "")
        n["titulo"], n["contenido"] = traducidos[i], traducidos[k + i]
    return noticias


def error_traduccion() -> str | None:
    return ERRORES_TRADUCCION[-1] if ERRORES_TRADUCCION else None


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
                "contenido": limpiar_texto(item.get("contents"), 250),
                "imagen": primera_imagen(item.get("contents")),
                "fecha": (
                    datetime.fromtimestamp(fecha, tz=timezone.utc).strftime("%Y-%m-%d")
                    if fecha else None
                ),
                "url": item.get("url"),
            })
        return _traducir_noticias(noticias)

    resp = requests.get(
        "https://store.steampowered.com/feeds/news/",
        params={"l": "spanish"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    noticias = []
    for item in root.findall(".//item")[:max_results]:
        try:
            fecha = parsedate_to_datetime(item.findtext("pubDate")).strftime("%Y-%m-%d")
        except Exception:
            fecha = item.findtext("pubDate")
        noticias.append({
            "titulo": item.findtext("title"),
            "contenido": limpiar_texto(item.findtext("description"), 250),
            "imagen": primera_imagen(item.findtext("description")),
            "fecha": fecha,
            "url": item.findtext("link"),
        })
    return _traducir_noticias(noticias)


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
    tag_ids: list[int] | None = None,
    excluir_appid: int | None = None,
    min_descuento: int = 0,
    excluir_appids: set | None = None,
) -> dict:
    """Juegos más vendidos ahora (opcional: de un género, solo en oferta, o ordenados por reseñas)."""
    params = {"json": 1, "filter": "topsellers", "cc": "mx", "l": "spanish"}
    nombre_genero = None
    if genero:
        nombre_genero, tag_id = GENEROS[genero.lower()]
        params["tags"] = tag_id
    if tag_ids:
        params["tags"] = ",".join(str(t) for t in tag_ids)
    if solo_ofertas:
        params["specials"] = 1
    resp = requests.get("https://store.steampowered.com/search/results/", params=params, timeout=TIMEOUT)
    resp.raise_for_status()

    # Para ordenar por reseñas revisamos más candidatos y nos quedamos con los mejores
    candidatos = 12 if por_resenas else cantidad
    revisados = 0
    juegos = []
    for item in resp.json().get("items", []):
        if min_descuento and revisados >= 25:   # tope para no tardar demasiado
            break
        m = re.search(r"/apps/(\d+)/", item.get("logo", ""))
        if not m:
            continue
        appid = int(m.group(1))
        if appid == excluir_appid or (excluir_appids and appid in excluir_appids):
            continue
        detalle = get_game_details(appid)
        revisados += 1
        if detalle and min_descuento and (detalle.get("descuento") or 0) < min_descuento:
            continue
        if detalle and detalle.get("tipo") in (None, "game"):  # fuera hardware, DLC y soundtracks
            juegos.append({
                "appid": appid,
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
        "min_descuento": min_descuento,
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


# Etiquetas demasiado generales para buscar juegos parecidos
TAGS_GENERICOS = {
    "Singleplayer", "Multiplayer", "Indie", "Casual", "Early Access", "Free to Play", "3D", "2D",
    "First-Person", "Third Person", "Funny", "Great Soundtrack", "Atmospheric", "Colorful", "Cute",
    "Realistic", "Family Friendly", "Stylized", "Comedy", "Physics", "Online Co-Op", "PvP", "PvE",
    "Controller", "Steam Achievements", "Steam Trading Cards", "Full controller support", "Beautiful",
    "Relaxing", "Fantasy", "Difficult", "Story Rich", "Adventure", "Action", "Simulation",
}

# Géneros de la tienda (en español) → etiqueta de Steam, por si no se pueden leer las etiquetas
GENERO_A_TAG = {
    "Simuladores": 599, "Acción": 19, "Aventura": 21, "Rol": 122, "Estrategia": 9,
    "Carreras": 699, "Deportes": 701, "Casual": 597, "Indie": 492,
}


TAGS_ES = {
    "Management": "Gestión", "Building": "Construcción", "Simulation": "Simulación", "Strategy": "Estrategia",
    "Open World": "Mundo abierto", "Survival": "Supervivencia", "Horror": "Terror", "Survival Horror": "Survival horror",
    "Farming Sim": "Granja", "Platformer": "Plataformas", "Racing": "Carreras", "Sports": "Deportes",
    "Co-op": "Cooperativo", "Base Building": "Construcción de bases", "Crafting": "Fabricación",
    "City Builder": "Construcción de ciudades", "Life Sim": "Simulación de vida", "Exploration": "Exploración",
    "Pixel Graphics": "Pixel art", "Turn-Based": "Por turnos", "Tactical": "Táctico", "Card Game": "Cartas",
    "Sci-fi": "Ciencia ficción", "Space": "Espacio", "Zombies": "Zombis", "Psychological Horror": "Terror psicológico",
    "Economy": "Economía", "Automation": "Automatización", "Colony Sim": "Colonias", "Hack and Slash": "Hack and slash",
    "Fighting": "Pelea", "Mystery": "Misterio", "Detective": "Detective", "Medieval": "Medieval",
}


def get_store_tags(appid: int, maximo: int = 3) -> list[dict]:
    """Etiquetas que la comunidad le puso al juego (ej. Management, Building, Sandbox)."""
    resp = requests.get(
        f"https://store.steampowered.com/app/{appid}/",
        params={"l": "english"},
        cookies={"birthtime": "0", "lastagecheckage": "1-0-1990", "wants_mature_content": "1"},
        timeout=TIMEOUT,
    )
    if not resp.ok:
        return []
    m = re.search(r"InitAppTagModal\(\s*\d+,\s*(\[.*?\])\s*,", resp.text, re.DOTALL)
    if not m:
        return []
    try:
        tags = json.loads(m.group(1))
    except ValueError:
        return []
    utiles = [t for t in tags if t.get("name") not in TAGS_GENERICOS and t.get("tagid")]
    return [{"id": int(t["tagid"]), "nombre": TAGS_ES.get(t["name"], t["name"])} for t in utiles[:maximo]]


def juegos_similares(appid: int, cantidad: int = 6) -> dict:
    """Juegos parecidos: mismas etiquetas principales, ordenados por reseñas."""
    base = get_game_details(appid)
    tags = get_store_tags(appid)
    if not tags and base:
        especificos = [g for g in base["generos"] if g in GENERO_A_TAG and g not in ("Indie", "Casual")]
        tags = [{"id": GENERO_A_TAG[g], "nombre": g} for g in (especificos or base["generos"]) if g in GENERO_A_TAG][:2]

    resultado = {"juegos": []}
    # Si con 3 etiquetas hay pocos resultados, se relaja a 2 y luego a 1
    for n in range(len(tags), 0, -1):
        resultado = buscar_juegos(
            cantidad=cantidad,
            por_resenas=True,
            tag_ids=[t["id"] for t in tags[:n]],
            excluir_appid=appid,
        )
        if len(resultado["juegos"]) >= 3:
            break
    resultado["parecidos_a"] = base["nombre"] if base else None
    resultado["etiquetas"] = [t["nombre"] for t in tags]
    return resultado
