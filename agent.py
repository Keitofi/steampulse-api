"""Agente SteamPulse: decide la herramienta, ejecuta y sintetiza con el LLM del curso."""

import difflib
import json
import os
import re
import threading
import time
import uuid

import requests

from observability import LLAMADAS_DETALLE, estimar_tokens, registrar_llamada_llm, registrar_pregunta
from limits import LLAMADAS_LLM, TIEMPO_MAX_PARA_LLM, PresupuestoAgotado, consumir_llamada_llm, estado_llm
from steam_tools import (
    GENEROS,
    buscar_appid,
    buscar_candidatos,
    buscar_juegos,
    buscar_precios,
    compare_games,
    get_best_deals,
    get_game_info,
    get_resenas_destacadas,
    juegos_similares,
    get_steam_news,
    get_trending_games,
    error_traduccion,
    parece_ingles,
)

TOOLS_PERMITIDAS = {
    "get_trending_games",
    "get_best_deals",
    "compare_games",
    "get_steam_news",
    "buscar_por_genero",
    "top_juegos",
    "info_juego",
    "precios",
    "similares",
    "none",
}

TOOLS_DESCRIPCION = """
1. get_trending_games — Juegos más jugados en Steam ahora mismo. params: {}
2. get_best_deals — Mejores ofertas y descuentos activos. params: {"min_descuento": 0} (ej. 50 si piden "más del 50%")
3. compare_games — Compara dos juegos. params: {"juego1": "nombre", "juego2": "nombre"}
4. get_steam_news — Noticias o actualizaciones de un juego. params: {"juego": "nombre"} (vacío = noticias generales de Steam)
5. buscar_por_genero — Juegos populares de un género (terror, rpg, carreras, estrategia...). params: {"genero": "terror", "solo_ofertas": false, "cantidad": 6, "por_resenas": false, "min_descuento": 0}
5b. top_juegos — Los mejores / más vendidos / mejor calificados del momento. params: {"cantidad": 5, "por_resenas": false}
6. info_juego — Ficha de un juego: precio, géneros, descripción y jugadores conectados ahora. params: {"juego": "nombre"}
7. precios — Precios de un juego o de toda una saga (ej. Five Nights at Freddy's). params: {"juego": "nombre"}
8. similares — Juegos parecidos a uno que le gustó al usuario. params: {"juego": "nombre"}
9. none — La pregunta no necesita datos de Steam. params: {}
"""

PALABRAS_BLOQUEADAS = ["hack", "crack", "pirate", "pirata", "cheat"]


# ---------------- LLM del curso ----------------

def _extract_text(data) -> str | None:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = first.get("message")
        if isinstance(message, dict) and message.get("content"):
            return message["content"]
        if first.get("text"):
            return first["text"]
    for key in ("text", "content", "response", "answer", "output", "completion"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("data", "result", "generation"):
        if key in data:
            text = _extract_text(data[key])
            if text:
                return text
    return None


# ---------------- Optimización de costo ----------------
# 1) Caché: la misma consulta a Steam y su respuesta del LLM se reutilizan por un rato.
CACHE_TTL_S = {
    "get_best_deals": 600, "get_trending_games": 600, "top_juegos": 900, "buscar_por_genero": 900,
    "precios": 900, "get_steam_news": 1800, "info_juego": 1800, "compare_games": 1800, "similares": 3600,
}
_cache: dict[str, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def _clave_cache(tipo: str, decision: dict) -> str:
    return f"{tipo}:{decision.get('tool')}:{json.dumps(decision.get('params') or {}, sort_keys=True, ensure_ascii=False)}"


def cache_get(tipo: str, decision: dict):
    ttl = CACHE_TTL_S.get(decision.get("tool"))
    if not ttl:
        return None
    with _cache_lock:
        item = _cache.get(_clave_cache(tipo, decision))
    if item and time.time() - item[0] < ttl:
        return item[1]
    return None


def cache_set(tipo: str, decision: dict, valor) -> None:
    if decision.get("tool") not in CACHE_TTL_S or valor is None:
        return
    if isinstance(valor, dict) and valor.get("error"):
        return   # los errores no se guardan
    with _cache_lock:
        if len(_cache) > 500:   # tope de memoria
            _cache.clear()
        _cache[_clave_cache(tipo, decision)] = (time.time(), valor)


# 2) El LLM solo redacta donde aporta (resumir noticias, comparar, recomendar).
#    Ofertas, precios, listas y fichas salen bien con el formato fijo y no gastan créditos.
LLM_PARA = {t.strip() for t in os.getenv("LLM_PARA", "get_steam_news,compare_games,similares").split(",") if t.strip()}

# 3) Pedirle a GPT-5 que razone poco (cada respuesta tardaba ~17 s por el razonamiento oculto)
LLM_REASONING = os.getenv("LLM_REASONING", "low")
_acepta_reasoning = [True]

# 4) Cortacircuitos del LLM: sin créditos (403) → pausa 30 min; servidor caído (5xx) → pausa 3 min
_llm_pausa = {"hasta": 0.0, "motivo": ""}


def _pausar_llm(segundos: int, motivo: str) -> None:
    _llm_pausa["hasta"], _llm_pausa["motivo"] = time.time() + segundos, motivo


def llm_en_pausa() -> str | None:
    if time.time() < _llm_pausa["hasta"]:
        minutos = max(1, round((_llm_pausa["hasta"] - time.time()) / 60))
        return f"LLM en pausa ~{minutos} min ({_llm_pausa['motivo']})"
    return None


def call_llm(system_prompt: str, prompt: str, max_tokens: int = 600, proposito: str = "respuesta") -> str:
    base_url = os.getenv("BASE_URL", "https://api.artesaniadigital.org").rstrip("/")
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Falta la variable API_KEY")
    pausa = llm_en_pausa()
    if pausa:
        raise RuntimeError(pausa)   # no se gasta tiempo ni créditos en un LLM que sabemos que va a fallar

    payload = {
        "model": "gpt-5",
        "prompt": prompt,
        "system_prompt": system_prompt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1,
        "max_tokens": max_tokens,
    }
    if LLM_REASONING and _acepta_reasoning[0]:
        payload["reasoning_effort"] = LLM_REASONING
    headers = {"X-Student-Key": api_key, "Content-Type": "application/json"}

    tokens_prompt = estimar_tokens(system_prompt) + estimar_tokens(prompt)

    # Un reintento si el backend falla con error 5xx (suele ser transitorio)
    for intento in range(2):
        consumir_llamada_llm()  # cada intento cuenta contra el presupuesto diario
        inicio = time.perf_counter()
        try:
            resp = requests.post(f"{base_url}/v1/completions", headers=headers, json=payload, timeout=90)
        except requests.RequestException as exc:
            registrar_llamada_llm(proposito, round((time.perf_counter() - inicio) * 1000), None,
                                  tokens_prompt, 0, True, type(exc).__name__)
            if intento == 1:
                raise
            continue
        ms = round((time.perf_counter() - inicio) * 1000)
        if not resp.ok:
            registrar_llamada_llm(proposito, ms, resp.status_code, tokens_prompt, 0, True, resp.text[:150])
        # Si el backend no acepta reasoning_effort, se quita y se repite una vez (sin contar como reintento)
        if resp.status_code in (400, 422) and "reasoning_effort" in payload:
            _acepta_reasoning[0] = False
            payload.pop("reasoning_effort")
            consumir_llamada_llm()
            resp = requests.post(f"{base_url}/v1/completions", headers=headers, json=payload, timeout=90)
            ms = round((time.perf_counter() - inicio) * 1000)
        if resp.status_code < 500 or intento == 1:
            break
        time.sleep(2)

    if resp.status_code == 403 and "credit" in resp.text.lower():
        _pausar_llm(30 * 60, "sin créditos")
    elif resp.status_code >= 500:
        _pausar_llm(3 * 60, f"servidor del LLM con error {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    text = _extract_text(data)
    # Tokens reales si el backend los reporta; si no, estimados (~4 caracteres por token)
    uso = data.get("usage") if isinstance(data, dict) else None
    uso = uso if isinstance(uso, dict) else {}
    entrada = uso.get("prompt_tokens") or uso.get("input_tokens")
    salida = uso.get("completion_tokens") or uso.get("output_tokens")
    registrar_llamada_llm(proposito, ms, resp.status_code,
                          entrada or tokens_prompt, salida or estimar_tokens(text or ""),
                          estimado=not (entrada and salida))
    if not text:
        raise RuntimeError("El backend no devolvió texto")
    return text


# ---------------- Decidir herramienta ----------------

def _limpiar_nombre(nombre: str) -> str:
    return nombre.strip(" ¿?¡!.,;:\"'")


# Abreviaturas que usa la gente → nombre como aparece en Steam
ALIAS = {
    "rdr2": "Red Dead Redemption 2", "rdr 2": "Red Dead Redemption 2", "rdr": "Red Dead Redemption",
    "gta5": "Grand Theft Auto V", "gta 5": "Grand Theft Auto V", "gtav": "Grand Theft Auto V", "gta v": "Grand Theft Auto V",
    "gta4": "Grand Theft Auto IV", "gta 4": "Grand Theft Auto IV", "gta iv": "Grand Theft Auto IV",
    "gta": "Grand Theft Auto V", "gta online": "Grand Theft Auto V", "gta 6": "Grand Theft Auto VI", "gta vi": "Grand Theft Auto VI",
    "kcd 2": "Kingdom Come: Deliverance II", "gta 3": "Grand Theft Auto III", "gta iii": "Grand Theft Auto III",
    "gta vice city": "Grand Theft Auto: Vice City", "gta vc": "Grand Theft Auto: Vice City", "gta san andreas": "Grand Theft Auto: San Andreas", "gta sa": "Grand Theft Auto: San Andreas",
    "cs2": "Counter-Strike 2", "cs 2": "Counter-Strike 2", "csgo": "Counter-Strike 2", "cs go": "Counter-Strike 2",
    "csgo2": "Counter-Strike 2", "csgo 2": "Counter-Strike 2", "cs go 2": "Counter-Strike 2", "cs:go": "Counter-Strike 2",
    "cs:go 2": "Counter-Strike 2", "csg2": "Counter-Strike 2", "csg 2": "Counter-Strike 2", "csgo": "Counter-Strike 2",
    "counter strike 2": "Counter-Strike 2", "counterstrike 2": "Counter-Strike 2", "counter strike": "Counter-Strike 2",
    "tf2": "Team Fortress 2", "l4d2": "Left 4 Dead 2", "l4d": "Left 4 Dead",
    "pubg": "PUBG: BATTLEGROUNDS", "fnaf": "Five Nights at Freddy's",
    "re4": "Resident Evil 4", "re2": "Resident Evil 2", "re3": "Resident Evil 3", "re8": "Resident Evil Village",
    "re village": "Resident Evil Village", "bg3": "Baldur's Gate 3", "baldurs gate 3": "Baldur's Gate 3",
    "ds3": "DARK SOULS III", "dark souls 3": "DARK SOULS III", "ds2": "DARK SOULS II",
    "sdv": "Stardew Valley", "tw3": "The Witcher 3: Wild Hunt", "witcher 3": "The Witcher 3: Wild Hunt",
    "cp2077": "Cyberpunk 2077", "mhw": "Monster Hunter: World",
    "mh wilds": "Monster Hunter Wilds", "mhwilds": "Monster Hunter Wilds", "ror2": "Risk of Rain 2",
    "dbd": "Dead by Daylight", "r6": "Tom Clancy's Rainbow Six Siege", "r6s": "Tom Clancy's Rainbow Six Siege", "apex": "Apex Legends", "poe": "Path of Exile",
    "poe2": "Path of Exile 2", "poe 2": "Path of Exile 2", "ets2": "Euro Truck Simulator 2",
    "ats": "American Truck Simulator", "hk": "Hollow Knight", "hl2": "Half-Life 2",
    "kcd2": "Kingdom Come: Deliverance II", "kcd": "Kingdom Come: Deliverance", "ac valhalla": "Assassin's Creed Valhalla",
    "gow": "God of War", "rl": "Rocket League", "cod": "Call of Duty",
    "fh5": "Forza Horizon 5", "forza 5": "Forza Horizon 5", "fh4": "Forza Horizon 4", "nms": "No Man's Sky",
    "dst": "Don't Starve Together", "aoe2": "Age of Empires II: Definitive Edition", "aoe4": "Age of Empires IV",
    "sekiro": "Sekiro: Shadows Die Twice", "p5r": "Persona 5 Royal", "p3r": "Persona 3 Reload",
}
# Ordenados de más largo a más corto para que "gta san andreas" gane sobre "gta"
_PATRON_ALIAS = re.compile(
    r"(?<![\w'])(" + "|".join(re.escape(a) for a in sorted(ALIAS, key=len, reverse=True)) + r")(?![\w'])(?!\s*(?:\d|[ivx]+\b))",
    re.IGNORECASE,
)


def expandir_alias(texto: str) -> str:
    """'¿rdr2 está en oferta?' → '¿Red Dead Redemption 2 está en oferta?'"""
    return _PATRON_ALIAS.sub(lambda m: ALIAS[m.group(1).lower()], texto)


def _palabras(texto: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", texto.lower().replace("'", ""))


def _siglas(nombre: str) -> str:
    """'Red Dead Redemption 2' → 'rdr2' · 'Counter-Strike 2' → 'cs2'."""
    return "".join(w if w.isdigit() else w[0] for w in _palabras(nombre))


def _parece_el_juego(texto: str, nombre: str) -> bool:
    """¿Lo que escribió el usuario corresponde al juego que encontró Steam?"""
    t, n = _palabras(texto), _palabras(nombre)
    if not t or not n:
        return False
    compacto = "".join(t)
    if compacto in (_siglas(nombre), "".join(n)):   # "rdr2", "hollowknight"
        return True

    def coincide(w: str) -> bool:
        # exacta, o con un error de dedo ("mith" ≈ "myth") en palabras de 4+ letras
        return w in n or (len(w) >= 4 and bool(difflib.get_close_matches(w, n, n=1, cutoff=0.75)))

    aciertos = sum(coincide(w) for w in t)
    if len(compacto) >= 4 and aciertos == len(t):   # "stardew" en "Stardew Valley"
        return True
    return len(t) >= 2 and aciertos / len(t) >= 0.6


def _juegos_a_comparar(p: str) -> tuple[str, str] | None:
    m = re.search(r"compar\w*\s+(?:a\s+|el\s+|entre\s+)?(.+?)\s+(?:contra|vs\.?|versus|con|y)\s+(.+)", p, re.IGNORECASE)
    if m:
        j1, j2 = _limpiar_nombre(m.group(1)), _limpiar_nombre(m.group(2))
        if j1 and j2:
            return j1, j2
    return None


RELLENO_NOTICIAS = {
    "qué", "que", "cuáles", "cuales", "hay", "ay", "noticias", "noticia", "nuevas", "nuevo", "nuevos", "recientes",
    "últimas", "ultimas", "actualizaciones", "actualización", "actualizacion", "updates", "update", "novedades",
    "novedad", "parches", "parche", "de", "del", "e", "d", "sobre", "tiene", "tuvo", "el", "la", "los", "las", "en",
    "dame", "dime", "me", "algo", "alguna", "algunas", "hubo", "salieron", "salió", "salio", "para", "con", "y",
}
SIN_JUEGO = {"steam", "hoy", "la semana", "esta semana", "ahora", "semana", "esta", "este mes", "mes"}


RELLENO_FINAL = {"hay", "ay", "porfa", "porfavor", "plis", "pls", "please", "hoy", "ahora", "ahorita", "actualmente",
                 "recientes", "nuevas", "nuevos", "nuevo", "nueva", "últimas", "ultimas", "gracias", "tiene", "tuvo",
                 "salió", "salio", "salieron", "hubo", "también", "tambien", "pues", "xfa", "x", "fa"}


def _quitar_relleno_final(nombre: str) -> str:
    """'terraria hay' → 'terraria' · 'elden ring por favor' → 'elden ring'."""
    nombre = re.sub(r"\s+por\s+favor\s*$", "", _limpiar_nombre(nombre), flags=re.IGNORECASE)
    palabras = nombre.split()
    while len(palabras) > 1 and palabras[-1].lower().strip("¿?¡!.,") in RELLENO_FINAL:
        palabras.pop()
    return _limpiar_nombre(" ".join(palabras))


def _juego_de_noticias(p: str) -> str | None:
    m = re.search(r"(?:\btuvo\b|\btiene\b|\bsobre\b|\bdel\b|\bde\b|\bpara\b)\s+(?:el\s+|la\s+)?(.+)$", p, re.IGNORECASE)
    if m:
        # "qué hay de noticias de terraria" → quitar "noticias de" que quedó al inicio
        palabras = _limpiar_nombre(m.group(1)).split()
        while palabras and palabras[0].lower().strip("¿?¡!.,") in RELLENO_NOTICIAS - {"hay", "ay"}:
            palabras.pop(0)
        nombre = _quitar_relleno_final(" ".join(palabras))
        if nombre and nombre.lower() not in SIN_JUEGO:
            return nombre
    # Con errores de dedo ("qué noticias hay e minecraft"): lo que sobra al quitar las palabras de relleno
    palabras = [w for w in re.findall(r"[\w'’:\-]+", p) if w.lower() not in RELLENO_NOTICIAS]
    nombre = _limpiar_nombre(" ".join(palabras))
    if nombre and nombre.lower() not in SIN_JUEGO and len(palabras) <= 6:
        return nombre
    return None


NUMEROS = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
           "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}


def _descuento_minimo(p: str) -> int:
    """'más del 50%', '70% o más', 'arriba de 60', 'a mitad de precio' → % mínimo de descuento (0 si no pide)."""
    p = p.lower()
    if "mitad de precio" in p:
        return 50
    m = (re.search(r"(?:m[aá]s\s+del?|arriba\s+del?|mayor(?:es)?\s+(?:a|al|de|del)|m[ií]nimo(?:\s+de(?:l)?)?|al\s+menos|por\s+lo\s+menos|desde|sobre)\s*(?:un\s+|el\s+)?(\d{1,2})\s*%?", p)
         or re.search(r"(\d{1,2})\s*%\s*(?:o\s+m[aá]s|para\s+arriba|o\s+mayor|en\s+adelante)", p)
         or re.search(r"(\d{1,2})\s*%", p))
    return int(m.group(1)) if m else 0


def _cantidad_en(p: str, por_defecto: int) -> int:
    m = re.search(r"\b(\d{1,2})\b(?!\s*%)(?!\s+(?:de\s+)?(?:descuento|por\s*ciento))", p)
    if m:
        return max(1, min(int(m.group(1)), 10))
    for palabra, n in NUMEROS.items():
        if re.search(rf"\b{palabra}\b", p.lower()):
            return n
    return por_defecto


SOLO_SALUDO = re.compile(r"^\W*buen[oa]s?(?:\s+(?:tardes|noches|d[ií]as))?\W*$", re.IGNORECASE)


def _pide_buenos(p: str) -> bool:
    """'qué buenos juegos hay', 'juegos chidos', 'algo bueno para jugar' (pero no 'buenas tardes')."""
    if SOLO_SALUDO.match(p):
        return False
    return bool(re.search(r"\b(?:buen[oa]s?|chid[oa]s?|padres?|recomendable|vale[n]? la pena)\b|de bueno", p))


def _pide_resenas(p: str) -> bool:
    return _pide_buenos(p) or any(w in p for w in ["rankead", "calificad", "valorad", "reseña", "resena", "rating",
                                                    "puntuad", "review", "mejor evaluad", "recomi"])


def _pide_top(p: str) -> bool:
    return _pide_buenos(p) or any(w in p for w in [
        "mejor", "top", "rankead", "ranking", "más vendid", "mas vendid", "temporada", "recomi", "calificad",
        "valorad", "qué juego compro", "que juego compro", "qué puedo jugar", "que puedo jugar",
        "qué juego juego", "que juego juego", "algo para jugar", "qué jugar", "que jugar", "estoy aburrid"])


def _genero_en(p: str) -> str | None:
    p = p.lower()
    for clave in sorted(GENEROS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(clave)}(?:s|es)?(?!\w)", p):   # "cooperativos", "rpgs"
            return clave
    return None


def _juego_seguimiento(p: str) -> str | None:
    """Saca el juego de una pregunta de seguimiento como '¿y de Dota 2?'."""
    texto = _limpiar_nombre(p)
    texto = re.sub(r"^(?:y\s+)?(?:qué tal\s+|que tal\s+)?(?:de\s+|del\s+|sobre\s+|con\s+|el\s+|la\s+)*", "", texto, flags=re.IGNORECASE)
    texto = _limpiar_nombre(texto)
    return texto if 0 < len(texto) <= 40 else None


PATRON_INFO = re.compile(
    r"(?:sabes (?:algo )?(?:de|del|sobre)|conoces|qu[eé] (?:es|tal es)|"
    r"info(?:rmaci[oó]n)? (?:de|del|sobre)|h[aá]blame (?:de|del|sobre)|"
    r"cu[eé]ntame (?:de|del|sobre)|"
    r"cu[aá]ntos jugadores tiene)\s+(?:el\s+|la\s+)?(.+)",
    re.IGNORECASE,
)


PATRON_PRECIO = re.compile(
    r"(?:en cu[aá]nto (?:est[aá]n?|sale[n]?)|cu[aá]nto (?:cuesta[n]?|vale[n]?|sale[n]?)|precios?(?: (?:de|del))?)"
    r"\s+(?:los\s+|las\s+|el\s+|la\s+)?(?:juegos?\s+(?:de\s+(?:la\s+saga\s+(?:de\s+)?)?)?)?(.+)",
    re.IGNORECASE,
)


PATRON_SIMILARES = [
    re.compile(r"(?:parecid[oa]s?|similares?)\s+(?:a|al)\s+(?:el\s+|la\s+)?(.+?)(?=[,.?¿!]|$)", re.IGNORECASE),
    re.compile(r"juegos?\s+(?:como|tipo|estilo)\s+(?:el\s+|la\s+)?(.+?)(?=[,.?¿!]|$)", re.IGNORECASE),
    re.compile(r"me\s+(?:gust[óo]|encant[óo]|gusta|encanta)\s+(?:mucho\s+)?(?:el\s+|la\s+|jugar\s+)?(.+?)(?=[,.?¿!]|\s+(?:qu[eé]|y|pero|cu[aá]les?|algo|otros?)\b|$)", re.IGNORECASE),
]
PIDE_SIMILARES = ["parecid", "similar", "otros juegos", "qué más", "que mas", "qué otros", "que otros", "algo así", "algo asi", "me pueden gustar", "me puede gustar"]


def _juego_para_similares(pregunta: str) -> str | None:
    for patron in PATRON_SIMILARES:
        m = patron.search(pregunta)
        if m:
            nombre = _limpiar_nombre(m.group(1))
            if nombre and not _genero_en(nombre) and len(nombre.split()) <= 6:
                return nombre
    return None


PALABRAS_ATRIBUTO = ["multijugador", "multiplayer", "cooperativ", "co-op", "coop", "online", "en línea", "en linea",
                     "pvp", "español", "espanol", "idioma", "subtítulo", "subtitulo", "doblaje", "doblado",
                     "mando", "control", "gamepad", "mac", "linux", "steam deck", "gratis", "free to play",
                     "requisito", "requerimiento", "specs", "bueno", "buenas", "vale la pena", "jugadores", "un jugador",
                     "reseña", "resena", "opinion", "opinión", "critica", "crítica"]
NO_ES_JUEGO = re.compile(r"^(?:qu[eé]|cu[aá]l|cu[aá]les|hay|alg[uú]n|alguno|algo|juegos?|un juego|el juego|este|ese|esto|eso|y|me|mi|si|s[ií])\b", re.IGNORECASE)
PATRONES_ATRIBUTO = [
    # requisitos mínimos del sistema de Undertale / specs de Elden Ring / qué pc necesito para X
    re.compile(r"(?:requisitos|requerimientos|specs|especificaciones)(?:\s+(?:m[ií]nimos|recomendados|t[eé]cnicos))?"
               r"(?:\s+(?:del?\s+)?(?:sistema|pc|computadora))?\s+(?:de|del|para|pide|necesita|ocupa)\s+(?:jugar\s+)?(?:el\s+|la\s+)?(.+)", re.IGNORECASE),
    re.compile(r"(?:pc|computadora|compu)\s+(?:necesito|ocupo|se necesita|pide)\s+para\s+(?:jugar\s+|correr\s+)?(?:el\s+|la\s+)?(.+)", re.IGNORECASE),
    # qué dicen los usuarios de DbD / reseñas de Elden Ring / opiniones sobre Hades
    re.compile(r"(?:qu[eé]\s+(?:dicen|opinan|piensan|comentan|tal\s+(?:las\s+)?(?:rese[ñn]as|opiniones|cr[ií]ticas))"
               r"(?:\s+(?:los|las|la)\s+(?:usuarios|jugadores|gamers|personas|gente|rese[ñn]as|comunidad))?"
               r"|opiniones?|rese[ñn]as?|comentarios|cr[ií]ticas)\s+(?:de|del|sobre|acerca\s+de|en)\s+(?:el\s+|la\s+)?(.+)", re.IGNORECASE),
    # en qué idiomas está Danganronpa / idiomas de X
    re.compile(r"qu[eé]\s+idiomas?\s+(?:est[aá]|viene|tiene|trae|se puede jugar|hay)\s+(?:en\s+)?(?:el\s+|la\s+)?(.+)", re.IGNORECASE),
    re.compile(r"idiomas?\s+(?:de|del|tiene|trae)\s+(?:el\s+|la\s+)?(.+)", re.IGNORECASE),
]


def _juego_con_atributo(pregunta: str) -> str | None:
    """Saca el juego de preguntas como '¿Elden Ring es cooperativo?' o 'requisitos de Undertale'."""
    for patron in PATRONES_ATRIBUTO:
        m = patron.search(pregunta)
        if m:
            nombre = _limpiar_nombre(m.group(1))
            if nombre and len(nombre.split()) <= 6 and not NO_ES_JUEGO.match(nombre):
                return nombre
    m = re.match(r"^\W*(?:el\s+|la\s+)?(.+?)\s+(?:es|tiene|trae|viene|est[aá]|soporta|incluye|corre|sirve|se puede jugar)\b(.*)$",
                 pregunta, re.IGNORECASE)
    if m and any(w in m.group(2).lower() for w in PALABRAS_ATRIBUTO):
        nombre = _limpiar_nombre(m.group(1))
        if nombre and len(nombre.split()) <= 6 and not NO_ES_JUEGO.match(nombre) and not _genero_en(nombre):
            return nombre
    return None


def decidir_por_reglas(pregunta: str, contexto: dict | None = None) -> dict:
    """Router determinista: resuelve la mayoría de preguntas sin gastar tokens.

    `contexto` es la pregunta anterior ({pregunta, tool, params}) para seguimientos.
    """
    p = pregunta.lower()
    previo = (contexto or {}).get("tool")
    params_previos = (contexto or {}).get("params") or {}
    pide_ofertas = any(w in p for w in ["oferta", "ofrerta", "oferts", "descuento", "barato", "rebaja", "promo", "mitad de precio"])
    min_desc = _descuento_minimo(p) if pide_ofertas else 0

    juegos = _juegos_a_comparar(pregunta)
    if juegos:
        return {"tool": "compare_games", "params": {"juego1": juegos[0], "juego2": juegos[1]}}

    # "Me gustó X, ¿qué otros juegos me pueden gustar?" / "juegos parecidos a X"
    if (any(w in p for w in PIDE_SIMILARES) or "me gust" in p or "me encant" in p
            or re.search(r"juegos?\s+(?:como|tipo|estilo)\s", p)):
        juego = _juego_para_similares(pregunta)
        if not juego and previo == "info_juego" and any(w in p for w in PIDE_SIMILARES):
            juego = params_previos.get("juego")
        if juego:
            return {"tool": "similares", "params": {"juego": juego}}

    # Pregunta sobre un atributo de un juego nombrado: requisitos, idiomas, "¿Elden Ring es cooperativo?"
    juego = _juego_con_atributo(pregunta)
    if juego:
        return {"tool": "info_juego", "params": {"juego": juego, "atributo": pregunta}}

    # Pregunta sobre el juego anterior: "¿es multijugador?", "¿tiene español?", "¿está en oferta?"
    if previo == "info_juego" and params_previos.get("juego") and re.match(
        r"^\W*(?:y\s+)?(?:es|tiene|trae|viene|se puede|puedo|est[aá]|hay|cu[aá]nt[oa]s?|sirve|corre|incluye|soporta|acepta|funciona)\b",
        pregunta, re.IGNORECASE,
    ):
        return {"tool": "info_juego", "params": {"juego": params_previos["juego"], "atributo": pregunta}}

    genero = _genero_en(pregunta)
    if genero:
        solo_ofertas = pide_ofertas or previo == "get_best_deals"
        recomienda = any(w in p for w in ["recomi", "me gusta", "qué juego", "que juego", "qué juegos", "que juegos"])
        params_genero = {"min_descuento": min_desc} if min_desc else {}
        return {"tool": "buscar_por_genero", "params": {
            **params_genero,
            "genero": genero, "solo_ofertas": solo_ofertas or bool(min_desc),
            "cantidad": _cantidad_en(pregunta, 6), "por_resenas": _pide_resenas(p) or recomienda}}

    m = PATRON_PRECIO.search(pregunta)
    if m and _limpiar_nombre(m.group(1)):
        return {"tool": "precios", "params": {"juego": _limpiar_nombre(m.group(1))}}

    if pide_ofertas:
        # "¿Red Dead Redemption 2 está en oferta?" → revisar ese juego, no todas las ofertas
        m = re.match(r"^\W*(?:el\s+|la\s+)?(.+?)\s+(?:est[aá]n?|tiene)\s+(?:en\s+)?(?:oferta|descuento|rebaja)", pregunta, re.IGNORECASE)
        if m and _limpiar_nombre(m.group(1)) and _limpiar_nombre(m.group(1)).lower() not in {"que", "qué", "cuál", "cual", "algo", "qué juego", "que juego"}:
            return {"tool": "info_juego", "params": {"juego": _limpiar_nombre(m.group(1)), "atributo": pregunta}}
        # "¿y en oferta?" después de buscar un género → ofertas de ese género
        if previo == "buscar_por_genero" and params_previos.get("genero"):
            return {"tool": "buscar_por_genero", "params": {"genero": params_previos["genero"], "solo_ofertas": True,
                                                            **({"min_descuento": min_desc} if min_desc else {})}}
        return {"tool": "get_best_deals", "params": {"min_descuento": min_desc} if min_desc else {}}

    if any(w in p for w in ["noticia", "actualizaci", "update", "parche", "novedad"]):
        juego = _juego_de_noticias(pregunta)
        return {"tool": "get_steam_news", "params": {"juego": juego} if juego else {}}

    if _pide_top(p) and not any(w in p for w in ["jugado", "jugadores", "moda", "tendencia"]):
        return {"tool": "top_juegos", "params": {"cantidad": _cantidad_en(pregunta, 5), "por_resenas": _pide_resenas(p)}}


    m = PATRON_INFO.search(pregunta)
    if m and _limpiar_nombre(m.group(1)):
        return {"tool": "info_juego", "params": {"juego": _limpiar_nombre(m.group(1))}}

    if any(w in p for w in ["tendencia", "moda", "más jugado", "mas jugado", "popular", "jugadores", "jugado"]):
        return {"tool": "get_trending_games", "params": {"cantidad": _cantidad_en(pregunta, 10)}}

    # Seguimientos usando la pregunta anterior
    if previo == "info_juego" and pregunta.strip(" ¿?").lower().startswith("y "):
        juego = _juego_seguimiento(pregunta)
        if juego:
            return {"tool": "info_juego", "params": {"juego": juego}}
    if previo == "get_steam_news":
        juego = _juego_seguimiento(pregunta)
        if juego:
            return {"tool": "get_steam_news", "params": {"juego": juego}}
    if previo == "compare_games" and params_previos.get("juego1"):
        m = (re.search(r"(?:contra|vs\.?|versus)\s+(.+)", pregunta, re.IGNORECASE)
             or re.search(r"^\W*y\s+(?:con\s+)?(.+)", pregunta, re.IGNORECASE))
        if m and _limpiar_nombre(m.group(1)):
            return {"tool": "compare_games", "params": {"juego1": params_previos["juego1"], "juego2": _limpiar_nombre(m.group(1))}}

    return {"tool": "none", "params": {}}


def decidir_tool(pregunta: str, contexto: dict | None = None) -> tuple[dict, str]:
    """Devuelve (decisión, origen). El origen es 'reglas' o 'llm'.

    Primero reglas deterministas (gratis e instantáneas). Solo se usa el LLM
    cuando la pregunta necesita extraer nombres de juegos o no es clara.
    """
    por_reglas = decidir_por_reglas(pregunta, contexto)
    if por_reglas["tool"] != "none":
        return por_reglas, "reglas"

    # Antes de gastar el LLM: ¿escribió solo el nombre de un juego? ("rdr2", "hollow knight")
    como_juego = _ultimo_recurso(pregunta, por_reglas)
    if como_juego["tool"] != "none":
        return como_juego, "reglas"

    system_prompt = f"""Eres el router de SteamPulse, un agente de inteligencia gaming.
Herramientas disponibles:
{TOOLS_DESCRIPCION}
Responde ÚNICAMENTE con un JSON: {{"tool": "...", "params": {{...}}}}
Usa nombres de juegos tal como los escribió el usuario.
Si la pregunta es un seguimiento, usa la pregunta anterior para entenderla."""
    entrada = pregunta
    if contexto:
        entrada = (
            f"Pregunta anterior: {contexto.get('pregunta')} "
            f"(herramienta: {contexto.get('tool')}, params: {contexto.get('params')})\n"
            f"Pregunta actual: {pregunta}"
        )
    try:
        texto = call_llm(system_prompt, entrada, max_tokens=600, proposito="router")
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        decision = json.loads(match.group(0)) if match else {}
        if decision.get("tool") in TOOLS_PERMITIDAS and decision["tool"] != "none":
            decision.setdefault("params", {})
            return decision, "llm"
    except Exception:
        pass
    return por_reglas, "reglas"


SALUDOS = {"hola", "gracias", "buenas", "buenos dias", "buenos días", "buenas tardes", "buenas noches",
           "ok", "va", "adios", "adiós", "que onda", "qué onda", "ayuda", "help", "si", "sí", "no"}


def _ultimo_recurso(pregunta: str, decision: dict) -> dict:
    """Si nada coincidió, prueba si el texto es el nombre de un juego y muestra su ficha."""
    if decision["tool"] != "none":
        return decision
    texto = RELLENO.sub("", _limpiar_nombre(pregunta)).strip()
    if texto.lower() in NO_EN_STEAM:   # "minecraft" → avisar, no mostrar Minecraft Dungeons
        return {"tool": "info_juego", "params": {"juego": texto}}
    if len(texto) < 2 or texto.lower() in SALUDOS or len(texto.split()) > 6:
        return decision
    try:
        encontrado = buscar_tolerante(texto)
    except Exception:
        return decision
    if encontrado and _parece_el_juego(texto, encontrado["nombre"] or ""):
        return {"tool": "info_juego", "params": {"juego": encontrado["nombre"]}}
    return decision


# ---------------- Ejecutar herramienta ----------------

_NUMERO = re.compile(r"^(?:\d+|[ivx]{1,5})$")   # 2, 3, 2077, II, IV, XIV…


def _sobran_numeros(texto: str, nombre: str) -> bool:
    """¿El juego encontrado tiene un número que el usuario no escribió? ('Red Dead Redemption' → 'Red Dead Redemption 2')"""
    usuario = {w for w in _palabras(texto) if _NUMERO.match(w)}
    juego = {w for w in _palabras(nombre) if _NUMERO.match(w)}
    return bool(juego - usuario)


def _elegir(texto: str, candidatos: list[dict]) -> dict | None:
    """De los resultados de Steam, el que se parece a lo escrito, prefiriendo el que no agrega una secuela."""
    parecidos = [c for c in candidatos if _parece_el_juego(texto, c["nombre"] or "")]
    exactos = [c for c in parecidos if not _sobran_numeros(texto, c["nombre"] or "")]
    return (exactos or parecidos or [None])[0]


def _corregir(texto: str, nombre: str) -> str:
    """Corrige errores de dedo usando las palabras del juego encontrado: 'read dead' → 'red dead'."""
    palabras_juego = _palabras(nombre)
    corregidas = []
    for w in _palabras(texto):
        cercana = difflib.get_close_matches(w, palabras_juego, n=1, cutoff=0.75) if len(w) >= 3 else []
        corregidas.append(cercana[0] if cercana else w)
    return " ".join(corregidas)


def buscar_tolerante(nombre: str) -> dict | None:
    """Busca en Steam aguantando errores de dedo y sin brincar a la secuela.

    1. Busca normal y elige el resultado que más se parece (no siempre el primero).
    2. Si no hay, o el encontrado es una secuela que el usuario no pidió, reintenta:
       con las palabras corregidas ('read dead redemption' → 'red dead redemption')
       y quitando una palabra a la vez ('black mith wukong' → 'black wukong').
    """
    candidatos = buscar_candidatos(nombre)
    elegido = _elegir(nombre, candidatos)
    if elegido and not _sobran_numeros(nombre, elegido["nombre"] or ""):
        return elegido

    intentos = []
    base = elegido or (candidatos[0] if candidatos else None)
    if base:
        corregido = _corregir(nombre, base["nombre"] or "")
        if corregido != " ".join(_palabras(nombre)):
            intentos.append(corregido)
    palabras = [w for w in re.split(r"[\s:\-]+", nombre) if w]
    if len(palabras) >= 2:
        for i in sorted(range(len(palabras)), key=lambda k: len(palabras[k]))[:3]:   # primero quita las más cortas
            intentos.append(" ".join(palabras[:i] + palabras[i + 1:]))

    for intento in intentos[:4]:
        otro = _elegir(nombre, buscar_candidatos(intento))
        if otro and not _sobran_numeros(nombre, otro["nombre"] or ""):
            return otro
        elegido = elegido or otro
    return elegido or (candidatos[0] if candidatos else None)


RELLENO = re.compile(r"^(?:(?:el|la|los|las|un|una)\s+)?(?:video)?juegos?\s+(?:de\s+|llamado\s+|que se llama\s+)?", re.IGNORECASE)


def _resolver(nombre: str | None) -> dict | None:
    if not nombre:
        return None
    nombre = RELLENO.sub("", _limpiar_nombre(nombre)).strip() or nombre   # "el juego restory" → "restory"
    nombre = _quitar_relleno_final(nombre) or nombre                       # "terraria hay" → "terraria"
    return buscar_tolerante(nombre)


# Juegos famosos que NO están en Steam: mejor decirlo que mostrar otro juego parecido
NO_EN_STEAM = {
    "minecraft": "Minecraft no se vende en Steam (se compra en minecraft.net o en la Microsoft Store). En Steam solo están Minecraft Dungeons y Minecraft Legends.",
    "fortnite": "Fortnite no está en Steam; se descarga gratis desde la Epic Games Store.",
    "valorant": "Valorant no está en Steam; se descarga gratis desde el cliente de Riot Games.",
    "league of legends": "League of Legends no está en Steam; se descarga gratis desde el cliente de Riot Games.",
    "lol": "League of Legends no está en Steam; se descarga gratis desde el cliente de Riot Games.",
    "roblox": "Roblox no está en Steam; se descarga desde roblox.com o la Microsoft Store.",
    "genshin": "Genshin Impact no está en Steam; se descarga desde su página oficial o la Epic Games Store.",
    "genshin impact": "Genshin Impact no está en Steam; se descarga desde su página oficial o la Epic Games Store.",
    "free fire": "Free Fire es un juego de celular, no está en Steam.",
    "clash royale": "Clash Royale es un juego de celular, no está en Steam.",
    "grand theft auto vi": "GTA 6 todavía no está en Steam; Rockstar lo anunció primero para consolas.",
    "zelda": "Los juegos de Zelda son exclusivos de Nintendo, no están en Steam.",
    "mario": "Los juegos de Mario son exclusivos de Nintendo, no están en Steam.",
    "pokemon": "Los juegos de Pokémon son exclusivos de Nintendo, no están en Steam.",
    "pokémon": "Los juegos de Pokémon son exclusivos de Nintendo, no están en Steam.",
}


def _fuera_de_steam(params: dict) -> str | None:
    for clave in ("juego", "juego1", "juego2"):
        nombre = RELLENO.sub("", _limpiar_nombre(str(params.get(clave) or ""))).strip().lower()
        if nombre in NO_EN_STEAM:
            return NO_EN_STEAM[nombre]
    return None


def ejecutar_tool(decision: dict):
    tool = decision.get("tool")
    params = decision.get("params", {}) or {}

    aviso = _fuera_de_steam(params)
    if aviso:
        return {"error": aviso}

    if tool == "get_trending_games":
        return get_trending_games(limit=int(params.get("cantidad") or 10))
    if tool == "top_juegos":
        return buscar_juegos(cantidad=int(params.get("cantidad") or 5), por_resenas=bool(params.get("por_resenas")))
    if tool == "get_best_deals":
        minimo = int(params.get("min_descuento") or 0)
        ofertas = get_best_deals(min_descuento=minimo)
        if not ofertas:
            return {"error": f"Ahorita no encontré ofertas con {minimo}% o más de descuento. Prueba con un porcentaje menor."
                    if minimo else "No encontré ofertas activas en este momento."}
        return ofertas
    if tool == "compare_games":
        j1 = _resolver(params.get("juego1"))
        j2 = _resolver(params.get("juego2"))
        if not j1 or not j2:
            return {"error": "No encontré uno de los juegos a comparar"}
        return compare_games(j1["appid"], j2["appid"])
    if tool == "get_steam_news":
        juego = _resolver(params.get("juego"))
        if params.get("juego") and not juego:
            return {"error": f"No encontré '{params['juego']}' en Steam. Revisa el nombre o prueba con otro juego."}
        noticias = get_steam_news(appid=juego["appid"] if juego else None)
        resultado = {"juego": juego, "noticias": noticias}
        if noticias and not any(n.get("traducido") for n in noticias) and any(parece_ingles(n.get("titulo") or "") for n in noticias):
            resultado["error_traduccion"] = error_traduccion()   # para diagnosticar desde /docs
        return resultado
    if tool == "similares":
        juego = _resolver(params.get("juego"))
        if not juego:
            return {"error": f"No encontré '{params.get('juego')}' en Steam para buscar parecidos."}
        return juegos_similares(juego["appid"])
    if tool == "precios":
        return buscar_precios(str(params.get("juego", "")))
    if tool == "info_juego":
        juego = _resolver(params.get("juego"))
        if not juego:
            return {"error": f"No encontré '{params.get('juego')}' en Steam."}
        info = get_game_info(juego["appid"])
        if not info:
            return {"error": f"No pude obtener la ficha de {juego['nombre']}."}
        info["buscado"] = params.get("juego")
        if params.get("atributo") and any(w in params["atributo"].lower() for w in PIDE_OPINION):
            try:
                info["resenas_destacadas"] = get_resenas_destacadas(juego["appid"])
            except Exception:
                info["resenas_destacadas"] = []
        if params.get("atributo"):
            info["respuesta_atributo"] = _responder_atributo(params["atributo"], info)
        return info
    if tool == "buscar_por_genero":
        genero = str(params.get("genero", "")).lower()
        if genero not in GENEROS:
            return {"error": f"No tengo el género '{genero}'. Prueba con terror, rpg, carreras, estrategia, indie, supervivencia..."}
        return buscar_juegos(
            genero,
            solo_ofertas=bool(params.get("solo_ofertas")),
            cantidad=int(params.get("cantidad") or 6),
            por_resenas=bool(params.get("por_resenas")),
            min_descuento=int(params.get("min_descuento") or 0),
        )
    return None


PIDE_OPINION = ["dicen", "opinan", "piensan", "comentan", "comentario", "reseña", "resena", "opinion", "opinión", "crític", "critic"]


def _tiene_categoria(info: dict, claves: list[str]) -> list[str]:
    return [c for c in info.get("categorias", []) if any(k in c.lower() for k in claves)]


def _responder_atributo(pregunta: str, info: dict) -> str | None:
    """Responde preguntas de sí/no sobre un juego usando sus datos de Steam."""
    p = pregunta.lower()
    nombre = info["nombre"]

    if any(w in p for w in ["requisit", "requerimient", "specs", "especificacion", "qué pc", "que pc", "computadora", "compu "]):
        minimos, recomendados = info.get("requisitos_minimos"), info.get("requisitos_recomendados")
        if not minimos and not recomendados:
            return f"Steam no publica requisitos de PC para {nombre}."
        texto = f"Requisitos mínimos de {nombre}:\n{minimos or 'N/D'}"
        if recomendados:
            texto += f"\n\nRecomendados:\n{recomendados}"
        return texto

    if any(w in p for w in ["cooperativ", "co-op", "coop"]):
        coop = _tiene_categoria(info, ["cooperativo", "co-op"])
        if coop:
            return f"Sí, {nombre} tiene modo cooperativo: {', '.join(coop)}."
        multi = _tiene_categoria(info, ["multijugador", "jcj", "pvp", "multi-player", "online"])
        extra = f" Lo que sí tiene: {', '.join(multi)}." if multi else " Según Steam es para un jugador."
        return f"No, {nombre} no tiene modo cooperativo.{extra}"

    if re.search(r"(?:qu[eé]|cu[aá]les?)\s+idiomas?|idiomas\s+(?:de|del|tiene|trae)", p):
        return f"{nombre} está en: {info.get('idiomas') or 'N/D'}"

    if any(w in p for w in ["multijugador", "multiplayer", "online", "en línea", "en linea", "amigos", "cooperativo", "co-op", "coop", "jugar con"]):
        encontradas = _tiene_categoria(info, ["multijugador", "cooperativo", "jcj", "pvp", "co-op", "multi-player", "online"])
        if encontradas:
            return f"Sí, {nombre} tiene: {', '.join(encontradas)}."
        return f"No, según Steam {nombre} es solo para un jugador."

    if any(w in p for w in ["español", "espanol", "idioma", "subtítulo", "subtitulo", "doblaje", "doblado"]):
        idiomas = info.get("idiomas") or ""
        if "español" in idiomas.lower():
            return f"Sí, {nombre} está en español. Idiomas: {idiomas}"
        return f"No aparece el español en {nombre}. Idiomas: {idiomas or 'N/D'}"

    if any(w in p for w in ["mando", "control", "gamepad"]):
        encontradas = _tiene_categoria(info, ["mando"])
        return f"Sí: {', '.join(encontradas)}." if encontradas else f"Steam no indica compatibilidad con mando para {nombre}."

    if any(w in p for w in ["mac", "linux", "windows", "plataforma", "steam deck"]):
        plataformas = ", ".join(info.get("plataformas") or []) or "N/D"
        return f"{nombre} está disponible en: {plataformas}."

    if any(w in p for w in ["oferta", "descuento", "rebaja"]):
        if info.get("descuento"):
            return f"Sí, {nombre} tiene {info['descuento']}% de descuento: {info['precio']} (antes {info['precio_original']})."
        return f"No, {nombre} no está en oferta ahorita. Precio: {info['precio']}."

    if any(w in p for w in ["gratis", "free", "cuesta", "precio", "vale", "cuánto", "cuanto"]):
        if "jugador" in p or "juega" in p:
            n = info.get("jugadores_ahora")
            return f"Ahorita hay {n:,} personas jugando {nombre}." if isinstance(n, int) else "No pude obtener los jugadores actuales."
        return f"{nombre} cuesta {info['precio']}."

    if any(w in p for w in ["bueno", "buenas", "vale la pena", "calificaci", "recomend", "malo"] + PIDE_OPINION):
        r = info.get("resenas")
        if not r:
            return f"{nombre} todavía no tiene suficientes reseñas en Steam."
        pct = r["porcentaje_positivas"]
        veredicto = ("muy bien valorado" if pct >= 80 else "bien valorado" if pct >= 70
                     else "con opiniones mixtas" if pct >= 40 else "mal valorado")
        texto = f"{nombre} está {veredicto}: {r['porcentaje_positivas']}% de reseñas positivas de {r['total_resenas']:,} ({r['calificacion']})."
        destacadas = info.get("resenas_destacadas") or []
        if destacadas:
            texto += "\n\nLo que dicen los jugadores:"
            for d in destacadas:
                horas = f" — {d['horas']:,} h jugadas" if d.get("horas") else ""
                texto += f"\n{'👍' if d['positiva'] else '👎'} «{d['texto']}»{horas}"
            if all(d.get("idioma") == "english" for d in destacadas):
                texto += "\n(No encontré reseñas en español; estas vienen en inglés.)"
        return texto

    if any(w in p for w in ["jugador", "juegan", "gente", "activo"]):
        n = info.get("jugadores_ahora")
        return f"Ahorita hay {n:,} personas jugando {nombre}." if isinstance(n, int) else "No pude obtener los jugadores actuales."

    return None  # sin respuesta específica: se muestra la ficha completa


# ---------------- Generar respuesta ----------------

def formatear_sin_llm(datos) -> str:
    """Respuesta legible cuando el LLM no está disponible."""
    if not datos:
        return ("No entendí qué buscar. Puedo ayudarte con:\n"
                "• Ofertas: ¿Qué ofertas buenas hay?\n"
                "• Tendencias: ¿Qué juegos están de moda?\n"
                "• Géneros: juegos de terror, ofertas de RPG\n"
                "• Un juego: ¿Sabes de Elden Ring?\n"
                "• Noticias: ¿Qué actualizaciones tuvo Counter-Strike 2?\n"
                "• Comparar: Compara Palworld contra Elden Ring")

    if isinstance(datos, dict) and datos.get("error"):
        return datos["error"]

    lineas = []
    sugerencia = None
    if isinstance(datos, dict) and "resultados" in datos:
        if not datos["resultados"]:
            return f"No encontré juegos que coincidan con '{datos['buscado']}' en Steam."
        lineas.append(f"Precios en Steam para '{datos['buscado']}':")
        for j in datos["resultados"]:
            precio = j["precio"]
            if j.get("descuento"):
                precio += f" ({j['descuento']}% de descuento, antes {j['precio_original']})"
            lineas.append(f"• {j['nombre']} — {precio}")
        sugerencia = f"¿Sabes de {datos['resultados'][0]['nombre']}?"
    elif isinstance(datos, dict) and datos.get("respuesta_atributo"):
        lineas.append(datos["respuesta_atributo"])
        sugerencia = f"¿tiene español?, ¿está en oferta?, ¿es multijugador? (sobre {datos['nombre']})"
    elif isinstance(datos, dict) and "jugadores_ahora" in datos:
        if datos.get("buscado") and datos["buscado"].lower() not in (datos["nombre"] or "").lower():
            lineas.append(f"No encontré '{datos['buscado']}' tal cual; lo más parecido en Steam es:")
        lineas.append(datos["nombre"])
        if datos.get("descripcion"):
            lineas.append(datos["descripcion"])
        precio = datos["precio"]
        if datos.get("descuento"):
            precio += f" ({datos['descuento']}% de descuento, antes {datos['precio_original']})"
        lineas.append(f"\n  Precio: {precio}")
        lineas.append(f"  Géneros: {', '.join(datos['generos']) or 'N/D'}")
        lineas.append(f"  Desarrollador: {datos['desarrollador']}")
        lineas.append(f"  Lanzamiento: {datos['fecha_lanzamiento']}")
        if isinstance(datos.get("jugadores_ahora"), int):
            lineas.append(f"  Jugando ahora: {datos['jugadores_ahora']:,}")
        if datos.get("resenas"):
            r = datos["resenas"]
            lineas.append(f"  Reseñas: {r['porcentaje_positivas']}% positivas ({r['total_resenas']:,})")
        sugerencia = f"¿Qué actualizaciones tuvo {datos['nombre']}?, o compáralo contra otro juego"
    elif isinstance(datos, dict) and "parecidos_a" in datos:
        if not datos["juegos"]:
            return f"No encontré juegos parecidos a {datos['parecidos_a']} en este momento."
        etiquetas = f" (comparten: {', '.join(datos['etiquetas'])})" if datos.get("etiquetas") else ""
        lineas.append(f"Si te gustó {datos['parecidos_a']}, te pueden gustar{etiquetas}:")
        for i, j in enumerate(datos["juegos"], 1):
            precio = j["precio"]
            if j.get("descuento"):
                precio += f" ({j['descuento']}% de descuento)"
            r = j.get("resenas")
            calif = f" — {r['porcentaje_positivas']}% positivas" if r else ""
            lineas.append(f"{i}. {j['nombre']} — {precio}{calif}")
        sugerencia = f"¿Sabes de {datos['juegos'][0]['nombre']}?"
    elif isinstance(datos, dict) and "juegos" in datos:
        if datos.get("por_resenas"):
            tipo = "mejor calificados (entre los más vendidos)"
        elif datos.get("min_descuento"):
            tipo = f"con {datos['min_descuento']}% o más de descuento"
        elif datos.get("solo_ofertas"):
            tipo = "en oferta"
        else:
            tipo = "más vendidos ahora"
        if datos.get("genero") == "Gratis":
            titulo = f"Juegos gratis {tipo}"
        else:
            titulo = f"Juegos de {datos['genero']} {tipo}" if datos.get("genero") else f"Juegos {tipo}"
        if not datos["juegos"]:
            return f"No encontré resultados para: {titulo.lower()}."
        lineas.append(f"{titulo} en Steam:")
        for i, j in enumerate(datos["juegos"], 1):
            precio = j["precio"]
            if j.get("descuento"):
                precio += f" ({j['descuento']}% de descuento, antes {j['precio_original']})"
            r = j.get("resenas")
            calif = f" — {r['porcentaje_positivas']}% positivas ({r['total_resenas']:,} reseñas)" if r else ""
            lineas.append(f"{i}. {j['nombre']} — {precio}{calif}")
        if not datos.get("solo_ofertas"):
            sugerencia = "¿Y en oferta?"
        else:
            sugerencia = f"¿Sabes de {datos['juegos'][0]['nombre']}?"
    elif isinstance(datos, list) and datos and "descuento" in datos[0]:
        minimo = datos[0].get("minimo")
        lineas.append(f"Ofertas en Steam con {minimo}% o más de descuento:" if minimo else "Mejores ofertas en Steam ahora mismo:")
        for o in sorted(datos, key=lambda x: int(x["descuento"].rstrip("%")), reverse=True):
            termina = f" — termina el {o['termina']}" if o.get("termina") else ""
            lineas.append(f"• {o['nombre']} — {o['precio_final']} ({o['descuento']} de descuento, antes {o['precio_original']}){termina}")
        sugerencia = "ofertas de terror, de RPG o de carreras"
    elif isinstance(datos, list) and datos and "peak_jugadores" in datos[0]:
        lineas.append("Juegos más jugados en Steam ahora mismo:")
        for j in datos:
            jugadores = f"{j['peak_jugadores']:,}" if isinstance(j.get("peak_jugadores"), int) else "N/D"
            lineas.append(f"{j['rank']}. {j['nombre']} — {jugadores} jugadores — {j['precio']}")
        sugerencia = f"¿Qué actualizaciones tuvo {datos[0]['nombre']}?, o juegos de un género (terror, RPG...)"
    elif isinstance(datos, dict) and "noticias" in datos:
        juego = (datos.get("juego") or {}).get("nombre")
        if not datos["noticias"]:
            lineas.append(f"{juego} no tiene anuncios oficiales recientes en Steam." if juego
                          else "No pude obtener noticias de Steam en este momento.")
        else:
            lineas.append(f"Noticias recientes de {juego}:" if juego else "Noticias recientes de Steam:")
        for n in datos["noticias"]:
            fecha = f" ({n['fecha']})" if n.get("fecha") else ""
            lineas.append(f"• {n['titulo']}{fecha}")
            if n.get("contenido"):
                lineas.append(f"  {n['contenido'][:200]}…")
        if any(n.get("traducido") for n in datos["noticias"]):
            lineas.append("\n(Traducción automática del inglés)")
        elif any(parece_ingles(n.get("titulo") or "") for n in datos["noticias"]):
            lineas.append("\n(Las noticias vienen en inglés, tal como las publica el desarrollador en Steam. "
                          "En este momento no pude traducirlas.)")
        sugerencia = "¿y de otro juego? (por ejemplo: ¿y de Dota 2?)"
    elif isinstance(datos, dict) and "juego1" in datos:
        lineas.append("Comparación:")
        for clave in ("juego1", "juego2"):
            j = datos.get(clave)
            if j:
                lineas.append(f"\n{j['nombre']}")
                lineas.append(f"  Precio: {j['precio']}")
                lineas.append(f"  Géneros: {', '.join(j['generos']) or 'N/D'}")
                lineas.append(f"  Desarrollador: {j['desarrollador']}")
                lineas.append(f"  Lanzamiento: {j['fecha_lanzamiento']}")
        sugerencia = "¿y contra otro juego? (por ejemplo: ¿y contra Hollow Knight?)"
    else:
        return json.dumps(datos, ensure_ascii=False, indent=2, default=str)

    if sugerencia:
        lineas.append(f"\nTambién puedes preguntar: {sugerencia}")
    return "\n".join(lineas)


def _compactar(datos):
    """Quita campos que el LLM no necesita (URLs, appids) para un prompt más corto."""
    if isinstance(datos, list):
        return [_compactar(d) for d in datos]
    if isinstance(datos, dict):
        return {k: _compactar(v) for k, v in datos.items() if k not in ("url", "appid", "categorias", "imagen", "traducido", "error_traduccion")}
    return datos


def generar_respuesta(pregunta: str, datos, contexto: dict | None = None) -> tuple[str, bool, str | None]:
    """Devuelve (respuesta, llm_usado, error_llm)."""
    system_prompt = """Eres SteamPulse, un asistente experto en videojuegos y Steam.
Responde en español, de forma clara y amigable.
Usa ÚNICAMENTE los datos proporcionados; no inventes cifras ni causas.
Si los datos no alcanzan para responder, dilo explícitamente.
Sé breve: máximo 150 palabras. No razones de más: solo resume los datos.
Al final sugiere en una línea qué más puede preguntar (ofertas, géneros, noticias o comparar juegos)."""
    anterior = f"Pregunta anterior: {contexto.get('pregunta')}\n" if contexto else ""
    prompt = (
        f"{anterior}Pregunta del usuario: {pregunta}\n\n"
        f"Datos de Steam: {json.dumps(_compactar(datos), ensure_ascii=False, default=str)}"
    )
    try:
        return call_llm(system_prompt, prompt, max_tokens=1200, proposito="respuesta"), True, None
    except PresupuestoAgotado as exc:
        return formatear_sin_llm(datos), False, str(exc)
    except requests.HTTPError as exc:
        detalle = exc.response.text[:300] if exc.response is not None else ""
        return formatear_sin_llm(datos), False, f"HTTP {exc.response.status_code}: {detalle}"
    except Exception as exc:
        return formatear_sin_llm(datos), False, f"{type(exc).__name__}: {exc}"


# ---------------- Agente principal ----------------

def steampulse(pregunta: str, contexto: dict | None = None) -> dict:
    """Responde una pregunta y la registra para observabilidad."""
    llamadas = []
    token = LLAMADAS_DETALLE.set(llamadas)
    try:
        resultado = _steampulse(pregunta, contexto)
    finally:
        LLAMADAS_DETALLE.reset(token)
    resultado["observabilidad"] = {"llm": llamadas}
    try:
        registrar_pregunta(resultado)
    except Exception:
        pass   # la observabilidad nunca debe tumbar una respuesta
    return resultado


def _steampulse(pregunta: str, contexto: dict | None = None) -> dict:
    request_id = str(uuid.uuid4())[:8]
    trace = []
    inicio_total = time.perf_counter()
    contador_llm = [0]
    LLAMADAS_LLM.set(contador_llm)

    def paso(nombre, inicio):
        trace.append({"paso": nombre, "ms": round((time.perf_counter() - inicio) * 1000)})

    # 1. Moderación
    if any(p in pregunta.lower() for p in PALABRAS_BLOQUEADAS):
        return {
            "request_id": request_id,
            "pregunta": pregunta,
            "status": "bloqueada",
            "respuesta": "Solicitud bloqueada por moderación.",
            "trace": [{"paso": "moderacion", "ms": 0}],
        }

    # 2. Decidir herramienta (primero se traducen abreviaturas: rdr2, gta5, cs2…)
    t = time.perf_counter()
    decision, origen = decidir_tool(expandir_alias(pregunta), contexto)
    paso(f"router:{origen}:{decision['tool']}", t)

    # 3. Ejecutar herramienta (o reutilizar el resultado si ya está en caché)
    datos, error, desde_cache = None, None, False
    if decision["tool"] != "none":
        t = time.perf_counter()
        datos = cache_get("datos", decision)
        if datos is not None:
            desde_cache = True
            paso(f"cache:{decision['tool']}", t)
        else:
            try:
                datos = ejecutar_tool(decision)
                cache_set("datos", decision, datos)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            paso(f"tool:{decision['tool']}", t)

    # 4. Generar respuesta
    t = time.perf_counter()
    error_llm = None
    respuesta_guardada = cache_get("respuesta", decision) if not error else None
    if respuesta_guardada:
        respuesta, llm_usado = respuesta_guardada, False
        desde_cache = True
    elif decision["tool"] == "none" or decision["tool"] not in LLM_PARA:
        # Respuestas con formato fijo: no gastan créditos
        respuesta, llm_usado = formatear_sin_llm(datos), False
    elif error:
        respuesta, llm_usado = "Steam no respondió en este momento. Intenta de nuevo en unos minutos.", False
    elif time.perf_counter() - inicio_total > TIEMPO_MAX_PARA_LLM:
        # Límite de tiempo total: si ya tardamos mucho, no esperamos además al LLM
        respuesta, llm_usado = formatear_sin_llm(datos), False
        error_llm = f"Se omitió el LLM: la pregunta ya llevaba más de {TIEMPO_MAX_PARA_LLM:.0f} s"
    else:
        respuesta, llm_usado, error_llm = generar_respuesta(pregunta, datos, contexto)
        if llm_usado:
            cache_set("respuesta", decision, respuesta)
    if respuesta_guardada:
        paso("respuesta_en_cache", t)
    elif llm_usado:
        paso("llm", t)
    elif decision["tool"] == "none" or decision["tool"] not in LLM_PARA:
        paso("formato_fijo", t)
    else:
        paso("fallback_sin_llm", t)

    return {
        "request_id": request_id,
        "pregunta": pregunta,
        "tool": decision["tool"],
        "params": decision.get("params", {}),
        "router": origen,
        "llm_usado": llm_usado,
        "desde_cache": desde_cache,
        "status": "error_tool" if error else "completed",
        "error": error,
        "error_llm": error_llm,
        "respuesta": respuesta,
        "datos": datos,
        "trace": trace,
        "limites": {
            "llamadas_llm": contador_llm[0],
            "tiempo_total_ms": round((time.perf_counter() - inicio_total) * 1000),
            "llm_restantes_hoy": estado_llm()["restantes_hoy"],
        },
    }
