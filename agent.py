"""Agente SteamPulse: decide la herramienta, ejecuta y sintetiza con el LLM del curso."""

import json
import os
import re
import time
import uuid

import requests

from limits import LLAMADAS_LLM, TIEMPO_MAX_PARA_LLM, PresupuestoAgotado, consumir_llamada_llm, estado_llm
from steam_tools import (
    GENEROS,
    buscar_appid,
    buscar_juegos,
    buscar_precios,
    compare_games,
    get_best_deals,
    get_game_info,
    juegos_similares,
    get_steam_news,
    get_trending_games,
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
2. get_best_deals — Mejores ofertas y descuentos activos. params: {}
3. compare_games — Compara dos juegos. params: {"juego1": "nombre", "juego2": "nombre"}
4. get_steam_news — Noticias o actualizaciones de un juego. params: {"juego": "nombre"} (vacío = noticias generales de Steam)
5. buscar_por_genero — Juegos populares de un género (terror, rpg, carreras, estrategia...). params: {"genero": "terror", "solo_ofertas": false, "cantidad": 6, "por_resenas": false}
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


def call_llm(system_prompt: str, prompt: str, max_tokens: int = 600) -> str:
    base_url = os.getenv("BASE_URL", "https://api.artesaniadigital.org").rstrip("/")
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Falta la variable API_KEY")

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
    headers = {"X-Student-Key": api_key, "Content-Type": "application/json"}

    # Un reintento si el backend falla con error 5xx (suele ser transitorio)
    for intento in range(2):
        consumir_llamada_llm()  # cada intento cuenta contra el presupuesto diario
        resp = requests.post(f"{base_url}/v1/completions", headers=headers, json=payload, timeout=90)
        if resp.status_code < 500 or intento == 1:
            break
        time.sleep(2)

    resp.raise_for_status()
    text = _extract_text(resp.json())
    if not text:
        raise RuntimeError("El backend no devolvió texto")
    return text


# ---------------- Decidir herramienta ----------------

def _limpiar_nombre(nombre: str) -> str:
    return nombre.strip(" ¿?¡!.,;:\"'")


def _juegos_a_comparar(p: str) -> tuple[str, str] | None:
    m = re.search(r"compar\w*\s+(?:a\s+|el\s+|entre\s+)?(.+?)\s+(?:contra|vs\.?|versus|con|y)\s+(.+)", p, re.IGNORECASE)
    if m:
        j1, j2 = _limpiar_nombre(m.group(1)), _limpiar_nombre(m.group(2))
        if j1 and j2:
            return j1, j2
    return None


def _juego_de_noticias(p: str) -> str | None:
    m = re.search(r"(?:\btuvo\b|\btiene\b|\bsobre\b|\bdel\b|\bde\b|\bpara\b)\s+(?:el\s+|la\s+)?(.+)$", p, re.IGNORECASE)
    if m:
        nombre = _limpiar_nombre(m.group(1))
        if nombre and nombre.lower() not in {"steam", "hoy", "la semana", "esta semana", "ahora"}:
            return nombre
    return None


NUMEROS = {"un": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
           "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}


def _cantidad_en(p: str, por_defecto: int) -> int:
    m = re.search(r"\b(\d{1,2})\b", p)
    if m:
        return max(1, min(int(m.group(1)), 10))
    for palabra, n in NUMEROS.items():
        if re.search(rf"\b{palabra}\b", p.lower()):
            return n
    return por_defecto


def _pide_resenas(p: str) -> bool:
    return any(w in p for w in ["rankead", "calificad", "valorad", "reseña", "resena", "rating", "puntuad", "review", "mejor evaluad"])


def _pide_top(p: str) -> bool:
    return any(w in p for w in ["mejor", "top", "rankead", "ranking", "más vendid", "mas vendid",
                                "temporada", "recomi", "calificad", "valorad", "qué juego compro", "que juego compro"])


def _genero_en(p: str) -> str | None:
    p = p.lower()
    for clave in sorted(GENEROS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(clave)}(?!\w)", p):
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


def decidir_por_reglas(pregunta: str, contexto: dict | None = None) -> dict:
    """Router determinista: resuelve la mayoría de preguntas sin gastar tokens.

    `contexto` es la pregunta anterior ({pregunta, tool, params}) para seguimientos.
    """
    p = pregunta.lower()
    previo = (contexto or {}).get("tool")
    params_previos = (contexto or {}).get("params") or {}
    pide_ofertas = any(w in p for w in ["oferta", "descuento", "barato", "rebaja", "sale"])

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
        return {"tool": "buscar_por_genero", "params": {
            "genero": genero, "solo_ofertas": solo_ofertas,
            "cantidad": _cantidad_en(pregunta, 6), "por_resenas": _pide_resenas(p) or recomienda}}

    m = PATRON_PRECIO.search(pregunta)
    if m and _limpiar_nombre(m.group(1)):
        return {"tool": "precios", "params": {"juego": _limpiar_nombre(m.group(1))}}

    if pide_ofertas:
        # "¿y en oferta?" después de buscar un género → ofertas de ese género
        if previo == "buscar_por_genero" and params_previos.get("genero"):
            return {"tool": "buscar_por_genero", "params": {"genero": params_previos["genero"], "solo_ofertas": True}}
        return {"tool": "get_best_deals", "params": {}}

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
        texto = call_llm(system_prompt, entrada, max_tokens=1000)
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        decision = json.loads(match.group(0)) if match else {}
        if decision.get("tool") in TOOLS_PERMITIDAS and decision["tool"] != "none":
            decision.setdefault("params", {})
            return decision, "llm"
    except Exception:
        pass
    return _ultimo_recurso(pregunta, decidir_por_reglas(pregunta, contexto)), "reglas"


SALUDOS = {"hola", "gracias", "buenas", "buenos dias", "buenos días", "buenas tardes", "buenas noches",
           "ok", "va", "adios", "adiós", "que onda", "qué onda", "ayuda", "help", "si", "sí", "no"}


def _ultimo_recurso(pregunta: str, decision: dict) -> dict:
    """Si nada coincidió, prueba si el texto es el nombre de un juego (ej. 'five nights at freddy's')."""
    if decision["tool"] != "none":
        return decision
    texto = _limpiar_nombre(pregunta)
    if len(texto) < 4 or texto.lower() in SALUDOS or len(texto.split()) > 6:
        return decision
    try:
        encontrado = buscar_appid(texto)
    except Exception:
        return decision
    if encontrado and texto.lower() in (encontrado["nombre"] or "").lower():
        return {"tool": "precios", "params": {"juego": texto}}
    return decision


# ---------------- Ejecutar herramienta ----------------

def _resolver(nombre: str | None) -> dict | None:
    if not nombre:
        return None
    return buscar_appid(nombre)


def ejecutar_tool(decision: dict):
    tool = decision.get("tool")
    params = decision.get("params", {}) or {}

    if tool == "get_trending_games":
        return get_trending_games(limit=int(params.get("cantidad") or 10))
    if tool == "top_juegos":
        return buscar_juegos(cantidad=int(params.get("cantidad") or 5), por_resenas=bool(params.get("por_resenas")))
    if tool == "get_best_deals":
        return get_best_deals()
    if tool == "compare_games":
        j1 = _resolver(params.get("juego1"))
        j2 = _resolver(params.get("juego2"))
        if not j1 or not j2:
            return {"error": "No encontré uno de los juegos a comparar"}
        return compare_games(j1["appid"], j2["appid"])
    if tool == "get_steam_news":
        juego = _resolver(params.get("juego"))
        noticias = get_steam_news(appid=juego["appid"] if juego else None)
        return {"juego": juego, "noticias": noticias}
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
        )
    return None


def _tiene_categoria(info: dict, claves: list[str]) -> list[str]:
    return [c for c in info.get("categorias", []) if any(k in c.lower() for k in claves)]


def _responder_atributo(pregunta: str, info: dict) -> str | None:
    """Responde preguntas de sí/no sobre un juego usando sus datos de Steam."""
    p = pregunta.lower()
    nombre = info["nombre"]

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

    if any(w in p for w in ["bueno", "vale la pena", "reseña", "resena", "calificaci", "opinion", "opinión", "recomend", "malo"]):
        r = info.get("resenas")
        if not r:
            return f"{nombre} todavía no tiene suficientes reseñas en Steam."
        veredicto = "muy bien valorado" if r["porcentaje_positivas"] >= 80 else "con opiniones mixtas" if r["porcentaje_positivas"] >= 60 else "mal valorado"
        return f"{nombre} está {veredicto}: {r['porcentaje_positivas']}% de reseñas positivas de {r['total_resenas']:,} ({r['calificacion']})."

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
        elif datos.get("solo_ofertas"):
            tipo = "en oferta"
        else:
            tipo = "más vendidos ahora"
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
        lineas.append("Mejores ofertas en Steam ahora mismo:")
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
        lineas.append(f"Noticias recientes de {juego}:" if juego else "Noticias recientes de Steam:")
        for n in datos["noticias"]:
            fecha = f" ({n['fecha']})" if n.get("fecha") else ""
            lineas.append(f"• {n['titulo']}{fecha}")
            if n.get("contenido"):
                lineas.append(f"  {n['contenido'][:200]}…")
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
    lineas.append("\n(Respuesta generada sin el LLM)")
    return "\n".join(lineas)


def _compactar(datos):
    """Quita campos que el LLM no necesita (URLs, appids) para un prompt más corto."""
    if isinstance(datos, list):
        return [_compactar(d) for d in datos]
    if isinstance(datos, dict):
        return {k: _compactar(v) for k, v in datos.items() if k not in ("url", "appid", "categorias")}
    return datos


def generar_respuesta(pregunta: str, datos, contexto: dict | None = None) -> tuple[str, bool, str | None]:
    """Devuelve (respuesta, llm_usado, error_llm)."""
    system_prompt = """Eres SteamPulse, un asistente experto en videojuegos y Steam.
Responde en español, de forma clara y amigable.
Usa ÚNICAMENTE los datos proporcionados; no inventes cifras ni causas.
Si los datos no alcanzan para responder, dilo explícitamente.
Al final sugiere en una línea qué más puede preguntar (ofertas, géneros, noticias o comparar juegos)."""
    anterior = f"Pregunta anterior: {contexto.get('pregunta')}\n" if contexto else ""
    prompt = (
        f"{anterior}Pregunta del usuario: {pregunta}\n\n"
        f"Datos de Steam: {json.dumps(_compactar(datos), ensure_ascii=False, default=str)}"
    )
    try:
        return call_llm(system_prompt, prompt, max_tokens=2000), True, None
    except PresupuestoAgotado as exc:
        return formatear_sin_llm(datos), False, str(exc)
    except requests.HTTPError as exc:
        detalle = exc.response.text[:300] if exc.response is not None else ""
        return formatear_sin_llm(datos), False, f"HTTP {exc.response.status_code}: {detalle}"
    except Exception as exc:
        return formatear_sin_llm(datos), False, f"{type(exc).__name__}: {exc}"


# ---------------- Agente principal ----------------

def steampulse(pregunta: str, contexto: dict | None = None) -> dict:
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

    # 2. Decidir herramienta
    t = time.perf_counter()
    decision, origen = decidir_tool(pregunta, contexto)
    paso(f"router:{origen}:{decision['tool']}", t)

    # 3. Ejecutar herramienta
    datos, error = None, None
    if decision["tool"] != "none":
        t = time.perf_counter()
        try:
            datos = ejecutar_tool(decision)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        paso(f"tool:{decision['tool']}", t)

    # 4. Generar respuesta
    t = time.perf_counter()
    error_llm = None
    if error:
        respuesta, llm_usado = "Steam no respondió en este momento. Intenta de nuevo en unos minutos.", False
    elif time.perf_counter() - inicio_total > TIEMPO_MAX_PARA_LLM:
        # Límite de tiempo total: si ya tardamos mucho, no esperamos además al LLM
        respuesta, llm_usado = formatear_sin_llm(datos), False
        error_llm = f"Se omitió el LLM: la pregunta ya llevaba más de {TIEMPO_MAX_PARA_LLM:.0f} s"
    else:
        respuesta, llm_usado, error_llm = generar_respuesta(pregunta, datos, contexto)
    paso("llm" if llm_usado else "fallback_sin_llm", t)

    return {
        "request_id": request_id,
        "pregunta": pregunta,
        "tool": decision["tool"],
        "params": decision.get("params", {}),
        "router": origen,
        "llm_usado": llm_usado,
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
