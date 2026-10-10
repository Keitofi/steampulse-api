"""Agente SteamPulse: decide la herramienta, ejecuta y sintetiza con el LLM del curso."""

import json
import os
import re
import time
import uuid

import requests

from steam_tools import (
    GENEROS,
    buscar_appid,
    buscar_por_genero,
    compare_games,
    get_best_deals,
    get_steam_news,
    get_trending_games,
)

TOOLS_PERMITIDAS = {
    "get_trending_games",
    "get_best_deals",
    "compare_games",
    "get_steam_news",
    "buscar_por_genero",
    "none",
}

TOOLS_DESCRIPCION = """
1. get_trending_games — Juegos más jugados en Steam ahora mismo. params: {}
2. get_best_deals — Mejores ofertas y descuentos activos. params: {}
3. compare_games — Compara dos juegos. params: {"juego1": "nombre", "juego2": "nombre"}
4. get_steam_news — Noticias o actualizaciones de un juego. params: {"juego": "nombre"} (vacío = noticias generales de Steam)
5. buscar_por_genero — Juegos populares de un género (terror, rpg, carreras, estrategia...). params: {"genero": "terror", "solo_ofertas": false}
6. none — La pregunta no necesita datos de Steam. params: {}
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

    genero = _genero_en(pregunta)
    if genero:
        solo_ofertas = pide_ofertas or previo == "get_best_deals"
        return {"tool": "buscar_por_genero", "params": {"genero": genero, "solo_ofertas": solo_ofertas}}

    if pide_ofertas:
        # "¿y en oferta?" después de buscar un género → ofertas de ese género
        if previo == "buscar_por_genero" and params_previos.get("genero"):
            return {"tool": "buscar_por_genero", "params": {"genero": params_previos["genero"], "solo_ofertas": True}}
        return {"tool": "get_best_deals", "params": {}}

    if any(w in p for w in ["noticia", "actualizaci", "update", "parche", "novedad"]):
        juego = _juego_de_noticias(pregunta)
        return {"tool": "get_steam_news", "params": {"juego": juego} if juego else {}}

    if any(w in p for w in ["tendencia", "moda", "más jugado", "mas jugado", "popular", "jugadores"]):
        return {"tool": "get_trending_games", "params": {}}

    # Seguimientos usando la pregunta anterior
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
        if decision.get("tool") in TOOLS_PERMITIDAS:
            decision.setdefault("params", {})
            return decision, "llm"
    except Exception:
        pass
    return decidir_por_reglas(pregunta, contexto), "reglas"


# ---------------- Ejecutar herramienta ----------------

def _resolver(nombre: str | None) -> dict | None:
    if not nombre:
        return None
    return buscar_appid(nombre)


def ejecutar_tool(decision: dict):
    tool = decision.get("tool")
    params = decision.get("params", {}) or {}

    if tool == "get_trending_games":
        return get_trending_games()
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
    if tool == "buscar_por_genero":
        genero = str(params.get("genero", "")).lower()
        if genero not in GENEROS:
            return {"error": f"No tengo el género '{genero}'. Prueba con terror, rpg, carreras, estrategia, indie, supervivencia..."}
        return buscar_por_genero(genero, solo_ofertas=bool(params.get("solo_ofertas")))
    return None


# ---------------- Generar respuesta ----------------

def formatear_sin_llm(datos) -> str:
    """Respuesta legible cuando el LLM no está disponible."""
    if not datos:
        return "No encontré datos para esa pregunta. Prueba con ofertas, tendencias, noticias de un juego o comparar dos juegos."

    if isinstance(datos, dict) and datos.get("error"):
        return datos["error"]

    lineas = []
    sugerencia = None
    if isinstance(datos, dict) and "juegos" in datos and "genero" in datos:
        tipo = "en oferta" if datos.get("solo_ofertas") else "más vendidos"
        if not datos["juegos"]:
            return f"No encontré juegos de {datos['genero']} {tipo} en este momento."
        lineas.append(f"Juegos de {datos['genero']} {tipo} en Steam:")
        for j in datos["juegos"]:
            precio = j["precio"]
            if j.get("descuento"):
                precio += f" ({j['descuento']}% de descuento, antes {j['precio_original']})"
            lineas.append(f"• {j['nombre']} — {precio}")
        sugerencia = "¿Y en oferta?" if not datos.get("solo_ofertas") else f"Compara {datos['juegos'][0]['nombre']} contra otro juego"
    elif isinstance(datos, list) and datos and "descuento" in datos[0]:
        lineas.append("Mejores ofertas en Steam ahora mismo:")
        for o in sorted(datos, key=lambda x: int(x["descuento"].rstrip("%")), reverse=True):
            lineas.append(f"• {o['nombre']} — {o['precio_final']} ({o['descuento']} de descuento, antes {o['precio_original']})")
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
    except requests.HTTPError as exc:
        detalle = exc.response.text[:300] if exc.response is not None else ""
        return formatear_sin_llm(datos), False, f"HTTP {exc.response.status_code}: {detalle}"
    except Exception as exc:
        return formatear_sin_llm(datos), False, f"{type(exc).__name__}: {exc}"


# ---------------- Agente principal ----------------

def steampulse(pregunta: str, contexto: dict | None = None) -> dict:
    request_id = str(uuid.uuid4())[:8]
    trace = []

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
    }
