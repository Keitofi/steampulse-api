"""Observabilidad de SteamPulse: en qué se va el tiempo, las llamadas al LLM y los tokens.

Tres salidas:
1. Una línea de log en JSON por pregunta ([obs] {...}) -> se ve en Render > Logs.
2. El detalle de cada llamada al LLM dentro de la respuesta de /ask (campo "observabilidad").
3. Métricas acumuladas en /metrics y un panel en /observabilidad.

Las métricas viven en memoria: si Render duerme o reinicia el servicio, vuelven a cero.
Los logs de Render sí se conservan.
"""

import json
import threading
import time
from collections import Counter, defaultdict, deque
from contextvars import ContextVar
from datetime import datetime
from zoneinfo import ZoneInfo

ZONA_MX = ZoneInfo("America/Mexico_City")
INICIO = time.time()

_lock = threading.Lock()
_preguntas: deque = deque(maxlen=500)      # últimas 500 preguntas con su detalle
_contadores: Counter = Counter()            # eventos sueltos (rechazos por límite, etc.)

# Llamadas al LLM de la pregunta en curso
LLAMADAS_DETALLE: ContextVar[list | None] = ContextVar("llamadas_detalle", default=None)


def _ahora() -> str:
    return datetime.now(ZONA_MX).strftime("%Y-%m-%d %H:%M:%S")


def estimar_tokens(texto: str) -> int:
    """Aproximación estándar: ~4 caracteres por token."""
    return max(1, len(texto or "") // 4)


def registrar_llamada_llm(proposito: str, ms: int, status: int | None, tokens_entrada: int,
                          tokens_salida: int, estimado: bool, error: str | None = None) -> None:
    """Se llama por cada intento al LLM (router o redacción de la respuesta)."""
    detalle = LLAMADAS_DETALLE.get()
    if detalle is None:
        return
    detalle.append({
        "proposito": proposito,            # "router" (decidir herramienta) o "respuesta" (redactar)
        "ms": ms,
        "status": status,
        "ok": status is not None and 200 <= status < 300,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "tokens_estimados": estimado,
        "error": error,
    })


def contar(evento: str) -> None:
    with _lock:
        _contadores[evento] += 1


def registrar_pregunta(resultado: dict) -> None:
    """Guarda el resumen de una pregunta y lo escribe en el log."""
    llamadas = resultado.get("observabilidad", {}).get("llm", [])
    datos = resultado.get("datos")
    evento = {
        "ts": _ahora(),
        "request_id": resultado.get("request_id"),
        "pregunta": (resultado.get("pregunta") or "")[:200],
        "tool": resultado.get("tool", "none"),
        "router": resultado.get("router"),
        "status": resultado.get("status"),
        "llm_usado": resultado.get("llm_usado"),
        "ms_total": (resultado.get("limites") or {}).get("tiempo_total_ms")
                    or sum(p["ms"] for p in resultado.get("trace", [])),
        "pasos": resultado.get("trace", []),
        "llm": llamadas,
        "error": resultado.get("error"),
        "error_llm": resultado.get("error_llm"),
        "error_traduccion": datos.get("error_traduccion") if isinstance(datos, dict) else None,
    }
    print("[obs] " + json.dumps(evento, ensure_ascii=False), flush=True)
    with _lock:
        _preguntas.append(evento)


def _p95(valores: list[int]) -> int | None:
    if not valores:
        return None
    ordenados = sorted(valores)
    return ordenados[min(len(ordenados) - 1, int(round(0.95 * (len(ordenados) - 1))))]


def _prom(valores: list[int]) -> int | None:
    return round(sum(valores) / len(valores)) if valores else None


def _tipo_paso(nombre: str) -> str:
    if nombre.startswith("router"):
        return "Decidir herramienta"
    if nombre.startswith("tool"):
        return "Consultar Steam"
    if nombre == "llm":
        return "Redactar con LLM"
    if nombre.startswith("fallback"):
        return "Redactar sin LLM (respaldo)"
    return nombre


def _tipo_error(llamada: dict) -> str:
    if llamada.get("status"):
        texto = (llamada.get("error") or "").lower()
        if llamada["status"] == 403 and "credit" in texto:
            return "403 sin créditos"
        return f"HTTP {llamada['status']}"
    return llamada.get("error") or "desconocido"


def metricas(estado_llm: dict | None = None) -> dict:
    with _lock:
        eventos = list(_preguntas)
        contadores = dict(_contadores)

    total = len(eventos)
    por_tool: dict[str, list[int]] = defaultdict(list)
    tiempo_por_paso: Counter = Counter()
    llamadas = []
    for e in eventos:
        por_tool[e["tool"] or "none"].append(e["ms_total"] or 0)
        for paso in e["pasos"]:
            tiempo_por_paso[_tipo_paso(paso["paso"])] += paso["ms"]
        llamadas.extend(e["llm"])

    tiempo_total = sum(tiempo_por_paso.values()) or 1
    ms_llm = [c["ms"] for c in llamadas]
    no_entendidas = [e for e in eventos if e["tool"] == "none"]
    errores = [e for e in eventos if e["error"] or e["error_llm"] or e["error_traduccion"]]

    return {
        "desde": datetime.fromtimestamp(INICIO, ZONA_MX).strftime("%Y-%m-%d %H:%M:%S"),
        "minutos_activo": round((time.time() - INICIO) / 60),
        "preguntas": {
            "total": total,
            "resueltas_por_reglas": sum(1 for e in eventos if e["router"] == "reglas" and e["tool"] != "none"),
            "resueltas_por_llm_router": sum(1 for e in eventos if e["router"] == "llm"),
            "no_entendidas": len(no_entendidas),
            "con_error_de_steam": sum(1 for e in eventos if e["status"] == "error_tool"),
            "rechazadas_por_limite": contadores.get("rechazada_por_limite", 0),
            "respuesta_con_llm": sum(1 for e in eventos if e["llm_usado"]),
            "respuesta_sin_llm": sum(1 for e in eventos if not e["llm_usado"] and e["tool"] != "none"),
        },
        "tiempo": {
            "ms_promedio": _prom([e["ms_total"] or 0 for e in eventos]),
            "ms_p95": _p95([e["ms_total"] or 0 for e in eventos]),
            # ¿En qué se va el tiempo? (suma de todos los pasos)
            "por_paso": [
                {"paso": paso, "ms": ms, "porcentaje": round(100 * ms / tiempo_total, 1)}
                for paso, ms in tiempo_por_paso.most_common()
            ],
        },
        "por_herramienta": [
            {"tool": tool, "preguntas": len(ms), "ms_promedio": _prom(ms), "ms_p95": _p95(ms)}
            for tool, ms in sorted(por_tool.items(), key=lambda kv: -len(kv[1]))
        ],
        "llm": {
            "llamadas": len(llamadas),
            "exitosas": sum(1 for c in llamadas if c["ok"]),
            "fallidas": sum(1 for c in llamadas if not c["ok"]),
            "por_proposito": dict(Counter(c["proposito"] for c in llamadas)),
            "errores": dict(Counter(_tipo_error(c) for c in llamadas if not c["ok"])),
            "ms_promedio": _prom(ms_llm),
            "ms_p95": _p95(ms_llm),
            "tokens_entrada": sum(c["tokens_entrada"] for c in llamadas if c["ok"]),
            "tokens_salida": sum(c["tokens_salida"] for c in llamadas if c["ok"]),
            "tokens_son_estimados": any(c["tokens_estimados"] for c in llamadas if c["ok"]),
            "presupuesto_hoy": estado_llm,
        },
        # Failure slices: lo que el agente no supo resolver, para mejorar las reglas
        "ultimas_no_entendidas": [{"ts": e["ts"], "pregunta": e["pregunta"]} for e in no_entendidas[-15:]][::-1],
        "mas_lentas": [
            {"ts": e["ts"], "pregunta": e["pregunta"], "tool": e["tool"], "ms": e["ms_total"]}
            for e in sorted(eventos, key=lambda e: -(e["ms_total"] or 0))[:5]
        ],
        "errores_recientes": [
            {"ts": e["ts"], "pregunta": e["pregunta"], "tool": e["tool"],
             "error": str(e["error"] or e["error_llm"] or e["error_traduccion"])[:160]}
            for e in errores[-10:]
        ][::-1],
    }
