"""Límites de uso de SteamPulse: preguntas por usuario y presupuesto diario del LLM.

Se guardan en memoria del servidor: si Render reinicia el servicio, los contadores
vuelven a cero. Para el proyecto es suficiente; en producción irían en una base de datos.
"""

import os
import threading
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from datetime import datetime
from zoneinfo import ZoneInfo

ZONA_MX = ZoneInfo("America/Mexico_City")

PREGUNTAS_POR_MINUTO = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
LLM_LLAMADAS_POR_DIA = int(os.getenv("LLM_DAILY_LIMIT", "100"))
TIEMPO_MAX_PARA_LLM = float(os.getenv("LLM_TIME_BUDGET_S", "40"))  # si ya pasaron 40 s, no se llama al LLM

_lock = threading.Lock()
_peticiones: dict[str, deque] = defaultdict(deque)
_llm = {"dia": None, "usadas": 0}

# Llamadas al LLM de la pregunta en curso (para mostrarlas en la traza)
LLAMADAS_LLM: ContextVar[list | None] = ContextVar("llamadas_llm", default=None)


class PresupuestoAgotado(Exception):
    """Se alcanzó el tope diario de llamadas al LLM."""


def _hoy() -> str:
    return datetime.now(ZONA_MX).strftime("%Y-%m-%d")


def permitir_pregunta(cliente: str) -> tuple[bool, int]:
    """Ventana deslizante de 60 s por cliente. Devuelve (permitido, segundos_de_espera)."""
    ahora = time.time()
    with _lock:
        cola = _peticiones[cliente]
        while cola and ahora - cola[0] > 60:
            cola.popleft()
        if len(cola) >= PREGUNTAS_POR_MINUTO:
            return False, int(60 - (ahora - cola[0])) + 1
        cola.append(ahora)
        return True, 0


def consumir_llamada_llm() -> None:
    """Cuenta una llamada al LLM; lanza PresupuestoAgotado si ya no hay cupo hoy."""
    with _lock:
        if _llm["dia"] != _hoy():
            _llm["dia"], _llm["usadas"] = _hoy(), 0
        if _llm["usadas"] >= LLM_LLAMADAS_POR_DIA:
            raise PresupuestoAgotado(f"Se alcanzó el límite de {LLM_LLAMADAS_POR_DIA} llamadas al LLM por día")
        _llm["usadas"] += 1
    contador = LLAMADAS_LLM.get()
    if contador is not None:
        contador[0] += 1


def estado_llm() -> dict:
    with _lock:
        usadas = _llm["usadas"] if _llm["dia"] == _hoy() else 0
    return {"usadas_hoy": usadas, "limite_diario": LLM_LLAMADAS_POR_DIA, "restantes_hoy": max(0, LLM_LLAMADAS_POR_DIA - usadas)}
