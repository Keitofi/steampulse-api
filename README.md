# 🎮 SteamPulse

Agente de inteligencia gaming que responde preguntas sobre Steam en español: ofertas, tendencias, noticias, géneros, fichas de juegos, requisitos, idiomas y comparaciones. Usa datos reales de las APIs de Steam y un LLM para redactar las respuestas.

**En vivo:** https://steampulse-api.onrender.com
**Documentación de la API:** https://steampulse-api.onrender.com/docs
**Observabilidad:** https://steampulse-api.onrender.com/observabilidad

---

## Cómo funciona

```
Pregunta → reglas / NER (agent.py) → herramienta de Steam (steam_tools.py) → LLM redacta → respuesta
                    │                                                         │
                    └── si las reglas no entienden, decide el LLM             └── si el LLM falla, respuesta de respaldo sin LLM
```

1. **Router híbrido:** primero reglas (regex, diccionario de alias como `rdr2` o `gta`, fuzzy matching para errores de dedo). Solo si no entienden la pregunta se le pregunta al LLM qué herramienta usar.
2. **Herramientas:** ofertas (con descuento mínimo), tendencias, noticias (con imágenes y traducción), géneros, ficha de un juego, requisitos, idiomas, precios, juegos similares y comparación.
3. **Respuesta:** el LLM redacta usando únicamente los datos de Steam. Si no está disponible, se arma una respuesta con formato fijo.
4. **Memoria corta:** el chat manda la pregunta anterior para entender seguimientos como "¿y en oferta?".

## Estructura

| Archivo | Qué hace |
|---|---|
| `main.py` | API con FastAPI: `/ask`, `/health`, `/metrics`, chat web y panel |
| `agent.py` | Router (reglas + LLM), ejecución de herramientas y redacción de respuestas |
| `steam_tools.py` | Consultas a las APIs de Steam y traducción |
| `limits.py` | Control de costo: preguntas por minuto, tope diario del LLM y presupuesto de tiempo |
| `observability.py` | Logs `[obs]`, detalle de llamadas al LLM y métricas |
| `static/index.html` | Chat web |
| `static/observabilidad.html` | Panel de métricas |

## Correrlo en tu computadora

Requisitos: Python 3.11 o más reciente.

```bash
# 1. Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar las variables
copy .env.example .env          # Windows  (Mac/Linux: cp .env.example .env)
# Abre .env y pon tu propia API_KEY

# 4. Arrancar
uvicorn main:app --reload
```

Abre http://127.0.0.1:8000 para el chat o http://127.0.0.1:8000/docs para probar la API.

> Sin `API_KEY` la app funciona igual, pero responde con el modo de respaldo (sin LLM).

## Variables de entorno

| Variable | Para qué | Valor por defecto |
|---|---|---|
| `API_KEY` | Clave del LLM del curso (**obligatoria para usar el LLM**) | — |
| `BASE_URL` | Servidor del LLM | `https://api.artesaniadigital.org` |
| `RATE_LIMIT_PER_MIN` | Preguntas por minuto por usuario | `10` |
| `LLM_DAILY_LIMIT` | Llamadas al LLM por día | `100` |
| `LLM_TIME_BUDGET_S` | Si una pregunta ya tardó esto, no se llama al LLM | `40` |
| `MYMEMORY_EMAIL` | (Opcional) sube la cuota del traductor MyMemory | — |

## Desplegar en Render

1. Sube el repo a GitHub (el `.env` **nunca** se sube; ya está en `.gitignore`).
2. En Render: **New → Web Service** y conecta el repo.
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Environment:** agrega `API_KEY` y `BASE_URL`.
6. Cada `git push` vuelve a desplegar automáticamente.

## Limitaciones conocidas

- Los contadores de límites y las métricas viven en memoria: si Render duerme o reinicia el servicio, vuelven a cero (los logs sí se conservan).
- En el plan gratis, Render duerme el servicio tras un rato sin uso; la primera pregunta puede tardar hasta un minuto.
- Los traductores gratuitos pueden bloquear a Render por cuota; en ese caso las noticias se muestran en inglés.
- Juegos que no están en Steam (Minecraft, Fortnite, Valorant…) se informan como tales.
