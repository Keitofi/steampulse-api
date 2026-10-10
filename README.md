# SteamPulse API

Agente de inteligencia gaming en tiempo real con datos de la API pública de Steam
y síntesis con el LLM del curso.

## Endpoints
- `GET /health` — estado del servicio
- `GET /trending` — juegos más jugados ahora
- `GET /deals` — mejores ofertas
- `GET /compare?juego1=Palworld&juego2=Elden Ring` — comparar dos juegos
- `GET /news?juego=Counter-Strike 2` — noticias de un juego (sin `juego` = noticias generales)
- `POST /ask` con `{"pregunta": "..."}` — el agente decide la herramienta y responde

## Correr local
```
pip install -r requirements.txt
cp .env.example .env   # y pon tu API_KEY
uvicorn main:app --reload --port 8000
```

## Render
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Variables: `BASE_URL`, `API_KEY`
