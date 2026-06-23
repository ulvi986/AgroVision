# 🌱 AgroVision

AI-powered agronomy assistant. Ask questions about crop health, NDVI/NDWI,
irrigation, plant disease, and satellite/drone imagery — and get answers
grounded in your own saved fields.

Built with **Flask** + **LangChain** on top of **Google Gemini** (vision +
text). Deployed on **Railway**.

🔗 **Live:** https://agrovision-production-0d33.up.railway.app

---

## ✨ Features

- **Field-aware answers** — pick a tracking area in the sidebar and the
  assistant uses that field's real data (crop, coordinates, NDVI, NDWI) from
  the spatial database to answer.
- **Image analysis** — upload a field photo, drone shot, or satellite/NDVI map
  and get a structured agronomic diagnosis (problem, confidence, action plan).
- **On-topic guardrail** — a classification agent (and a separate vision
  classifier for images) keeps the assistant focused on agriculture / GIS /
  remote sensing and politely declines off-topic questions.
- **Personas** — answer as a *professional agronomist*, a *local farmer*, or in
  a neutral default voice.
- **Per-session memory** — each browser session has its own isolated chat
  history, so follow-up questions work without context bleeding between chats.
- **Feedback loop** — after a few replies the user is asked to rate the answers;
  feedback is stored in **Airtable**.

---

## 🏗️ How it works

```
Browser (templates/index.html)
        │
        ├── GET  /areas   → list of saved fields (from database.json)
        │
        └── POST /chat    → message + optional area_id + optional image
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Classification agent (Gemini)│  on/off-topic router
                   └──────────────┬───────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │   Agriculture agent (Gemini)  │  persona + spatial context
                   └──────────────────────────────┘
```

**Single source of truth:** both the sidebar dropdown and the chat context are
served from `database.json`. Editing that file updates the field list and the
answers — no frontend changes needed.

**Clean history:** the selected field's data is injected into the system prompt
for the *current turn only*, never stored in the conversation history. This
prevents a previously selected field from leaking into answers after the user
switches areas.

---

## 🔌 API

| Method | Route      | Description                                                        |
| ------ | ---------- | ------------------------------------------------------------------ |
| `GET`  | `/`        | Serves the chat UI.                                                |
| `GET`  | `/areas`   | Returns saved areas from `database.json` (id, name, crop, NDVI…).  |
| `POST` | `/chat`    | Chat. Form fields: `message`, `area_id`, `persona`, `session_id`, `file` (optional image). |
| `POST` | `/feedback`| Stores a rating + comment in Airtable. Form: `rating`, `comment`, `session_id`. |

---

## 🚀 Getting started

### 1. Clone & install

```bash
git clone https://github.com/ulvi986/AgroVision.git
cd AgroVision
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable             | Required | Purpose                                              |
| -------------------- | -------- | ---------------------------------------------------- |
| `GOOGLE_API_KEY`     | ✅       | Google Gemini API key (used by `langchain-google-genai`). |
| `GEMINI_API_KEY`     | ⬜       | Alternative Gemini key name.                          |
| `AIRTABLE_API_KEY`   | ⬜       | Personal access token for storing feedback.           |
| `AIRTABLE_BASE_ID`   | ⬜       | Airtable base ID (defaults to a built-in value).      |
| `AIRTABLE_TABLE_NAME`| ⬜       | Feedback table name (default `CRM`).                  |
| `LANGCHAIN_*`        | ⬜       | Optional LangSmith tracing.                           |

> `.env` is git-ignored — never commit your real keys. On Railway, set these in
> the service's **Variables** tab.

### 3. Run locally

```bash
python app.py
```

Open http://127.0.0.1:5000

---

## 🗃️ Field data (`database.json`)

Areas are defined per user. Add or edit fields here and both the dropdown and
the chat context update automatically:

```json
{
  "users": [
    {
      "user_id": "USR-1001",
      "name": "John Smith",
      "saved_areas": [
        {
          "area_id": "FIELD-01",
          "area_name": "North Wheat Field",
          "coordinates": [[40.45, 49.85], [40.46, 49.85], [40.46, 49.86], [40.45, 49.86]],
          "current_crop": "Winter Wheat",
          "baseline_metrics": { "average_ndvi": 0.68, "average_ndwi": -0.12 }
        }
      ]
    }
  ]
}
```

---

## ☁️ Deployment (Railway)

The app runs under `gunicorn` (see `Procfile`):

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```

Deploy the current code with the Railway CLI:

```bash
railway up
```

Make sure the required environment variables are set in the Railway service's
**Variables** tab.

---

## 🧰 Tech stack

- **Backend:** Flask, Gunicorn
- **AI / orchestration:** LangChain, Google Gemini (`gemini-2.5-pro`)
- **Frontend:** vanilla HTML/CSS/JS (single template)
- **Feedback storage:** Airtable
- **Hosting:** Railway

---

## 📁 Project structure

```
.
├── app.py              # Flask app, agents, routing, endpoints
├── database.json       # Saved fields (single source of truth)
├── templates/
│   └── index.html      # Chat UI
├── requirements.txt
├── Procfile            # gunicorn entrypoint for deployment
└── .env.example        # Environment variable template
```
