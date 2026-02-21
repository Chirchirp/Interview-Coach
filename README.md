# 🎙️ Coach Alex — Interview Coach AI

> *"Your personal AI interview coach — always honest, always in your corner."*

A fully standalone, production-ready interview coaching application powered by AI.
Two modes: **cloud API** (Groq is free) or **Ollama local** (100% free, 100% private).

---

## ✨ What Coach Alex Does

| Feature | Description |
|---------|-------------|
| 🎤 **Live Practice Session** | Alex asks 10 tailored interview questions based on your resume and target job |
| ⭐ **STAR Grading** | Every answer scored on Situation / Task / Action / Result (0–100) |
| 💬 **Follow-up Coaching** | After each answer, discuss with Alex — probe deeper, reframe, improve |
| 💡 **Real-Time Hints** | Stuck? Ask Alex for a tip before you answer — specific to your background |
| 📊 **Full Session Report** | Category breakdown, top strengths, priority improvements, personal action plan |
| 💬 **Free Chat** | Chat with Alex anytime — nerves, salary, career pivots, positioning |
| 📁 **Resume + JD Upload** | Upload PDF/DOCX/TXT — Alex tailors everything to your actual materials |
| 🖥️ **Ollama Local Mode** | Zero API cost, zero data leaving your machine |

---

## 🚀 Quick Start

### Option A — Groq (Cloud, FREE, Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
streamlit run app.py

# 3. In the app:
#    - Select "Groq — FREE ⚡"
#    - Paste your key from https://console.groq.com/keys
#    - Click Connect
#    - Go to Setup → upload resume + JD → Start Session
```

### Option B — Ollama (Local, FREE, Private)

```bash
# 1. Install Ollama
#    Mac/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

#    Windows: download from https://ollama.ai/download

# 2. Pull a model (pick one)
ollama pull llama3.1        # Best balance — recommended
ollama pull llama3.2        # Fastest (3B)
ollama pull mistral         # Excellent quality
ollama pull phi3            # Lightweight, good for lower-end hardware

# 3. Install app dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py

# 5. In the app:
#    - Select "Ollama — Local FREE 🖥️"
#    - Leave URL blank (default: http://localhost:11434)
#    - Click Connect
#    - Select your pulled model
```

### Option C — Docker

```bash
# Build and run
docker-compose up -d

# Visit http://localhost:8501

# To also run Ollama in Docker, uncomment the ollama service in docker-compose.yml
```

---

## 📁 Project Structure

```
interview_coach/
├── app.py                    # Full Streamlit application (single file)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .streamlit/
│   └── config.toml           # Theme: warm cream, teal accent
└── src/
    ├── core/
    │   └── llm.py            # All AI functions — token-efficient
    └── utils/
        └── file_parser.py    # PDF / DOCX / TXT extraction
```

---

## 🧠 AI Provider Comparison

| Provider | Cost | Privacy | Speed | Quality | Setup |
|----------|------|---------|-------|---------|-------|
| **Groq** | Free | Cloud | ⚡⚡⚡ | ⭐⭐⭐⭐ | 30s (get key) |
| **Ollama** | Free | Local 🔒 | ⚡⚡ | ⭐⭐⭐⭐ | 5 min (install) |
| **OpenRouter** | Free tier | Cloud | ⚡⚡ | ⭐⭐⭐⭐ | 1 min |
| **OpenAI** | Paid | Cloud | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 1 min |
| **Anthropic** | Paid | Cloud | ⚡⚡ | ⭐⭐⭐⭐⭐ | 1 min |

---

## 🎯 Session Flow

```
Upload Resume + JD
        ↓
Alex Analyses Materials
(builds personalised 10-question plan)
        ↓
Question 1 of 10
  → Get a Hint (optional)
  → Type your answer
  → Submit
        ↓
Alex Grades You
  → Score 0-100 + Grade A-F
  → STAR breakdown (S/T/A/R each /25)
  → What worked ✅
  → What to improve ⚡
  → Model answer
  → Follow-up question preview
  → Option to discuss with Alex 💬
        ↓
Next Question → Repeat × 10
        ↓
Full Session Report
  → Overall grade
  → Category scores
  → Top 3 strengths
  → Priority improvements with fixes
  → Personal 4-step action plan
  → Alex's personal note
  → Download as TXT
```

---

## 🔑 Getting API Keys

- **Groq (FREE)**: https://console.groq.com/keys
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys
- **OpenRouter (free tier)**: https://openrouter.ai/keys
- **Ollama (local)**: https://ollama.ai/download

---

## 💸 Token Usage (per session)

| Feature | Tokens (approx) | Cost on GPT-4o Mini |
|---------|-----------------|---------------------|
| Session plan (10 Qs) | ~1,000 | ~$0.0003 |
| Grade each answer | ~500 × 10 | ~$0.001 |
| Hint per question | ~300 | ~$0.0001 |
| Follow-up chat turn | ~700 | ~$0.0002 |
| Full session report | ~1,200 | ~$0.0004 |
| **Full session total** | **~7,000** | **~$0.002** |

With **Groq or Ollama**, all of this is **free**.

---

## 🚢 Deploy to Cloud

### Streamlit Community Cloud (Free)
1. Push to GitHub
2. Go to share.streamlit.io → New app
3. Point to your repo → `app.py`
4. Done — public URL instantly

### Railway
```bash
railway login
railway init
railway up
```

### Render
1. Connect GitHub repo
2. Create Web Service → `streamlit run app.py --server.port $PORT`
3. Deploy

---

## 🔒 Privacy

- **API mode**: your resume and answers are sent to the AI provider (Groq/OpenAI/etc.)
- **Ollama mode**: everything stays on your machine — no data ever leaves your device
- No data is stored by this application between sessions
