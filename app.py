"""
Interview Coach AI — app.py
Warm, editorial UI. Playfair Display + DM Sans. Teal & amber on cream.
"""
import streamlit as st, sys
from pathlib import Path

# ── Path setup — must happen before src imports ───────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# ── Core imports — fail loudly here rather than mid-page ─────────
try:
    from src.core.llm import (
        PROVIDER_MODELS, HELP_LINKS, verify_connection, call_llm,
        get_ollama_models, build_session_plan, build_field_plan,
        grade_answer, coach_followup, get_question_tip,
        build_session_report, free_chat, transcribe_audio,
    )
    from src.utils.file_parser import extract_text, clean
except ImportError as _e:
    st.error(f"Import error: {_e}. Check that src/ folder is committed to your repo.")
    st.stop()

st.set_page_config(
    page_title="Coach Alex — Interview Coach AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise session state ──────────────────────────────────────
DEFAULTS = {
    "provider": "groq",
    "api_key": "",
    "model": "llama-3.3-70b-versatile",
    "connected": False,
    "resume_text": "",
    "jd_text": "",
    "page": "home",           # home | setup | session | report | chat
    "session_plan": None,
    "current_q_idx": 0,
    "session_data": [],       # list of {question, category, answer, grade}
    "grading": False,
    "last_grade": None,
    "show_tip": False,
    "current_tip": "",
    "chat_messages": [],
    "session_report": None,
    "followup_mode": False,
    "followup_messages": [],
    "quick_field": "",
    "quick_exp": "Mid Level (3–5 yrs)",
    "setup_mode_radio": None,
    "voice_transcript": "",    # last voice→text result
    "voice_recording_b64": "", # raw b64 audio waiting to transcribe
}
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,800;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --cream:   #faf7f2;
  --paper:   #f4efe6;
  --teal:    #0d7377;
  --teal-lt: #14a8ad;
  --amber:   #d4813a;
  --amber-lt:#f0a96e;
  --ink:     #1a1a2e;
  --ink-mid: #3d3d5c;
  --ink-lt:  #7a7a96;
  --white:   #ffffff;
  --green:   #2d7a4f;
  --red:     #c0392b;
  --shadow:  0 2px 20px rgba(13,115,119,0.10);
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--cream) !important;
}

/* Hide Streamlit chrome */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
/* Deploy button lives in stToolbar — kept visible for Streamlit Cloud */

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--ink) !important;
  border-right: 3px solid var(--teal);
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #2a2a42 !important; border-color: var(--teal) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div { color: #e8e8f0 !important; }
[data-baseweb="popover"], [data-baseweb="menu"] { background: var(--white) !important; }
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] [role="option"] { background: var(--white) !important; color: var(--ink) !important; }
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover { background: #e0f5f5 !important; color: var(--teal) !important; }
[data-baseweb="popover"] [aria-selected="true"] { background: var(--teal) !important; color: white !important; }
[data-testid="stSidebar"] input {
  background: #2a2a42 !important; color: #e8e8f0 !important; border-color: var(--teal) !important;
}
[data-testid="stSidebar"] input::placeholder { color: #7a7a96 !important; }

/* ── Main layout ──────────────────────────────────────────────── */
.main .block-container { padding: 2rem 2.5rem; max-width: 1100px; }

/* ── Typography ───────────────────────────────────────────────── */
.coach-name {
  font-family: 'Playfair Display', serif;
  font-size: 2.6rem; font-weight: 800;
  color: var(--teal); letter-spacing: -1px;
}
.coach-tagline {
  font-family: 'DM Sans', sans-serif;
  font-size: 1rem; color: var(--ink-lt); margin-top: -4px;
  letter-spacing: 0.04em; text-transform: uppercase;
}
.section-head {
  font-family: 'Playfair Display', serif;
  font-size: 1.6rem; font-weight: 700; color: var(--ink);
  margin-bottom: 4px;
}
.section-sub {
  font-size: 0.9rem; color: var(--ink-lt); margin-bottom: 1.4rem;
}

/* ── Cards ────────────────────────────────────────────────────── */
.card {
  background: var(--white);
  border-radius: 16px;
  padding: 1.6rem 1.8rem;
  box-shadow: var(--shadow);
  border: 1px solid rgba(13,115,119,0.1);
  margin-bottom: 1.2rem;
}
.card-teal {
  background: linear-gradient(135deg, #0d7377 0%, #14a8ad 100%);
  border-radius: 16px; padding: 1.6rem 1.8rem; color: white;
  margin-bottom: 1.2rem;
}
.card-amber {
  background: linear-gradient(135deg, #d4813a 0%, #f0a96e 100%);
  border-radius: 16px; padding: 1.6rem 1.8rem; color: white;
  margin-bottom: 1.2rem;
}
.card-cream {
  background: var(--paper);
  border-radius: 16px; padding: 1.4rem 1.6rem;
  border: 1px solid rgba(212,129,58,0.2); margin-bottom: 1rem;
}

/* ── Chat bubbles ─────────────────────────────────────────────── */
.bubble-coach {
  background: var(--white);
  border-left: 4px solid var(--teal);
  border-radius: 0 14px 14px 14px;
  padding: 14px 18px; margin-bottom: 12px;
  box-shadow: var(--shadow);
  font-size: 0.95rem; color: var(--ink); line-height: 1.7;
}
.bubble-user {
  background: linear-gradient(135deg, #0d7377, #14a8ad);
  border-radius: 14px 0 14px 14px;
  padding: 14px 18px; margin-bottom: 12px;
  color: white; font-size: 0.95rem; line-height: 1.6;
}
.bubble-label {
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 5px;
}
.label-coach { color: var(--teal); }
.label-user  { color: rgba(255,255,255,0.7); text-align: right; }

/* ── Question card ────────────────────────────────────────────── */
.q-card {
  background: var(--white);
  border: 2px solid var(--teal);
  border-radius: 16px; padding: 1.8rem;
  margin-bottom: 1.2rem;
  position: relative;
}
.q-number {
  font-family: 'Playfair Display', serif;
  font-size: 3rem; font-weight: 800;
  color: rgba(13,115,119,0.12);
  position: absolute; top: 12px; right: 18px; line-height: 1;
}
.q-category {
  display: inline-block;
  background: var(--teal); color: white;
  border-radius: 100px; padding: 3px 14px;
  font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase;
  margin-bottom: 10px;
}
.q-text {
  font-family: 'Playfair Display', serif;
  font-size: 1.25rem; color: var(--ink); line-height: 1.55;
}

/* ── Score ring ───────────────────────────────────────────────── */
.score-ring {
  width: 100px; height: 100px;
  border-radius: 50%;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  font-family: 'Playfair Display', serif;
  font-weight: 800; margin: 0 auto;
}
.score-A { background: linear-gradient(135deg,#2d7a4f,#52b788); color:white; }
.score-B { background: linear-gradient(135deg,#2d7a4f,#74c69d); color:white; }
.score-C { background: linear-gradient(135deg,#d4813a,#f0a96e); color:white; }
.score-D { background: linear-gradient(135deg,#c0392b,#e55039); color:white; }
.score-F { background: linear-gradient(135deg,#922b21,#c0392b); color:white; }

/* ── STAR bar ─────────────────────────────────────────────────── */
.star-row {
  display:flex; align-items:center; gap:10px;
  margin-bottom:8px; font-size:0.85rem;
}
.star-label { width:90px; color:var(--ink-mid); font-weight:500; }
.star-track {
  flex:1; height:8px; background:#e8e8ee; border-radius:100px; overflow:hidden;
}
.star-fill { height:100%; border-radius:100px; background:linear-gradient(90deg,var(--teal),var(--teal-lt)); }
.star-val { width:36px; text-align:right; color:var(--teal); font-weight:700; font-size:0.82rem; }

/* ── Progress strip ───────────────────────────────────────────── */
.progress-strip {
  display:flex; gap:6px; margin-bottom:1.2rem;
}
.prog-dot {
  height:6px; flex:1; border-radius:100px; background:#e0e0ee;
}
.prog-done { background:var(--teal); }
.prog-current { background:var(--amber); }

/* ── Category badge colours ───────────────────────────────────── */
.cat-Opener      { background:#0d7377; }
.cat-Behavioral  { background:#6354a5; }
.cat-Technical   { background:#1565c0; }
.cat-Situational { background:#2d7a4f; }
.cat-Leadership  { background:#b45309; }
.cat-Cultural    { background:#c2185b; }
.cat-Gap         { background:#922b21; }
.cat-Closing     { background:#374151; }

/* ── Feedback blocks ──────────────────────────────────────────── */
.fb-green {
  background:#f0fdf4; border-left:3px solid #2d7a4f;
  border-radius:0 10px 10px 0; padding:10px 14px;
  font-size:0.88rem; color:#166534; margin-bottom:6px;
}
.fb-amber {
  background:#fffbeb; border-left:3px solid #d4813a;
  border-radius:0 10px 10px 0; padding:10px 14px;
  font-size:0.88rem; color:#92400e; margin-bottom:6px;
}
.fb-model {
  background:#f0f9ff; border:1px solid #7dd3fc;
  border-radius:12px; padding:14px 16px;
  font-size:0.9rem; color:#0c4a6e; line-height:1.7;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
  border: none !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--teal), var(--teal-lt)) !important;
  color: white !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(13,115,119,0.35) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--paper) !important;
  color: var(--ink) !important;
  border: 1px solid rgba(13,115,119,0.25) !important;
}

/* ── Inputs ───────────────────────────────────────────────────── */
textarea, input[type="text"] {
  border-radius: 10px !important;
  border-color: rgba(13,115,119,0.3) !important;
  font-family: 'DM Sans', sans-serif !important;
  background: var(--white) !important;
  color: var(--ink) !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 2px rgba(13,115,119,0.15) !important;
}

/* ── Divider ──────────────────────────────────────────────────── */
hr { border-color: rgba(13,115,119,0.12); margin: 1.5rem 0; }

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px; background: var(--paper);
  border-radius: 12px; padding: 5px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important; font-weight: 500 !important;
  color: var(--ink-lt) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--teal) !important; color: white !important;
}

/* ── Mode selector radio — styled as toggle tabs ─────────────── */
[data-testid="stRadio"] > div {
  display: flex !important;
  gap: 8px !important;
  flex-direction: row !important;
}
[data-testid="stRadio"] label {
  border: 1.5px solid rgba(13,115,119,0.25) !important;
  border-radius: 10px !important;
  padding: 10px 18px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  font-weight: 500 !important;
  color: var(--ink-mid) !important;
  background: var(--white) !important;
  flex: 1 !important;
  text-align: center !important;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: var(--teal) !important;
  border-color: var(--teal) !important;
  color: white !important;
  font-weight: 600 !important;
}
[data-testid="stRadio"] label span { pointer-events: none; }
[data-testid="stRadio"] input[type="radio"] { display: none !important; }

/* ── Checkbox row for focus areas ────────────────────────────── */
[data-testid="stCheckbox"] label {
  font-size: 0.88rem !important;
  color: var(--ink-mid) !important;
}

/* ── Select slider ────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stTickBar"] span {
  font-size: 0.7rem !important;
}

/* ── Report category bars ─────────────────────────────────────── */
.rpt-bar-wrap { margin-bottom: 10px; }
.rpt-bar-label { display:flex; justify-content:space-between; font-size:0.83rem; margin-bottom:3px; }
.rpt-bar-track { height:8px; background:#e8e8ee; border-radius:100px; }
.rpt-bar-fill  { height:8px; border-radius:100px; background:linear-gradient(90deg,#0d7377,#14a8ad); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# IMPORTS — moved to top of file for Streamlit Cloud compatibility
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _go(page): st.session_state.page = page; st.rerun()
def _api(): return st.session_state.api_key
def _prov(): return st.session_state.provider
def _model(): return st.session_state.model
def _ready(): return st.session_state.connected
def _has_context(): return bool(st.session_state.resume_text or st.session_state.jd_text)

GRADE_COLORS = {"A":"#2d7a4f","B":"#52b788","C":"#d4813a","D":"#e55039","F":"#c0392b"}

def _launch_session(plan):
    """Shared helper — stores plan and navigates to session page."""
    st.session_state.session_plan     = plan
    st.session_state.current_q_idx   = 0
    st.session_state.session_data     = []
    st.session_state.last_grade       = None
    st.session_state.show_tip         = False
    st.session_state.followup_mode   = False
    st.session_state.followup_messages= []
    st.session_state.session_report   = None
    _go("session")

def score_bar(label, val, max_val=25):
    pct = int(val / max_val * 100)
    st.markdown(f"""
    <div class="star-row">
      <span class="star-label">{label}</span>
      <div class="star-track"><div class="star-fill" style="width:{pct}%"></div></div>
      <span class="star-val">{val}/{max_val}</span>
    </div>""", unsafe_allow_html=True)


def rpt_bar(label, score):
    c = "#2d7a4f" if score>=80 else "#d4813a" if score>=60 else "#c0392b"
    st.markdown(f"""
    <div class="rpt-bar-wrap">
      <div class="rpt-bar-label">
        <span style="color:var(--ink-mid);font-weight:500">{label}</span>
        <span style="color:{c};font-weight:700">{score}/100</span>
      </div>
      <div class="rpt-bar-track"><div class="rpt-bar-fill" style="width:{score}%;background:{c}"></div></div>
    </div>""", unsafe_allow_html=True)


def progress_strip(total, current):
    dots = []
    for i in range(total):
        cls = "prog-done" if i < current else ("prog-current" if i == current else "prog-dot")
        dots.append(f'<div class="prog-dot {cls}"></div>')
    st.markdown(f'<div class="progress-strip">{"".join(dots)}</div>', unsafe_allow_html=True)


def coach_bubble(text, label="Coach Alex"):
    st.markdown(f"""
    <div class="bubble-coach">
      <div class="bubble-label label-coach">🎙️ {label}</div>
      {text}
    </div>""", unsafe_allow_html=True)


def user_bubble(text):
    st.markdown(f"""
    <div class="bubble-user">
      <div class="bubble-label label-user">You 👤</div>
      {text}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 16px">
      <div style="font-size:2.6rem">🎙️</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:800;
                  color:#14a8ad;margin-top:4px">Coach Alex</div>
      <div style="font-size:0.72rem;color:#7a7a96;text-transform:uppercase;
                  letter-spacing:0.1em;margin-top:2px">Interview Coach AI</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Provider
    st.markdown("<div style='font-size:11px;font-weight:600;color:#7a7a96;"
                "text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>"
                "🔌 AI Provider</div>", unsafe_allow_html=True)

    prov_labels = {
        "Groq — FREE ⚡":          "groq",
        "Ollama — Local FREE 🖥️":  "ollama",
        "OpenAI":                   "openai",
        "Anthropic (Claude)":       "anthropic",
        "OpenRouter":               "openrouter",
    }
    sel_prov = st.selectbox("Provider", list(prov_labels.keys()),
                             label_visibility="collapsed")
    provider = prov_labels[sel_prov]

    if provider == "ollama":
        st.markdown("""
        <div style="background:rgba(20,168,173,0.15);border:1px solid rgba(20,168,173,0.4);
                    border-radius:8px;padding:10px 12px;font-size:11.5px;color:#14a8ad;margin:6px 0">
            🖥️ <b>100% Free & Private</b><br>
            Runs on your machine.<br>
            <a href="https://ollama.ai" target="_blank" style="color:#14a8ad">
            Install Ollama →</a><br>
            Then: <code style="color:#f0a96e">ollama pull llama3.1</code>
        </div>""", unsafe_allow_html=True)

    # For Ollama: dynamically show actually-installed models
    if provider == "ollama":
        models = get_ollama_models(
            st.session_state.api_key if st.session_state.provider == "ollama" else ""
        )
    else:
        models = PROVIDER_MODELS.get(provider, {})
    sel_model = st.selectbox("Model", list(models.keys()), label_visibility="collapsed")

    placeholder = {"groq":"gsk_...","openai":"sk-...","anthropic":"sk-ant-...","openrouter":"sk-or-...",
                   "ollama":"http://localhost:11434 (or leave blank)"}
    api_key_in = st.text_input(
        "API Key" if provider != "ollama" else "Ollama URL",
        value=st.session_state.api_key if st.session_state.provider == provider else "",
        type="password" if provider != "ollama" else "default",
        placeholder=placeholder.get(provider, ""),
        key=f"key_{provider}"
    )

    if st.button("✅ Connect", type="primary", use_container_width=True):
        key_val = (api_key_in or "").strip()  # empty string is fine for Ollama — _ollama_base handles default
        if key_val or provider == "ollama":
            with st.spinner("Connecting…"):
                ok, msg = verify_connection(key_val, provider)
            if ok:
                st.session_state.api_key  = key_val
                st.session_state.provider = provider
                # For Ollama, re-fetch models now that we're connected
                if provider == "ollama":
                    live_models = get_ollama_models(key_val)
                    # Use first available model as default if current selection invalid
                    model_val = live_models.get(sel_model, list(live_models.values())[0])
                else:
                    model_val = models[sel_model]
                st.session_state.model    = model_val
                st.session_state.connected = True
                st.success(msg)
                st.rerun()
            else:
                st.session_state.connected = False
                st.error(msg)
        else:
            st.error("Enter your API key first.")

    if st.session_state.connected and st.session_state.provider == provider:
        st.markdown("""
        <div style="background:rgba(20,168,173,0.18);border:1px solid rgba(20,168,173,0.45);
                    border-radius:8px;padding:8px 12px;margin-top:8px;font-size:13px;
                    color:#14a8ad;font-weight:600">✓ Connected</div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(192,57,43,0.12);border:1px solid rgba(192,57,43,0.3);
                    border-radius:8px;padding:8px 12px;margin-top:8px;font-size:13px;
                    color:#e55039">⚠ Not connected</div>""", unsafe_allow_html=True)

    if provider in HELP_LINKS:
        lbl = "Install Ollama" if provider=="ollama" else f"Get API key"
        st.markdown(f"<div style='text-align:center;font-size:11px;margin-top:6px'>"
                    f"<a href='{HELP_LINKS[provider]}' target='_blank' "
                    f"style='color:#14a8ad'>{lbl} →</a></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    st.markdown("<div style='font-size:11px;font-weight:600;color:#7a7a96;"
                "text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>"
                "Navigation</div>", unsafe_allow_html=True)

    nav = [("🏠", "Home",            "home"),
           ("📋", "Setup Session",   "setup"),
           ("🎤", "Practice Session","session"),
           ("📊", "My Report",       "report"),
           ("💬", "Free Chat",       "chat")]

    for icon, label, key in nav:
        active = st.session_state.page == key
        if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True,
                     type="primary" if active else "secondary"):
            _go(key)

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(212,129,58,0.12);border-radius:10px;
                padding:12px;font-size:12px;color:#f0a96e">
        <b>💡 How it works:</b><br><br>
        1️⃣ Connect AI provider<br>
        2️⃣ Upload resume + job<br>
        3️⃣ Run practice session<br>
        4️⃣ Get graded + coached<br>
        5️⃣ Download your report
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;font-size:10px;color:#3d3d5c;margin-top:12px'>"
                "Coach Alex v1.0 · Built for job seekers</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div style="padding:2rem 0 1rem">
      <div class="coach-name">Meet Coach Alex.</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                  color:var(--amber);font-style:italic;margin-top:4px">
          Your personal AI interview coach — always honest, always in your corner.
      </div>
    </div>""", unsafe_allow_html=True)

    if not _ready():
        st.info("👈 **First step:** Connect your AI provider in the sidebar. "
                "**Groq is free** — [get a key in 30 seconds](https://console.groq.com/keys). "
                "Or **Ollama** runs entirely on your machine at zero cost.", icon="🔑")

    # Feature cards
    cols = st.columns(3, gap="medium")
    features = [
        ("🎤", "Live Practice Session",
         "Alex asks real interview questions tailored to your resume and the job. "
         "You answer. Alex grades you — honestly."),
        ("⭐", "STAR Method Grading",
         "Every answer scored across Situation, Task, Action, Result. "
         "See exactly where you're strong and where you're losing points."),
        ("📊", "Personal Session Report",
         "After 10 questions, get a full coaching report with your score, "
         "category breakdown, top gaps, and an action plan."),
        ("💡", "Real-Time Tips",
         "Stuck? Ask for a hint before you answer. Alex gives you "
         "specific guidance based on your actual background."),
        ("💬", "Free Chat Coach",
         "Outside of practice sessions, chat with Alex about anything — "
         "nerves, salary negotiation, how to position yourself."),
        ("🖥️", "100% Free with Ollama",
         "No API cost. No data leaving your machine. "
         "Run powerful local models for zero cost with Ollama."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card" style="height:100%;min-height:140px">
              <div style="font-size:1.8rem;margin-bottom:8px">{icon}</div>
              <div style="font-family:'Playfair Display',serif;font-size:1rem;
                          font-weight:700;color:var(--ink);margin-bottom:6px">{title}</div>
              <div style="font-size:0.87rem;color:var(--ink-lt);line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)
        if i == 2:
            cols2 = st.columns(3, gap="medium")
            for j in range(3):
                cols[j] = cols2[j]

    st.markdown("<br>", unsafe_allow_html=True)

    # CTA
    st.markdown("<br>", unsafe_allow_html=True)
    cta1, cta2 = st.columns(2, gap="medium")
    with cta1:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(13,115,119,0.08),rgba(13,115,119,0.02));
                    border:1.5px solid rgba(13,115,119,0.2);border-radius:14px;padding:20px 22px">
          <div style="font-size:1.5rem;margin-bottom:6px">📄</div>
          <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;
                      color:var(--teal);margin-bottom:6px">Full Session</div>
          <div style="font-size:0.85rem;color:var(--ink-lt);line-height:1.6;margin-bottom:12px">
              Upload your resume + job description for a fully personalised session
              tailored to your actual background and target role.
          </div>
        </div>""", unsafe_allow_html=True)
        if st.button("📄 Start Full Session", type="primary", use_container_width=True):
            st.session_state.setup_mode_radio = "📄 Full Session (Resume + Job Description)"
            _go("setup")
    with cta2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(212,129,58,0.08),rgba(212,129,58,0.02));
                    border:1.5px solid rgba(212,129,58,0.25);border-radius:14px;padding:20px 22px">
          <div style="font-size:1.5rem;margin-bottom:6px">⚡</div>
          <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;
                      color:var(--amber);margin-bottom:6px">Quick Session</div>
          <div style="font-size:0.85rem;color:var(--ink-lt);line-height:1.6;margin-bottom:12px">
              No documents needed. Just type your job field and experience level —
              Alex generates 10 industry-standard questions instantly.
          </div>
        </div>""", unsafe_allow_html=True)
        if st.button("⚡ Start Quick Session", use_container_width=True):
            st.session_state.setup_mode_radio = "⚡ Quick Session (Field Only — No Documents)"
            _go("setup")


# ══════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ══════════════════════════════════════════════════════════════════
def page_setup():
    if not _ready():
        st.warning("⚠️ Connect your AI provider in the sidebar first.")
        return

    # ── Mode selector ─────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div class="section-head">📋 Session Setup</div>
      <div class="section-sub">Choose how you want to practise — with your documents, or by simply typing your field.</div>
    </div>""", unsafe_allow_html=True)

    # Two-mode toggle
    mode_labels = ["📄 Full Session (Resume + Job Description)", "⚡ Quick Session (Field Only — No Documents)"]
    mode = st.radio("Session Mode", mode_labels, horizontal=True,
                    label_visibility="collapsed",
                    key="setup_mode_radio")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # MODE A — Full session with resume + JD
    # ══════════════════════════════════════════════════════════════
    if mode == mode_labels[0]:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(13,115,119,0.07),rgba(13,115,119,0.02));
                    border:1.5px solid rgba(13,115,119,0.2);border-radius:14px;
                    padding:1.2rem 1.4rem;margin-bottom:1.4rem">
          <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700;
                      color:var(--teal);margin-bottom:4px">📄 Full Personalised Session</div>
          <div style="font-size:0.87rem;color:var(--ink-lt);line-height:1.6">
              Upload or paste your resume and the job you're targeting.
              Alex will tailor every question to your actual background and the specific role —
              the most realistic interview practice possible.
          </div>
        </div>""", unsafe_allow_html=True)

        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown("#### 📄 Your Resume")
            up_tab, paste_tab = st.tabs(["📁 Upload File", "📝 Paste Text"])
            with up_tab:
                f = st.file_uploader("Resume", type=["pdf","docx","txt"],
                                     label_visibility="collapsed", key="resume_upload")
                if f:
                    t, err = extract_text(f)
                    if err: st.error(err)
                    else:
                        st.session_state.resume_text = clean(t)
                        st.success(f"✅ {f.name} — {len(t.split())} words extracted")
            with paste_tab:
                rt = st.text_area("Resume text", value=st.session_state.resume_text,
                                  height=260, label_visibility="collapsed",
                                  placeholder="Paste your full resume here…")
                st.session_state.resume_text = rt

        with col_r:
            st.markdown("#### 🏢 Job Description")
            up_jd, paste_jd = st.tabs(["📁 Upload File", "📝 Paste Text"])
            with up_jd:
                fj = st.file_uploader("JD", type=["pdf","docx","txt"],
                                      label_visibility="collapsed", key="jd_upload")
                if fj:
                    t, err = extract_text(fj)
                    if err: st.error(err)
                    else:
                        st.session_state.jd_text = clean(t)
                        st.success(f"✅ {fj.name} — {len(t.split())} words extracted")
            with paste_jd:
                jt = st.text_area("JD text", value=st.session_state.jd_text,
                                  height=260, label_visibility="collapsed",
                                  placeholder="Paste the full job description here…")
                st.session_state.jd_text = jt

        st.markdown("<br>", unsafe_allow_html=True)

        # Status indicators
        has_resume = bool(st.session_state.resume_text.strip())
        has_jd     = bool(st.session_state.jd_text.strip())
        c1, c2, c3 = st.columns(3)
        for col, ok, label in [(c1, has_resume, "Resume"),
                               (c2, has_jd,     "Job Description"),
                               (c3, _ready(),   f"AI ({st.session_state.provider.upper()})")]:
            with col:
                st.markdown(f"""
                <div class="card-cream" style="text-align:center;padding:12px">
                  <div style="font-size:1.4rem">{"✅" if ok else "⭕"}</div>
                  <div style="font-size:0.82rem;color:var(--ink-mid);margin-top:3px">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not has_resume and not has_jd:
            st.info("💡 You can start without documents — Alex will give general coaching. "
                    "For a fully personalised session, add both.")

        cb1, cb2 = st.columns([3, 1])
        with cb1:
            if st.button("🎤 Start Full Session", type="primary", use_container_width=True):
                with st.spinner("Alex is reviewing your materials and preparing your session…"):
                    try:
                        resume = st.session_state.resume_text or "No resume provided."
                        jd     = st.session_state.jd_text or "General interview — no specific role."
                        plan   = build_session_plan(_api(), _prov(), _model(), resume, jd)
                        _launch_session(plan)
                    except Exception as e:
                        st.error(f"Couldn't build session: {e}")
        with cb2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.resume_text = ""
                st.session_state.jd_text = ""
                st.rerun()

    # ══════════════════════════════════════════════════════════════
    # MODE B — Quick session, field only
    # ══════════════════════════════════════════════════════════════
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(212,129,58,0.08),rgba(212,129,58,0.02));
                    border:1.5px solid rgba(212,129,58,0.25);border-radius:14px;
                    padding:1.2rem 1.4rem;margin-bottom:1.4rem">
          <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700;
                      color:var(--amber);margin-bottom:4px">⚡ Quick Field Session</div>
          <div style="font-size:0.87rem;color:var(--ink-lt);line-height:1.6">
              Just type your job field and experience level — no documents needed.
              Alex will generate 10 industry-standard interview questions for your role
              and evaluate every answer you give.
          </div>
        </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 2], gap="large")

        with col_a:
            # Popular field suggestions
            FIELD_SUGGESTIONS = {
                "💻 Technology": ["Software Engineer", "Data Analyst", "Data Scientist", "Product Manager",
                                  "DevOps Engineer", "UX Designer", "Cybersecurity Analyst",
                                  "ML Engineer", "Cloud Architect"],
                "📊 Business": ["Business Analyst", "Project Manager", "Strategy Consultant",
                                "Operations Manager", "Supply Chain Manager"],
                "💰 Finance": ["Financial Analyst", "Investment Banker", "Accountant",
                               "Risk Manager", "Finance Manager"],
                "🏥 Healthcare": ["Nurse", "Healthcare Administrator", "Clinical Data Analyst",
                                  "Pharmacist", "Public Health Officer"],
                "📣 Marketing": ["Marketing Manager", "Digital Marketer", "Content Strategist",
                                 "Brand Manager", "Growth Hacker"],
                "⚖️ Legal & HR": ["HR Manager", "Recruiter", "Legal Counsel",
                                   "Compliance Officer", "Labour Relations"],
                "🎓 Education": ["Teacher", "Curriculum Developer", "School Administrator",
                                  "Corporate Trainer", "EdTech Specialist"],
            }

            st.markdown("**🏷️ Your Job Field / Role Title**")
            field_input = st.text_input(
                "Field",
                value=st.session_state.get("quick_field", ""),
                placeholder="e.g. Software Engineer, Data Analyst, Product Manager…",
                label_visibility="collapsed",
                key="field_input_box"
            )
            st.session_state.quick_field = field_input

            st.markdown("<div style='font-size:0.8rem;color:var(--ink-lt);margin:8px 0 4px'>Or pick from common roles:</div>", unsafe_allow_html=True)
            for category, roles in FIELD_SUGGESTIONS.items():
                with st.expander(category):
                    role_cols = st.columns(2)
                    for ri, role in enumerate(roles):
                        with role_cols[ri % 2]:
                            if st.button(role, key=f"role_{role}", use_container_width=True):
                                st.session_state.quick_field = role
                                st.rerun()

        with col_b:
            st.markdown("**📈 Experience Level**")
            exp_level = st.select_slider(
                "Experience",
                options=["Student / Intern", "Entry Level (0–2 yrs)",
                         "Mid Level (3–5 yrs)", "Senior (6–10 yrs)",
                         "Lead / Principal (10+ yrs)", "Executive / Director"],
                value=st.session_state.get("quick_exp", "Mid Level (3–5 yrs)"),
                label_visibility="collapsed",
            )
            st.session_state.quick_exp = exp_level

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🎯 Focus Areas**")
            st.caption("Which question types do you want to practise?")
            focus_options = ["Behavioral", "Technical", "Situational", "Leadership", "Culture Fit"]
            selected_focus = []
            focus_cols = st.columns(2)
            for fi, fo in enumerate(focus_options):
                with focus_cols[fi % 2]:
                    default_checked = fo in ["Behavioral", "Technical", "Situational"]
                    if st.checkbox(fo, value=default_checked, key=f"focus_{fo}"):
                        selected_focus.append(fo)

            st.markdown("<br>", unsafe_allow_html=True)

            # Preview card
            field_val = st.session_state.get("quick_field", "").strip()
            if field_val:
                st.markdown(f"""
                <div style="background:var(--ink);border-radius:12px;padding:14px 16px;color:white">
                  <div style="font-size:0.72rem;color:rgba(255,255,255,0.5);
                              text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
                      Session Preview
                  </div>
                  <div style="font-family:'Playfair Display',serif;font-size:1rem;
                              font-weight:700;margin-bottom:4px">{field_val}</div>
                  <div style="font-size:0.82rem;color:rgba(255,255,255,0.65);margin-bottom:6px">
                      {exp_level}
                  </div>
                  <div style="font-size:0.8rem;color:rgba(255,255,255,0.5)">
                      Focus: {", ".join(selected_focus) if selected_focus else "All areas"}
                  </div>
                  <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:4px">
                      10 tailored questions · Full STAR grading
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        field_val = st.session_state.get("quick_field", "").strip()
        if not field_val:
            st.info("👆 Type your job field above or pick a role to get started.")
        else:
            if st.button(f"⚡ Start Quick Session — {field_val}", type="primary",
                         use_container_width=True):
                with st.spinner(f"Alex is preparing your {field_val} interview session…"):
                    try:
                        focus = selected_focus if selected_focus else ["Behavioral","Technical","Situational"]
                        plan  = build_field_plan(_api(), _prov(), _model(),
                                                 field_val, exp_level, focus)
                        # Clear any resume/JD so session knows we're in field mode
                        st.session_state.resume_text = ""
                        st.session_state.jd_text     = ""
                        _launch_session(plan)
                    except Exception as e:
                        st.error(f"Couldn't build session: {e}")

def page_session():
    plan = st.session_state.session_plan
    if not plan:
        st.warning("No session active. Go to Setup first.")
        if st.button("Go to Setup"): _go("setup")
        return

    questions   = plan.get("question_pool", [])
    total_q     = len(questions)
    idx         = st.session_state.current_q_idx
    session_data = st.session_state.session_data
    completed    = len(session_data)
    name         = plan.get("candidate_name", "there")
    role         = plan.get("target_role", "this role")

    # Header
    col_h1, col_h2 = st.columns([3,1])
    with col_h1:
        st.markdown(f"""
        <div style="margin-bottom:6px">
          <span style="font-family:'Playfair Display',serif;font-size:1.4rem;
                       font-weight:700;color:var(--ink)">
              Coaching Session — {name}
          </span>
          <span style="font-size:0.85rem;color:var(--ink-lt);margin-left:10px">
              Practising for: <em>{role}</em>
          </span>
        </div>""", unsafe_allow_html=True)
    with col_h2:
        if completed > 0:
            avg = sum(d.get("grade",{}).get("score",0) for d in session_data) / completed
            gc  = "#2d7a4f" if avg>=75 else "#d4813a" if avg>=55 else "#c0392b"
            st.markdown(f"""
            <div style="text-align:right">
              <span style="font-size:1.6rem;font-weight:800;color:{gc}">{avg:.0f}</span>
              <span style="font-size:0.8rem;color:var(--ink-lt)"> /100 avg</span>
            </div>""", unsafe_allow_html=True)

    # Progress
    progress_strip(total_q, idx)
    st.markdown(f"<div style='font-size:0.8rem;color:var(--ink-lt);margin-bottom:1rem'>"
                f"Question {min(idx+1, total_q)} of {total_q}</div>", unsafe_allow_html=True)

    # ── Opening message (first question only) ─────────────────────
    if idx == 0 and not session_data:
        coach_bubble(plan.get("opening_message","Let's get started!"))

    # ── Session complete? ─────────────────────────────────────────
    if idx >= total_q:
        st.markdown('<div class="card-teal" style="text-align:center">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;padding:10px 0">
          <div style="font-size:2.5rem">🎉</div>
          <div style="font-family:'Playfair Display',serif;font-size:1.6rem;
                      font-weight:700;color:white;margin:8px 0">Session Complete!</div>
          <div style="color:rgba(255,255,255,0.85);font-size:0.95rem">
              {completed} questions answered — let's see how you did.
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        cb1, cb2 = st.columns(2)
        with cb1:
            if st.button("📊 Generate My Full Report", type="primary", use_container_width=True):
                with st.spinner("Alex is writing your personalised report…"):
                    try:
                        rpt = build_session_report(
                            _api(), _prov(), _model(),
                            st.session_state.session_data,
                            st.session_state.resume_text,
                            st.session_state.jd_text)
                        st.session_state.session_report = rpt
                        _go("report")
                    except Exception as e:
                        st.error(f"Report failed: {e}")
        with cb2:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.session_plan  = None
                st.session_state.session_data  = []
                st.session_state.current_q_idx = 0
                _go("setup")
        return

    # ── Current question ──────────────────────────────────────────
    q_obj    = questions[idx]
    q_text   = q_obj.get("question","")
    q_cat    = q_obj.get("category","General")
    q_wgl    = q_obj.get("what_great_looks_like","")
    q_diff   = q_obj.get("difficulty","Medium")
    diff_col = {"Easy":"#2d7a4f","Medium":"#d4813a","Hard":"#c0392b"}.get(q_diff,"#6354a5")

    # Previous answer's follow-up mode
    if st.session_state.followup_mode:
        st.markdown("#### 💬 Follow-up Coaching")
        coach_bubble(st.session_state.followup_messages[-1]["content"]
                     if st.session_state.followup_messages else "")

        for msg in st.session_state.followup_messages[:-1]:
            if msg["role"] == "assistant":
                coach_bubble(msg["content"])
            else:
                user_bubble(msg["content"])

        fu_input = st.text_input("Your response or question",
                                 placeholder="Ask a follow-up or respond to the coach…",
                                 key=f"fu_input_{idx}", label_visibility="collapsed")
        c1, c2 = st.columns([4,1])
        with c1:
            if st.button("💬 Continue", type="primary", use_container_width=True) and fu_input.strip():
                st.session_state.followup_messages.append({"role":"user","content":fu_input})
                with st.spinner("Alex is thinking…"):
                    try:
                        reply = coach_followup(
                            _api(), _prov(), _model(),
                            st.session_state.followup_messages,
                            st.session_state.resume_text, st.session_state.jd_text)
                        st.session_state.followup_messages.append({"role":"assistant","content":reply})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Chat error: {e}")
        with c2:
            if st.button("Next Q ➡️", use_container_width=True):
                st.session_state.followup_mode     = False
                st.session_state.followup_messages = []
                st.session_state.current_q_idx     += 1
                st.session_state.last_grade         = None
                st.session_state.show_tip           = False
                st.rerun()
        return

    # ── Show last grade (if just graded) ─────────────────────────
    if st.session_state.last_grade:
        g = st.session_state.last_grade
        score = g.get("score", 0)
        grade = g.get("grade","C")
        gc    = GRADE_COLORS.get(grade,"#d4813a")

        # Grade header
        g_col1, g_col2 = st.columns([1,3], gap="large")
        with g_col1:
            st.markdown(f"""
            <div class="score-ring score-{grade}">
              <div style="font-size:2rem">{score}</div>
              <div style="font-size:1.2rem">Grade {grade}</div>
            </div>""", unsafe_allow_html=True)
        with g_col2:
            st.markdown(f"""
            <div class="bubble-coach" style="margin-bottom:10px">
              <div class="bubble-label label-coach">🎙️ Coach Alex</div>
              {g.get('coach_reaction','')}
            </div>""", unsafe_allow_html=True)

        # STAR breakdown
        st.markdown("<div style='margin:12px 0 6px;font-weight:600;color:var(--ink-mid);"
                    "font-size:0.85rem'>STAR BREAKDOWN</div>", unsafe_allow_html=True)
        sb = g.get("star_scores", {})
        score_bar("Situation", sb.get("situation", 0))
        score_bar("Task",      sb.get("task", 0))
        score_bar("Action",    sb.get("action", 0))
        score_bar("Result",    sb.get("result", 0))

        # What worked / missed
        st.markdown("<br>", unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("**✅ What worked**")
            for w in g.get("what_worked", []):
                st.markdown(f'<div class="fb-green">✓ {w}</div>', unsafe_allow_html=True)
        with wc2:
            st.markdown("**⚡ What to improve**")
            for m in g.get("what_missed", []):
                st.markdown(f'<div class="fb-amber">→ {m}</div>', unsafe_allow_html=True)

        # Model answer — human voice + breakdown
        with st.expander("📖 See a full model answer — how you'd actually say it"):
            model_ans = g.get("model_answer", "")
            breakdown = g.get("model_answer_breakdown", "")
            if model_ans:
                st.markdown("""
                <div style="font-size:0.72rem;font-weight:700;color:var(--teal);
                            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
                    💬 How a top candidate would say it
                </div>""", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="fb-model" style="font-size:1rem;line-height:1.75;'
                    f'border-left:3px solid var(--teal);padding-left:14px;'
                    f'font-style:italic;color:var(--ink)">{model_ans}</div>',
                    unsafe_allow_html=True
                )
            if breakdown:
                st.markdown("""
                <div style="margin-top:16px;font-size:0.72rem;font-weight:700;
                            color:var(--amber);text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:8px">
                    🧠 Why this answer scores full marks
                </div>""", unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:0.88rem;line-height:1.65;color:var(--ink-mid);'
                    f'background:rgba(212,129,58,0.06);border-radius:8px;'
                    f'padding:12px 14px">{breakdown}</div>',
                    unsafe_allow_html=True
                )

        # Follow-up question preview
        fq = g.get("follow_up_question","")
        if fq:
            st.markdown(f"""
            <div class="card-cream" style="margin-top:10px">
              <div style="font-size:0.75rem;font-weight:600;color:var(--teal);
                          text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px">
                  🔍 An interviewer might follow up with…
              </div>
              <div style="font-style:italic;color:var(--ink-mid);font-size:0.92rem">"{fq}"</div>
            </div>""", unsafe_allow_html=True)

        # Encouragement
        enc = g.get("encouragement","")
        if enc:
            st.markdown(f"""
            <div style="background:rgba(13,115,119,0.06);border-radius:10px;padding:10px 14px;
                        margin-top:8px;font-size:0.88rem;color:var(--teal)">
                💪 {enc}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("💬 Discuss with Alex", use_container_width=True):
                coach_reply = g.get("coach_reaction","")
                st.session_state.followup_messages = [
                    {"role":"assistant","content":coach_reply},
                ]
                st.session_state.followup_mode = True
                st.rerun()
        with b2:
            if st.button("➡️ Next Question", type="primary", use_container_width=True):
                st.session_state.last_grade       = None
                st.session_state.show_tip         = False
                st.session_state.current_q_idx   += 1
                st.rerun()
        with b3:
            if idx + 1 >= total_q:
                if st.button("📊 Finish & Get Report", use_container_width=True, type="primary"):
                    st.session_state.last_grade     = None
                    st.session_state.current_q_idx = total_q
                    st.rerun()
        return

    # ── Ask the question ──────────────────────────────────────────
    st.markdown(f"""
    <div class="q-card">
      <div class="q-number">Q{idx+1}</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
        <span class="q-category">{q_cat}</span>
        <span style="font-size:0.75rem;color:{diff_col};font-weight:600">{q_diff}</span>
      </div>
      <div class="q-text">"{q_text}"</div>
    </div>""", unsafe_allow_html=True)

    # Tip section
    if st.session_state.show_tip and st.session_state.current_tip:
        st.markdown(f"""
        <div style="background:rgba(212,129,58,0.08);border:1px solid rgba(212,129,58,0.3);
                    border-radius:12px;padding:14px 16px;margin-bottom:12px">
          <div style="font-size:0.75rem;font-weight:600;color:var(--amber);
                      text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">
              💡 Alex's tip
          </div>
          <div style="font-size:0.9rem;color:var(--ink-mid);line-height:1.6">
              {st.session_state.current_tip}
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        tip_col, _ = st.columns([1,3])
        with tip_col:
            if st.button("💡 Get a Hint", use_container_width=True):
                with st.spinner("Alex is thinking…"):
                    try:
                        tip = get_question_tip(
                            _api(), _prov(), _model(),
                            q_text, q_cat,
                            st.session_state.resume_text,
                            st.session_state.jd_text)
                        st.session_state.current_tip = tip
                        st.session_state.show_tip = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Tip failed: {e}")

    # What a great answer looks like
    if q_wgl:
        st.markdown(f"""
        <div style="font-size:0.8rem;color:var(--ink-lt);margin-bottom:10px;
                    padding:6px 10px;border-left:2px solid rgba(13,115,119,0.3)">
            <em>What makes a great answer: {q_wgl}</em>
        </div>""", unsafe_allow_html=True)

    # ── Answer input — type OR speak ────────────────────────────────
    st.markdown("""
    <div style="font-size:0.85rem;font-weight:700;color:var(--ink);
                margin-bottom:6px;margin-top:4px">
        ✍️ Your Answer
        <span style="font-weight:400;color:var(--ink-lt);font-size:0.78rem;">
         — type below, or record your voice using the mic
        </span>
    </div>""", unsafe_allow_html=True)

    # ── Tab switcher: Type / Voice ────────────────────────────────
    ans_tab, voice_tab = st.tabs(["⌨️  Type Answer", "🎙️  Record Voice"])

    with ans_tab:
        typed_val = st.session_state.get("voice_transcript", "")
        ans = st.text_area(
            "Answer",
            value=typed_val,
            height=180,
            label_visibility="collapsed",
            placeholder="Type your answer here. Aim for 2-4 minutes of speaking. "
                        "Be specific — real examples beat generic statements every time.",
            key=f"ans_{idx}",
        )
        if typed_val and ans:
            st.session_state.voice_transcript = ""

    with voice_tab:
        provider_ok = _prov() in ("groq", "openai", "ollama")
        if not provider_ok:
            st.warning(
                f"Voice recording needs Groq (free), OpenAI, or Ollama. "
                f"Your current provider ({_prov()}) doesn't support audio transcription. "
                "Switch to Groq for instant free transcription."
            )
        else:
            provider_label = {"groq": "Groq Whisper (free)", "openai": "OpenAI Whisper",
                              "ollama": "Local Whisper"}.get(_prov(), _prov())

            st.markdown(f"""
            <div style="font-size:0.82rem;color:var(--ink-lt);margin-bottom:12px;
                        padding:8px 12px;border-left:3px solid rgba(13,115,119,0.35);
                        background:rgba(13,115,119,0.04);border-radius:0 8px 8px 0">
                🎙️ Click the mic below to record your answer. When done, click
                <b>🎙️ Transcribe</b> — your speech will be converted to text
                by <b>{provider_label}</b> and moved to the Type tab for review.
            </div>""", unsafe_allow_html=True)

            # ── st.audio_input — native Streamlit widget (1.41+) ──────────
            # Returns an UploadedFile (WAV bytes) or None. No JS bridge needed.
            audio_input = st.audio_input(
                "Record your answer",
                key=f"audio_input_{idx}",
                label_visibility="collapsed",
            )

            if audio_input is not None:
                # Show a success indicator
                st.markdown("""
                <div style="padding:8px 12px;border-radius:8px;margin-top:6px;
                            background:rgba(45,122,79,0.1);font-size:0.82rem;
                            color:#2d7a4f;font-weight:600">
                    ✅ Recording captured — click Transcribe below to convert to text
                </div>""", unsafe_allow_html=True)

                t1, t2 = st.columns([2, 1])
                with t1:
                    do_transcribe = st.button(
                        "🎙️ Transcribe & Use This Answer",
                        type="primary",
                        use_container_width=True,
                        key=f"transcribe_{idx}",
                        help="Converts your recording to text and fills the answer box",
                    )
                with t2:
                    st.markdown(
                        "<div style='font-size:0.72rem;color:var(--ink-lt);padding-top:10px'>"
                        "⚡ Powered by Whisper AI</div>",
                        unsafe_allow_html=True,
                    )

                if do_transcribe:
                    try:
                        audio_bytes = audio_input.read()
                        with st.spinner("🎙️ Transcribing your answer with Whisper…"):
                            transcript = transcribe_audio(audio_bytes, _api(), _prov())
                        if transcript and transcript.strip():
                            st.session_state.voice_transcript = transcript.strip()
                            st.success("✅ Done! Switch to the Type tab — your answer is ready to submit.")
                            st.rerun()
                        else:
                            st.error("Transcription returned empty — please try recording again.")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
            else:
                st.markdown("""
                <div style="padding:10px 14px;border-radius:8px;margin-top:4px;
                            background:rgba(0,0,0,0.03);font-size:0.8rem;color:var(--ink-lt)">
                    👆 Click the mic icon above to start recording. Click again to stop.
                    Your recording will appear here for review before transcribing.
                </div>""", unsafe_allow_html=True)

    # ── Final answer text ─────────────────────────────────────────
    final_ans = ans.strip() if ans.strip() else ""



    a1, a2, a3 = st.columns([3,1,1])
    with a1:
        submit = st.button("✅ Submit Answer", type="primary", use_container_width=True,
                           disabled=not final_ans)
    with a2:
        if st.button("⏭️ Skip", use_container_width=True):
            st.session_state.current_q_idx += 1
            st.session_state.last_grade     = None
            st.session_state.show_tip       = False
            st.rerun()
    with a3:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.session_plan  = None
            st.session_state.session_data  = []
            st.session_state.current_q_idx = 0
            _go("setup")

    if submit and final_ans:
        with st.spinner("Alex is reviewing your answer…"):
            try:
                grade = grade_answer(
                    _api(), _prov(), _model(),
                    q_text, final_ans, q_cat,
                    st.session_state.resume_text, st.session_state.jd_text)
                entry = {"question": q_text, "category": q_cat,
                         "answer": final_ans, "grade": grade}
                st.session_state.session_data.append(entry)
                st.session_state.last_grade = grade
                st.session_state.show_tip   = False
                st.session_state.voice_transcript = ""
                st.rerun()
            except Exception as e:
                st.error(f"Grading failed: {e}")


# ══════════════════════════════════════════════════════════════════
# PAGE: REPORT
# ══════════════════════════════════════════════════════════════════
def page_report():
    st.markdown('<div class="section-head">📊 Your Coaching Report</div>', unsafe_allow_html=True)

    rpt = st.session_state.session_report
    data = st.session_state.session_data

    if not data:
        st.info("No session data yet. Complete a practice session first.")
        if st.button("Go to Setup"): _go("setup")
        return

    # Quick summary if no full report yet
    if not rpt:
        st.markdown("**Session Summary** (full AI report not yet generated)")
        avg = sum(d.get("grade",{}).get("score",0) for d in data) / len(data)
        st.metric("Session Average", f"{avg:.0f}/100")
        if st.button("📊 Generate Full Report", type="primary"):
            with st.spinner("Alex is writing your report…"):
                try:
                    rpt = build_session_report(
                        _api(), _prov(), _model(),
                        data, st.session_state.resume_text, st.session_state.jd_text)
                    st.session_state.session_report = rpt
                    st.rerun()
                except Exception as e:
                    st.error(f"Report failed: {e}")
        return

    # ── Full report ───────────────────────────────────────────────
    overall  = rpt.get("overall_score", 0)
    grade    = rpt.get("overall_grade","C")
    tier     = rpt.get("tier","")
    headline = rpt.get("headline","")
    gc       = GRADE_COLORS.get(grade,"#d4813a")

    # Hero
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{gc}18,{gc}06);border:2px solid {gc}40;
                border-radius:20px;padding:28px;text-align:center;margin-bottom:24px">
      <div style="font-size:4.5rem;font-weight:900;color:{gc};
                  font-family:'Playfair Display',serif;line-height:1">{overall}</div>
      <div style="font-size:2rem;font-weight:800;color:{gc}">Grade {grade}</div>
      <div style="color:{gc};font-size:1rem;font-weight:600;margin-top:4px">{tier}</div>
      <div style="color:var(--ink-mid);font-size:0.95rem;margin-top:10px;
                  font-style:italic">"{headline}"</div>
    </div>""", unsafe_allow_html=True)

    # Category scores
    cat_scores = rpt.get("category_scores",{})
    if cat_scores:
        st.markdown("#### Category Breakdown")
        c1, c2 = st.columns(2)
        items = list(cat_scores.items())
        for i, (cat, sc) in enumerate(items):
            with (c1 if i % 2 == 0 else c2):
                rpt_bar(cat, sc)
        st.markdown("<br>", unsafe_allow_html=True)

    # Strengths & improvements
    sr_col, im_col = st.columns(2, gap="large")
    with sr_col:
        st.markdown("#### 🌟 Top Strengths")
        for s in rpt.get("top_strengths",[]):
            st.markdown(f'<div class="fb-green">{s}</div>', unsafe_allow_html=True)
    with im_col:
        st.markdown("#### ⚡ Priority Improvements")
        for p in rpt.get("priority_improvements",[]):
            with st.expander(f"📌 {p.get('area','')}"):
                st.markdown(f"**Issue:** {p.get('issue','')}")
                st.markdown(f"**Fix:** {p.get('fix','')}")

    # Action plan
    ap = rpt.get("action_plan",[])
    if ap:
        st.markdown("#### 🗓️ Your Action Plan")
        for i, item in enumerate(ap, 1):
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:8px">
              <div style="background:var(--teal);color:white;border-radius:50%;
                          width:24px;height:24px;min-width:24px;display:flex;
                          align-items:center;justify-content:center;font-size:0.75rem;
                          font-weight:700">{i}</div>
              <div style="color:var(--ink-mid);font-size:0.9rem;line-height:1.5">{item}</div>
            </div>""", unsafe_allow_html=True)

    # Personal note from coach
    note = rpt.get("personal_note","")
    if note:
        st.markdown("<br>", unsafe_allow_html=True)
        coach_bubble(note, label="Coach Alex — Personal Note")

    # Q-by-Q breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Question-by-Question Review")
    for i, item in enumerate(data, 1):
        g   = item.get("grade",{})
        sc  = g.get("score",0)
        gr  = g.get("grade","C")
        gcc = GRADE_COLORS.get(gr,"#d4813a")
        with st.expander(f"Q{i} [{item.get('category','')}] — {sc}/100 (Grade {gr}) "
                         f"· {item.get('question','')[:55]}…"):
            st.markdown(f"**Question:** {item.get('question','')}")
            st.markdown(f"**Your Answer:** {item.get('answer','')[:400]}…")
            st.markdown(f"**Coach's Take:** {g.get('coach_reaction','')}")
            if g.get("model_answer"):
                st.markdown(f'<div class="fb-model">'
                            f'<b>Model Answer:</b> {g["model_answer"]}</div>',
                            unsafe_allow_html=True)

    # Download
    st.markdown("<br>", unsafe_allow_html=True)
    lines = [
        "INTERVIEW COACHING REPORT — Coach Alex AI",
        "="*50,
        f"Overall Score: {overall}/100 · Grade {grade} · {tier}",
        f'"{headline}"', "",
    ]
    for i, item in enumerate(data,1):
        g = item.get("grade",{})
        lines += [f"\nQ{i}: {item.get('question','')}",
                  f"Score: {g.get('score',0)}/100 ({g.get('grade','')})",
                  f"Your Answer: {item.get('answer','')}",
                  f"Coach: {g.get('coach_reaction','')}",
                  f"Model Answer: {g.get('model_answer','')}","─"*50]
    if ap:
        lines += ["\nACTION PLAN:"] + [f"{i}. {a}" for i,a in enumerate(ap,1)]
    if note:
        lines += ["\nCOACH'S NOTE:", note]

    st.download_button("📄 Download Full Report (TXT)",
                       "\n".join(lines).encode(), "coaching_report.txt", "text/plain",
                       use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: FREE CHAT
# ══════════════════════════════════════════════════════════════════
def page_chat():
    st.markdown('<div class="section-head">💬 Chat with Coach Alex</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask anything — interview nerves, salary negotiation, '
                'career pivots, how to position yourself. Alex is here for you.</div>',
                unsafe_allow_html=True)

    if not _ready():
        st.warning("⚠️ Connect your AI provider in the sidebar first.")
        return

    msgs = st.session_state.chat_messages

    if not msgs:
        has_ctx = _has_context()
        ctx_note = ("I can see your resume and the job you're targeting — so my advice will be "
                    "tailored to you.") if has_ctx else \
                   ("You haven't added a resume or job description yet — I'll give general advice. "
                    "Head to Setup to add those for personalised coaching.")
        coach_bubble(
            f"Hey! I'm Alex, your interview coach. {ctx_note}<br><br>"
            "What's on your mind? You can ask me anything — "
            "how to answer a specific question, how to deal with nerves, "
            "what they're really looking for, salary negotiation… anything."
        )
    else:
        for msg in msgs:
            if msg["role"] == "user":
                user_bubble(msg["content"])
            else:
                coach_bubble(msg["content"])

    # Quick starters
    starters = [
        "How do I answer 'Tell me about yourself'?",
        "How should I handle a question about my biggest weakness?",
        "I'm nervous. How do I calm my interview anxiety?",
        "How do I negotiate salary without seeming greedy?",
        "What questions should I ask the interviewer?",
        "How do I explain a gap in my employment?",
    ]
    with st.expander("💡 Quick starters — tap to send"):
        cols = st.columns(2)
        for i, s in enumerate(starters):
            with cols[i % 2]:
                if st.button(s, key=f"starter_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role":"user","content":s})
                    with st.spinner("Alex is thinking…"):
                        try:
                            reply = free_chat(_api(), _prov(), _model(),
                                             st.session_state.chat_messages,
                                             st.session_state.resume_text,
                                             st.session_state.jd_text)
                            st.session_state.chat_messages.append({"role":"assistant","content":reply})
                        except Exception as e:
                            st.error(f"Chat error: {e}")
                    st.rerun()

    st.markdown("---")
    user_in = st.text_input("Your message", placeholder="Ask Alex anything…",
                             label_visibility="collapsed", key="free_chat_input")
    sc1, sc2 = st.columns([4,1])
    with sc1:
        send = st.button("💬 Send", type="primary", use_container_width=True)
    with sc2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    if send and user_in.strip():
        st.session_state.chat_messages.append({"role":"user","content":user_in})
        with st.spinner("Alex is thinking…"):
            try:
                reply = free_chat(_api(), _prov(), _model(),
                                  st.session_state.chat_messages,
                                  st.session_state.resume_text,
                                  st.session_state.jd_text)
                st.session_state.chat_messages.append({"role":"assistant","content":reply})
            except Exception as e:
                st.error(f"Chat error: {e}")
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
page = st.session_state.page
if   page == "home":    page_home()
elif page == "setup":   page_setup()
elif page == "session": page_session()
elif page == "report":  page_report()
elif page == "chat":    page_chat()
else:                   page_home()