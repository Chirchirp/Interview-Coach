"""
Interview Coach AI - LLM Core  (v3 — Stable + Ollama-Fixed)
Supports: Groq (FREE), OpenAI, Anthropic, OpenRouter, Ollama (LOCAL FREE)

Ollama fixes:
  - Uses OpenAI-compat /v1/chat/completions for ALL calls (no /api/chat 404)
  - Detects installed Ollama models dynamically at connection time
  - keep_alive warmup via /api/generate keeps model hot between calls
  - Tighter token budgets + context trimming for faster Ollama responses
  - Brevity hints reduce output token count on Ollama
"""
from __future__ import annotations
import json, re, time


# ── Token budgets ─────────────────────────────────────────────────
_RESUME_LIMIT  = 3000
_JD_LIMIT      = 2000
_CHAT_TOKENS   = 700
_GRADE_TOKENS  = 600
_PLAN_TOKENS   = 1000
_Q_TOKENS      = 300
_TIPS_TOKENS   = 800
_REPORT_TOKENS = 1200

# Ollama-specific tighter budgets (fewer tokens = faster)
_OLLAMA_TOKENS = {
    "plan":    800,
    "grade_a": 380,
    "grade_b": 250,
    "tip":     220,
    "chat":    380,
    "report":  800,
}

# Ollama context trim limits (chars) — smaller prompt = faster prefill
_OLLAMA_CTX = {
    "plan":   1200,   # full plan needs most context
    "grade":   250,   # grading only needs highlights
    "tip":     180,
    "chat":    350,
    "report":  350,
}


def _trim(text: str, limit: int) -> str:
    """Character trim — original behaviour, used for cloud providers."""
    return text[:limit] + "\n[truncated]" if len(text) > limit else text


def _ollama_trim(text: str, task: str) -> str:
    """Aggressive head+tail trim for Ollama to reduce prefill time."""
    if not text or not text.strip():
        return ""
    limit = _OLLAMA_CTX.get(task, 400)
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[...]\n" + text[-half:]


def _brevity(provider: str, words: int = 50) -> str:
    """Inject word-limit hint for Ollama only — cuts output token count."""
    return f"\nIMPORTANT: Keep every field under {words} words. Be concise." \
           if provider == "ollama" else ""


# ── Robust JSON extractor ─────────────────────────────────────────
def _ej(text: str):
    """Extract and parse JSON from LLM output."""
    if not text or not text.strip():
        raise ValueError("Empty response from model.")

    # Strip smart quotes (common Ollama quirk)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    s_obj, s_arr = text.find("{"), text.find("[")
    if s_obj == -1 and s_arr == -1:
        raise ValueError(f"No JSON in response. Got: {text[:300]}")
    if s_arr == -1 or (s_obj != -1 and s_obj < s_arr):
        start, oc, cc = s_obj, "{", "}"
    else:
        start, oc, cc = s_arr, "[", "]"

    depth = 0; in_str = False; esc = False
    for i, ch in enumerate(text[start:], start=start):
        if esc:                   esc = False;  continue
        if ch == "\\" and in_str: esc = True;   continue
        if ch == '"':             in_str = not in_str; continue
        if in_str:                continue
        if ch == oc:              depth += 1
        elif ch == cc:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    # Attempt repair
                    return json.loads(_repair_json(text[start:i+1]))

    # Truncated JSON — attempt repair
    candidate = text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_repair_json(candidate))


def _repair_json(s: str) -> str:
    """Close unclosed strings, brackets and braces."""
    depth_brace = 0; depth_bracket = 0
    in_str = False; esc = False
    for ch in s:
        if esc:           esc = False; continue
        if ch == "\\":    esc = True;  continue
        if ch == '"':     in_str = not in_str; continue
        if in_str:        continue
        if ch == "{":     depth_brace   += 1
        elif ch == "}":   depth_brace   -= 1
        elif ch == "[":   depth_bracket += 1
        elif ch == "]":   depth_bracket -= 1
    if in_str:            s += '"'
    s = re.sub(r",\s*$", "", s.rstrip())
    s += "]" * max(0, depth_bracket)
    s += "}" * max(0, depth_brace)
    return s


def _call_with_retry(api_key, provider, model, prompt, system_prompt,
                     temperature, max_tokens, retries=3) -> str:
    """Call LLM with up to 3 retries, adding a stricter hint on each retry."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            p = prompt if attempt == 1 else (
                prompt + f"\n\n[Attempt {attempt}: output ONLY valid JSON, "
                "starting with {{ or [. No other text whatsoever.]"
            )
            return call_llm(api_key, provider, model, p,
                            system_prompt=system_prompt,
                            temperature=max(0.1, temperature - 0.05 * (attempt-1)),
                            max_tokens=max_tokens)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.3)
    raise last_err


# ── Ollama URL helpers ────────────────────────────────────────────
def _ollama_root(raw: str) -> str:
    """
    Normalise user input to a clean root URL, e.g. http://localhost:11434
    Strips any /v1, /api, /api/v1 suffixes the user may have typed.
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        url = "http://localhost:11434"
    if not url.startswith("http"):
        url = "http://" + url
    for suffix in ["/v1/chat/completions", "/v1", "/api/v1", "/api"]:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def _ollama_base(raw: str) -> str:
    """root/v1 — for the OpenAI-compat client."""
    return _ollama_root(raw) + "/v1"


# ── keep_alive warmup ─────────────────────────────────────────────
def _ollama_warmup(api_key: str, model: str) -> None:
    """
    POST keep_alive=10m to /api/generate so Ollama keeps the model
    loaded in RAM between calls. Eliminates 5-20s reload lag.
    Fire-and-forget — failures are silently ignored.
    """
    try:
        import urllib.request as _ur
        root    = _ollama_root(api_key)
        payload = json.dumps({"model": model, "prompt": "", "keep_alive": "10m"}).encode()
        req = _ur.Request(f"{root}/api/generate", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        _ur.urlopen(req, timeout=4)
    except Exception:
        pass


# ── Dynamic Ollama model detection ───────────────────────────────
def get_ollama_models(api_key: str) -> dict:
    """
    Query the running Ollama instance and return models it actually has.
    Falls back to safe defaults if Ollama is unreachable.
    """
    try:
        import urllib.request as _ur, json as _json
        root = _ollama_root(api_key)
        req  = _ur.urlopen(f"{root}/api/tags", timeout=4)
        data = _json.loads(req.read())
        installed = [m["name"] for m in data.get("models", [])]
        if not installed:
            return {"(no models found — run: ollama pull phi3.5)": "phi3.5"}
        # Build display name → model id dict
        return {name: name for name in installed}
    except Exception:
        # Ollama not running — return placeholder
        return {"Connect Ollama first (ollama serve)": "llama3.2"}


# ── Provider model catalogue ──────────────────────────────────────
# Note: Ollama models are populated dynamically at connect-time via
# get_ollama_models(). The static list below is only shown before connecting.
PROVIDER_MODELS = {
    "groq": {
        "Llama 3.3 70B — Recommended (FREE)": "llama-3.3-70b-versatile",
        "Llama 3.1 8B — Fastest (FREE)":      "llama-3.1-8b-instant",
        "Mixtral 8x7B (FREE)":                 "mixtral-8x7b-32768",
    },
    "openai": {
        "GPT-4o Mini — Budget": "gpt-4o-mini",
        "GPT-4o — Best":        "gpt-4o",
        "GPT-3.5 Turbo":        "gpt-3.5-turbo",
    },
    "anthropic": {
        "Claude 3 Haiku — Cheapest": "claude-3-haiku-20240307",
        "Claude 3.5 Sonnet":         "claude-3-5-sonnet-20241022",
    },
    "openrouter": {
        "Llama 3.3 70B (Free tier)": "meta-llama/llama-3.3-70b-instruct",
        "Mistral 7B (Budget)":       "mistralai/mistral-7b-instruct",
        "GPT-4o Mini":               "openai/gpt-4o-mini",
    },
    "ollama": {
        # Static placeholder — replaced dynamically after connecting
        "⚡ Connect first to see your installed models": "llama3.2",
    },
}

HELP_LINKS = {
    "groq":       "https://console.groq.com/keys",
    "openai":     "https://platform.openai.com/api-keys",
    "anthropic":  "https://console.anthropic.com/settings/keys",
    "openrouter": "https://openrouter.ai/keys",
    "ollama":     "https://ollama.ai/download",
}


# ── Client factory ────────────────────────────────────────────────
def get_client(api_key: str, provider: str):
    from openai import OpenAI
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    if provider == "ollama":
        # OpenAI-compat /v1 — stable, no /api/chat 404 risk
        return OpenAI(api_key="ollama", base_url=_ollama_base(api_key))
    urls = {
        "groq":       "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "together":   "https://api.together.xyz/v1",
    }
    return OpenAI(api_key=api_key, base_url=urls.get(provider))


def verify_connection(api_key: str, provider: str) -> tuple[bool, str]:
    try:
        if provider == "ollama":
            import urllib.request, json as _json
            root = _ollama_root(api_key)
            req  = urllib.request.urlopen(f"{root}/api/tags", timeout=6)
            data = _json.loads(req.read())
            models = [m["name"] for m in data.get("models", [])]
            if models:
                hint = f" · {len(models)} model(s): {', '.join(models[:3])}"
                if len(models) > 3:
                    hint += f" +{len(models)-3} more"
            else:
                hint = " · No models yet — run: ollama pull phi3.5"
            return True, f"🖥️ Ollama connected — local & free!{hint}"

        elif provider == "anthropic":
            import anthropic
            c = anthropic.Anthropic(api_key=api_key)
            c.messages.create(model="claude-3-haiku-20240307", max_tokens=5,
                              messages=[{"role": "user", "content": "hi"}])
        else:
            c = get_client(api_key, provider)
            c.models.list()

        return True, "✅ Connected successfully."

    except Exception as e:
        msg = str(e).lower()
        if "authentication" in msg or "unauthorized" in msg or "invalid" in msg:
            return False, "❌ Invalid API key — please check and try again."
        if "connection" in msg or "timeout" in msg or "refused" in msg:
            return False, "❌ Cannot reach Ollama. Run: ollama serve"
        return False, f"❌ {str(e)[:140]}"


# ── Core LLM call (all providers via their official client) ───────
def call_llm(api_key, provider, model, prompt,
             system_prompt="You are a helpful assistant.",
             temperature=0.4, max_tokens=700) -> str:
    """
    Unified LLM call. Ollama uses OpenAI-compat /v1/chat/completions.
    This is the stable path — never calls /api/chat.
    """
    if provider == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=api_key)
        r = c.messages.create(model=model, max_tokens=max_tokens,
                              temperature=temperature, system=system_prompt,
                              messages=[{"role": "user", "content": prompt}])
        return r.content[0].text

    c    = get_client(api_key, provider)
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt}]
    r = c.chat.completions.create(model=model, messages=msgs,
                                  temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════
# 1. SESSION PLAN — resume + JD
# ═══════════════════════════════════════════════════════════════════
def build_session_plan(api_key, provider, model, resume_text, jd_text) -> dict:
    # Warm up Ollama so model stays loaded for the whole session
    if provider == "ollama":
        _ollama_warmup(api_key, model)

    is_ollama = provider == "ollama"
    resume_ctx = _ollama_trim(resume_text, "plan") if is_ollama else _trim(resume_text, _RESUME_LIMIT)
    jd_ctx     = _ollama_trim(jd_text,     "plan") if is_ollama else _trim(jd_text, _JD_LIMIT)
    max_tok    = _OLLAMA_TOKENS["plan"] if is_ollama else _PLAN_TOKENS

    prompt = (
        'You are an expert interview coach. Analyse this resume and job description.\n'
        'Return ONLY valid JSON:\n'
        '{"candidate_name":"<first name from resume or Candidate>",'
        '"target_role":"<role from JD>",'
        '"company_hints":"<company name if visible or empty string>",'
        '"key_strengths":["<strength>","<strength>","<strength>"],'
        '"key_gaps":["<gap>","<gap>","<gap>"],'
        '"opening_message":"<2-3 warm sentences welcoming candidate by name>",'
        '"question_pool":['
        '{"id":1,"category":"Opener","question":"<warm opener>","what_great_looks_like":"<1 sentence>","difficulty":"Easy"},'
        '{"id":2,"category":"Behavioral","question":"<STAR question from experience>","what_great_looks_like":"<1 sentence>","difficulty":"Medium"},'
        '{"id":3,"category":"Behavioral","question":"<challenge/failure question>","what_great_looks_like":"<1 sentence>","difficulty":"Medium"},'
        '{"id":4,"category":"Technical","question":"<role-specific technical question>","what_great_looks_like":"<1 sentence>","difficulty":"Medium"},'
        '{"id":5,"category":"Technical","question":"<deeper technical or tool question>","what_great_looks_like":"<1 sentence>","difficulty":"Hard"},'
        '{"id":6,"category":"Situational","question":"<hypothetical scenario>","what_great_looks_like":"<1 sentence>","difficulty":"Medium"},'
        '{"id":7,"category":"Leadership","question":"<influence or team question>","what_great_looks_like":"<1 sentence>","difficulty":"Medium"},'
        '{"id":8,"category":"Culture Fit","question":"<values or motivation question>","what_great_looks_like":"<1 sentence>","difficulty":"Easy"},'
        '{"id":9,"category":"Gap Challenge","question":"<probes weakest gap>","what_great_looks_like":"<1 sentence>","difficulty":"Hard"},'
        '{"id":10,"category":"Closing","question":"Do you have any questions for me?","what_great_looks_like":"Ask 2 thoughtful questions","difficulty":"Easy"}'
        ']}'
        + _brevity(provider, 40) +
        '\n\nRESUME:\n' + resume_ctx +
        '\n\nJOB DESCRIPTION:\n' + jd_ctx +
        '\n\nJSON only:'
    )
    raw = _call_with_retry(api_key, provider, model, prompt,
                           system_prompt="Return ONLY valid JSON. No markdown.",
                           temperature=0.5, max_tokens=max_tok)
    return _ej(raw)


# ═══════════════════════════════════════════════════════════════════
# 1b. FIELD-ONLY PLAN — no resume/JD
# ═══════════════════════════════════════════════════════════════════
def build_field_plan(api_key, provider, model, field,
                     experience_level, focus_areas) -> dict:
    if provider == "ollama":
        _ollama_warmup(api_key, model)

    focus_str = ", ".join(focus_areas) if focus_areas else "Behavioral, Technical, Situational"
    max_tok   = _OLLAMA_TOKENS["plan"] if provider == "ollama" else _PLAN_TOKENS

    prompt = (
        'You are an expert interview coach. Build a 10-question interview practice plan.\n'
        'Return ONLY valid JSON:\n'
        '{"candidate_name":"Candidate",'
        '"target_role":"' + field + '",'
        '"company_hints":"",'
        '"key_strengths":["Prepare specific examples","Show measurable outcomes","Use STAR structure"],'
        '"key_gaps":["Tailor to ' + field + ' context","Quantify impact","Be specific"],'
        '"opening_message":"<2-3 warm sentences for a ' + experience_level + ' ' + field + ' candidate>",'
        '"question_pool":['
        '{"id":1,"category":"Opener","question":"Tell me about yourself and what draws you to ' + field + '.","what_great_looks_like":"90-second story with key value","difficulty":"Easy"},'
        '{"id":2,"category":"Behavioral","question":"<STAR behavioral for ' + field + ' at ' + experience_level + '>","what_great_looks_like":"Specific example with result","difficulty":"Medium"},'
        '{"id":3,"category":"Behavioral","question":"<challenge question for ' + field + '>","what_great_looks_like":"Shows self-awareness","difficulty":"Medium"},'
        '{"id":4,"category":"Technical","question":"<core technical question for ' + field + '>","what_great_looks_like":"Clear explanation with example","difficulty":"Medium"},'
        '{"id":5,"category":"Technical","question":"<advanced technical for ' + field + '>","what_great_looks_like":"Structured thinking","difficulty":"Hard"},'
        '{"id":6,"category":"Situational","question":"<workplace scenario for ' + field + '>","what_great_looks_like":"Logical approach","difficulty":"Medium"},'
        '{"id":7,"category":"Leadership","question":"<collaboration question for ' + experience_level + ' ' + field + '>","what_great_looks_like":"Shows impact on others","difficulty":"Medium"},'
        '{"id":8,"category":"Culture Fit","question":"What work environment brings out your best?","what_great_looks_like":"Authentic and specific","difficulty":"Easy"},'
        '{"id":9,"category":"Motivation","question":"<career goals for ' + field + '>","what_great_looks_like":"Forward-looking and genuine","difficulty":"Easy"},'
        '{"id":10,"category":"Closing","question":"Do you have any questions for me?","what_great_looks_like":"Ask 2 thoughtful researched questions","difficulty":"Easy"}'
        ']}'
        + _brevity(provider, 40) +
        '\n\nFIELD: ' + field +
        '\nEXPERIENCE: ' + experience_level +
        '\nFOCUS: ' + focus_str +
        '\n\nJSON only:'
    )
    raw = _call_with_retry(api_key, provider, model, prompt,
                           system_prompt="Return ONLY valid JSON. No markdown.",
                           temperature=0.6, max_tokens=max_tok)
    return _ej(raw)


# ═══════════════════════════════════════════════════════════════════
# 2. GRADE ANSWER
# ═══════════════════════════════════════════════════════════════════
def grade_answer(api_key, provider, model, question, user_answer,
                 category, resume_text, jd_text) -> dict:
    is_ollama  = provider == "ollama"
    resume_ctx = _ollama_trim(resume_text, "grade") if is_ollama else _trim(resume_text, 1000)
    jd_ctx     = _ollama_trim(jd_text,     "grade") if is_ollama else _trim(jd_text, 600)
    max_tok    = _OLLAMA_TOKENS["grade_a"] if is_ollama else _GRADE_TOKENS

    prompt = (
        'Grade this interview answer. Return ONLY JSON:\n'
        '{"score":<0-100>,"grade":"A|B|C|D|F",'
        '"star_scores":{"situation":<0-25>,"task":<0-25>,"action":<0-25>,"result":<0-25>},'
        '"what_worked":["<strength>","<strength>"],'
        '"what_missed":["<gap>","<gap>"],'
        '"coach_reaction":"<1-2 warm sentences referencing their actual words>",'
        '"model_answer":"<strong 2-3 sentence ideal answer>",'
        '"follow_up_question":"<one natural follow-up>",'
        '"encouragement":"<1 sentence tip>"}'
        + _brevity(provider, 45) +
        f'\n\nCATEGORY: {category}\n'
        f'QUESTION: {question}\n'
        f'CANDIDATE ANSWER: {user_answer[:800]}\n'
        'RESUME:\n' + resume_ctx +
        '\nJOB:\n' + jd_ctx +
        '\nJSON only:'
    )
    raw = _call_with_retry(api_key, provider, model, prompt,
                           system_prompt="Return ONLY valid JSON. No markdown.",
                           temperature=0.3, max_tokens=max_tok)
    return _ej(raw)


# ═══════════════════════════════════════════════════════════════════
# 3. FOLLOW-UP COACHING CHAT
# ═══════════════════════════════════════════════════════════════════
def coach_followup(api_key, provider, model, conversation_history,
                   resume_text, jd_text) -> str:
    is_ollama = provider == "ollama"
    history_str = ""
    for msg in conversation_history[-6:]:
        role = "Candidate" if msg["role"] == "user" else "Coach"
        history_str += f"{role}: {msg['content']}\n\n"

    resume_ctx = _ollama_trim(resume_text, "chat") if is_ollama else _trim(resume_text, 1000)
    jd_ctx     = _ollama_trim(jd_text,     "chat") if is_ollama else _trim(jd_text, 700)
    max_tok    = _OLLAMA_TOKENS["chat"] if is_ollama else _CHAT_TOKENS

    sys = (
        "You are Alex, an experienced warm interview coach. "
        "Give direct, specific coaching in 2-3 short paragraphs. "
        "Reference the candidate's actual words. Be encouraging but honest."
    )
    prompt = (
        "RESUME:\n" + resume_ctx + "\n\nJOB:\n" + jd_ctx +
        "\n\nCONVERSATION:\n" + history_str + "Coach Alex:"
    )
    return call_llm(api_key, provider, model, prompt,
                    system_prompt=sys, temperature=0.65, max_tokens=max_tok)


# ═══════════════════════════════════════════════════════════════════
# 4. QUICK TIP
# ═══════════════════════════════════════════════════════════════════
def get_question_tip(api_key, provider, model, question, category,
                     resume_text, jd_text) -> str:
    is_ollama  = provider == "ollama"
    resume_ctx = _ollama_trim(resume_text, "tip") if is_ollama else _trim(resume_text, 1200)
    jd_ctx     = _ollama_trim(jd_text,     "tip") if is_ollama else _trim(jd_text, 800)
    max_tok    = _OLLAMA_TOKENS["tip"] if is_ollama else 300

    prompt = (
        f"Question: {question}\nCategory: {category}\n\n"
        "Give a 3-4 sentence coaching tip. Be specific, not generic.\n"
        + _brevity(provider, 55) +
        "\nRESUME:\n" + resume_ctx + "\nJOB:\n" + jd_ctx
    )
    return call_llm(api_key, provider, model, prompt,
                    system_prompt="You are an expert interview coach. Be specific and concise.",
                    temperature=0.5, max_tokens=max_tok)


# ═══════════════════════════════════════════════════════════════════
# 5. FINAL SESSION REPORT
# ═══════════════════════════════════════════════════════════════════
def build_session_report(api_key, provider, model, session_data,
                         resume_text, jd_text) -> dict:
    is_ollama = provider == "ollama"
    qa_summary = ""
    for i, item in enumerate(session_data, 1):
        g = item.get("grade", {})
        qa_summary += (
            f"Q{i} [{item.get('category','')}]: {item.get('question','')[:70]}\n"
            f"Score: {g.get('score',0)}/100 | Missed: {'; '.join(g.get('what_missed',[]))}\n\n"
        )

    resume_ctx = _ollama_trim(resume_text, "report") if is_ollama else _trim(resume_text, 800)
    jd_ctx     = _ollama_trim(jd_text,     "report") if is_ollama else _trim(jd_text, 600)
    max_tok    = _OLLAMA_TOKENS["report"] if is_ollama else _REPORT_TOKENS

    prompt = (
        'Generate a final interview coaching report. Return ONLY JSON:\n'
        '{"overall_score":<0-100>,"overall_grade":"A|B|C|D|F",'
        '"tier":"Interview Ready|Almost There|Needs Practice|Significant Work Needed",'
        '"headline":"<one punchy sentence>",'
        '"top_strengths":["<strength>","<strength>","<strength>"],'
        '"priority_improvements":['
        '{"area":"<area>","issue":"<issue>","fix":"<fix>"},'
        '{"area":"<area>","issue":"<issue>","fix":"<fix>"},'
        '{"area":"<area>","issue":"<issue>","fix":"<fix>"}],'
        '"category_scores":{"Opener":0,"Behavioral":0,"Technical":0,"Situational":0,"Leadership":0,"Culture Fit":0},'
        '"action_plan":["<action>","<action>","<action>","<action>"],'
        '"personal_note":"<2-3 warm closing sentences>"}'
        + _brevity(provider, 55) +
        '\n\nSESSION:\n' + qa_summary +
        'RESUME:\n' + resume_ctx +
        '\nJOB:\n' + jd_ctx +
        '\nJSON only:'
    )
    raw = _call_with_retry(api_key, provider, model, prompt,
                           system_prompt="Return ONLY valid JSON. No markdown.",
                           temperature=0.4, max_tokens=max_tok)
    return _ej(raw)


# ═══════════════════════════════════════════════════════════════════
# 6. FREE CHAT
# ═══════════════════════════════════════════════════════════════════
def free_chat(api_key, provider, model, messages,
              resume_text="", jd_text="") -> str:
    is_ollama = provider == "ollama"
    history = ""
    for m in messages[-8:]:
        role = "You" if m["role"] == "user" else "Coach Alex"
        history += f"{role}: {m['content']}\n\n"

    resume_ctx = _ollama_trim(resume_text, "chat") if is_ollama else _trim(resume_text, 1000)
    jd_ctx     = _ollama_trim(jd_text,     "chat") if is_ollama else _trim(jd_text, 700)
    max_tok    = _OLLAMA_TOKENS["chat"] if is_ollama else _CHAT_TOKENS

    sys = (
        "You are Alex, a warm, direct experienced interview coach. "
        "Give specific, actionable advice in 2-4 paragraphs. "
        "Reference the candidate's background when available."
    )
    prompt = (
        ("RESUME:\n" + resume_ctx + "\n\n" if resume_ctx else "") +
        ("JOB:\n"    + jd_ctx     + "\n\n" if jd_ctx     else "") +
        "CONVERSATION:\n" + history + "Coach Alex:"
    )
    return call_llm(api_key, provider, model, prompt,
                    system_prompt=sys, temperature=0.65, max_tokens=max_tok)
