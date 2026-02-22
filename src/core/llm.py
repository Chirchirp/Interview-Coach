"""
Interview Coach AI - LLM Core  (v4 — Human Voice Edition)
Supports: Groq (FREE), OpenAI, Anthropic, OpenRouter, Ollama (LOCAL FREE)

Key change in v4:
  model_answer is now a full first-person interview answer spoken as the
  candidate would say it in the room — conversational, STAR-structured,
  with real context and a genuine human feel. Not a coaching summary.
  Alex also explains WHY the answer scores full marks so the user learns
  the pattern, not just memorises the answer.
"""
from __future__ import annotations
import json, re, time


# ── Token budgets ─────────────────────────────────────────────────
_RESUME_LIMIT  = 3000
_JD_LIMIT      = 2000
_CHAT_TOKENS   = 900
_GRADE_TOKENS  = 1400   # raised — human model answers need more room
_PLAN_TOKENS   = 1000
_TIPS_TOKENS   = 500
_REPORT_TOKENS = 1400

# Ollama-specific tighter budgets
_OLLAMA_TOKENS = {
    "plan":   800,
    "grade":  900,     # raised from 380 to fit the human-voice model answer
    "tip":    280,
    "chat":   500,
    "report": 900,
}

# Ollama context trim limits (chars)
_OLLAMA_CTX = {
    "plan":   1200,
    "grade":   400,
    "tip":     200,
    "chat":    400,
    "report":  400,
}


def _trim(text: str, limit: int) -> str:
    return text[:limit] + "\n[truncated]" if len(text) > limit else text


def _ollama_trim(text: str, task: str) -> str:
    if not text or not text.strip():
        return ""
    limit = _OLLAMA_CTX.get(task, 400)
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[...]\n" + text[-half:]


def _brevity(provider: str, words: int = 50) -> str:
    return (f"\nIMPORTANT: Keep every field under {words} words. Be concise."
            if provider == "ollama" else "")


# ── Robust JSON extractor ─────────────────────────────────────────
def _ej(text: str):
    if not text or not text.strip():
        raise ValueError("Empty response from model.")
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
                try:    return json.loads(text[start:i+1])
                except: return json.loads(_repair_json(text[start:i+1]))
    candidate = text[start:]
    try:    return json.loads(candidate)
    except: return json.loads(_repair_json(candidate))


def _repair_json(s: str) -> str:
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
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            p = prompt if attempt == 1 else (
                prompt + f"\n\n[Attempt {attempt}: output ONLY valid JSON "
                "starting with {{ or [. No other text.]"
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
    return _ollama_root(raw) + "/v1"


def _ollama_warmup(api_key: str, model: str) -> None:
    try:
        import urllib.request as _ur
        root    = _ollama_root(api_key)
        payload = json.dumps({"model": model, "prompt": "", "keep_alive": "10m"}).encode()
        req = _ur.Request(f"{root}/api/generate", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        _ur.urlopen(req, timeout=4)
    except Exception:
        pass


def get_ollama_models(api_key: str) -> dict:
    try:
        import urllib.request as _ur, json as _json
        root = _ollama_root(api_key)
        req  = _ur.urlopen(f"{root}/api/tags", timeout=4)
        data = _json.loads(req.read())
        installed = [m["name"] for m in data.get("models", [])]
        if not installed:
            return {"(no models found — run: ollama pull phi3.5)": "phi3.5"}
        return {name: name for name in installed}
    except Exception:
        return {
            "llama3.2 (default)": "llama3.2",
            "llama3.1":           "llama3.1",
            "mistral":            "mistral",
            "phi3":               "phi3",
        }


# ── Provider model catalogue ──────────────────────────────────────
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


# ── Core LLM call ────────────────────────────────────────────────
def call_llm(api_key, provider, model, prompt,
             system_prompt="You are a helpful assistant.",
             temperature=0.4, max_tokens=700) -> str:
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
    if provider == "ollama":
        _ollama_warmup(api_key, model)

    is_ollama  = provider == "ollama"
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
# 1b. FIELD-ONLY PLAN
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
# 2. GRADE ANSWER  ←  core change in v4
# ═══════════════════════════════════════════════════════════════════
def grade_answer(api_key, provider, model, question, user_answer,
                 category, resume_text, jd_text) -> dict:
    """
    v4: model_answer is now a complete first-person spoken answer —
    the exact words a confident candidate would say in the room.
    Natural, STAR-structured without labels, specific, with a real outcome.

    New field: model_answer_breakdown — explains WHY each part scores well,
    so the candidate learns the pattern, not just memorises the answer.
    """
    is_ollama  = provider == "ollama"
    resume_ctx = _ollama_trim(resume_text, "grade") if is_ollama else _trim(resume_text, 1000)
    jd_ctx     = _ollama_trim(jd_text,     "grade") if is_ollama else _trim(jd_text, 600)
    max_tok    = _OLLAMA_TOKENS["grade"] if is_ollama else _GRADE_TOKENS

    system = (
        "You are Alex, a senior interview coach with 15 years of experience "
        "placing candidates at top companies. You have heard thousands of answers "
        "and know exactly what separates a 100-point answer from a 60-point one. "
        "You are warm, direct, and always specific — never generic. "
        "\n\n"
        "When you write a model_answer, you write it as if YOU are the candidate — "
        "speaking naturally in the first person, the way a confident person actually "
        "talks in an interview room. Think of it as: if I were this candidate and I "
        "wanted to give the perfect answer, what would I literally say out loud? "
        "\n\n"
        "The model_answer must:\n"
        "- Be written in first person (I, me, my, we)\n"
        "- Sound conversational and human — no bullet points, no labels like 'Situation:'\n"
        "- Follow the STAR flow naturally without announcing it\n"
        "- Include a specific, believable scenario with real-feeling context\n"
        "- Mention concrete actions, tools, or decisions made\n"
        "- End with a clear, quantified or observable result\n"
        "- Be 5-8 sentences — enough to be complete, not so long it feels rehearsed\n"
        "- Use the candidate's resume background where possible to make it personal\n"
        "\n"
        "Return ONLY valid JSON. No markdown. No extra text."
    )

    prompt = (
        'Grade this interview answer and write a full human-voice model answer.\n\n'
        'Return ONLY this JSON:\n'
        '{\n'
        '  "score": <integer 0-100>,\n'
        '  "grade": "<A|B|C|D|F>",\n'
        '  "star_scores": {"situation":<0-25>,"task":<0-25>,"action":<0-25>,"result":<0-25>},\n'
        '  "what_worked": [\n'
        '    "<specific strength referencing the candidate\'s actual words>",\n'
        '    "<another specific strength>"\n'
        '  ],\n'
        '  "what_missed": [\n'
        '    "<specific gap — what was vague, missing, or weak>",\n'
        '    "<another gap>"\n'
        '  ],\n'
        '  "coach_reaction": "<2-3 warm human sentences from Alex reacting to THIS specific '
        'answer. Use the candidate\'s actual words. Acknowledge what landed before noting what '
        'to improve. Sound like a real coach who just listened carefully, not an AI report.>",\n'
        '  "model_answer": "<Write the complete answer as if you ARE the candidate speaking '
        'in the interview room right now. First person. Conversational. Natural STAR flow without '
        'labels. Open with the situation — set the scene briefly so the interviewer can picture it. '
        'Then explain the challenge or what you needed to do. Then walk through what you specifically '
        'did — decisions made, tools used, people involved. Then land on the result — what changed, '
        'what improved, what you learned, ideally with a number or observable outcome. '
        'A brief genuine reflection at the end is fine. 5-8 sentences. '
        'Use the candidate\'s resume background to personalise it. '
        'This should feel like something a real, articulate person said — not a template.>",\n'
        '  "model_answer_breakdown": "<3-4 sentences explaining WHY this answer would score top '
        'marks. What STAR elements does it hit? Why does the specificity matter to the interviewer? '
        'What signals does this send about the candidate\'s competence? '
        'Teach the pattern so they can apply it to any question.>",\n'
        '  "follow_up_question": "<one natural follow-up the interviewer would ask based on '
        'what the model answer revealed — something that probes deeper or explores a new angle>",\n'
        '  "encouragement": "<1-2 sentences of specific encouragement. Find something real in '
        'their actual answer that showed promise and name it. Then give one concrete thing to '
        'practise. Not \'great effort!\' — something real and useful.>"\n'
        '}\n\n'
        f'QUESTION: {question}\n'
        f'CATEGORY: {category}\n\n'
        f'CANDIDATE ANSWER:\n{user_answer[:1200]}\n\n'
        + (f'CANDIDATE RESUME (personalise the model answer using this):\n{resume_ctx}\n\n' if resume_ctx else '')
        + (f'JOB CONTEXT:\n{jd_ctx}\n\n' if jd_ctx else '')
        + 'JSON only:'
    )

    raw = _call_with_retry(api_key, provider, model, prompt,
                           system_prompt=system,
                           temperature=0.72,  # higher temp = more natural-sounding answers
                           max_tokens=max_tok)
    result = _ej(raw)

    # Backward compat — add breakdown if model skipped it
    result.setdefault(
        "model_answer_breakdown",
        "This answer works because it gives the interviewer a specific, memorable story "
        "with a clear outcome. Specificity is what separates a B answer from an A — "
        "vague answers make interviewers nervous, concrete ones build confidence."
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# 3. FOLLOW-UP COACHING CHAT
# ═══════════════════════════════════════════════════════════════════
def coach_followup(api_key, provider, model, conversation_history,
                   resume_text, jd_text) -> str:
    is_ollama = provider == "ollama"
    history_str = ""
    for msg in conversation_history[-8:]:
        role = "Candidate" if msg["role"] == "user" else "Coach Alex"
        history_str += f"{role}: {msg['content']}\n\n"

    resume_ctx = _ollama_trim(resume_text, "chat") if is_ollama else _trim(resume_text, 1000)
    jd_ctx     = _ollama_trim(jd_text,     "chat") if is_ollama else _trim(jd_text, 700)
    max_tok    = _OLLAMA_TOKENS["chat"] if is_ollama else _CHAT_TOKENS

    sys = (
        "You are Alex, a senior interview coach with 15 years of experience. "
        "You speak like a real human being — warm, direct, occasionally using phrases like "
        "'here's the thing', 'what I'd actually say here is', 'honestly', 'in my experience'. "
        "You always reference the candidate's actual words from the conversation. "
        "You never give generic advice. When you suggest how to improve an answer, "
        "you sometimes give a short example of how you'd phrase it — written in first "
        "person as if the candidate is saying it. "
        "You are a real coach who has just spent time with this person, not a chatbot. "
        "Keep to 2-4 short paragraphs."
    )
    prompt = (
        "CANDIDATE RESUME:\n" + resume_ctx +
        "\n\nJOB THEY'RE APPLYING FOR:\n" + jd_ctx +
        "\n\nCONVERSATION SO FAR:\n" + history_str +
        "\nCoach Alex (respond naturally, as a real human coach would):"
    )
    return call_llm(api_key, provider, model, prompt,
                    system_prompt=sys, temperature=0.72, max_tokens=max_tok)


# ═══════════════════════════════════════════════════════════════════
# 4. QUICK TIP
# ═══════════════════════════════════════════════════════════════════
def get_question_tip(api_key, provider, model, question, category,
                     resume_text, jd_text) -> str:
    is_ollama  = provider == "ollama"
    resume_ctx = _ollama_trim(resume_text, "tip") if is_ollama else _trim(resume_text, 1200)
    jd_ctx     = _ollama_trim(jd_text,     "tip") if is_ollama else _trim(jd_text, 800)
    max_tok    = _OLLAMA_TOKENS["tip"] if is_ollama else _TIPS_TOKENS

    sys = (
        "You are Alex, a senior interview coach. Give a quick, specific tip on how to "
        "answer this question. Sound like a real human — direct, warm. Use phrases like "
        "'what interviewers are really looking for here is...' or "
        "'the mistake most people make with this one is...'. "
        "Reference the candidate's background specifically. 3-5 sentences."
    )
    prompt = (
        f"Question: {question}\nCategory: {category}\n\n"
        "Give a specific, personalised coaching tip for this question.\n"
        + _brevity(provider, 70) +
        "\n\nCANDIDATE RESUME:\n" + resume_ctx +
        "\n\nJOB:\n" + jd_ctx
    )
    return call_llm(api_key, provider, model, prompt,
                    system_prompt=sys, temperature=0.65, max_tokens=max_tok)


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
            f"Q{i} [{item.get('category','')}]: {item.get('question','')[:80]}\n"
            f"Score: {g.get('score',0)}/100 ({g.get('grade','?')}) | "
            f"Worked: {'; '.join(g.get('what_worked',[])[:1])} | "
            f"Missed: {'; '.join(g.get('what_missed',[]))}\n\n"
        )

    resume_ctx = _ollama_trim(resume_text, "report") if is_ollama else _trim(resume_text, 800)
    jd_ctx     = _ollama_trim(jd_text,     "report") if is_ollama else _trim(jd_text, 600)
    max_tok    = _OLLAMA_TOKENS["report"] if is_ollama else _REPORT_TOKENS

    prompt = (
        'Generate a final interview coaching report. Return ONLY JSON:\n'
        '{"overall_score":<0-100>,"overall_grade":"A|B|C|D|F",'
        '"tier":"Interview Ready|Almost There|Needs Practice|Significant Work Needed",'
        '"headline":"<one punchy specific sentence summarising their performance — not generic>",'
        '"top_strengths":["<specific strength observed in their answers>","<strength>","<strength>"],'
        '"priority_improvements":['
        '{"area":"<area>","issue":"<specific issue from their actual answers>",'
        '"fix":"<concrete actionable fix — if possible, give an example of what good looks like>"},'
        '{"area":"<area>","issue":"<specific issue>","fix":"<fix>"},'
        '{"area":"<area>","issue":"<specific issue>","fix":"<fix>"}],'
        '"category_scores":{"Opener":0,"Behavioral":0,"Technical":0,"Situational":0,"Leadership":0,"Culture Fit":0},'
        '"action_plan":["<specific concrete thing to practise this week>","<action>","<action>","<action>"],'
        '"personal_note":"<3-4 warm personal closing sentences from Alex. Reference specific moments '
        'from the session. Acknowledge the candidate\'s effort genuinely. End on an encouraging note '
        'that feels like it comes from a real coach who just spent an hour with them, not a template.>"}'
        + _brevity(provider, 60) +
        '\n\nSESSION DATA:\n' + qa_summary +
        'RESUME:\n' + resume_ctx +
        '\nJOB:\n' + jd_ctx +
        '\nJSON only:'
    )
    raw = _call_with_retry(api_key, provider, model, prompt,
                           system_prompt=(
                               "You are Alex, a senior interview coach writing a personalised "
                               "end-of-session report. Be specific, warm, and human — this person "
                               "just spent time with you and deserves a real response, not a template. "
                               "Return ONLY valid JSON. No markdown."
                           ),
                           temperature=0.65, max_tokens=max_tok)
    return _ej(raw)


# ═══════════════════════════════════════════════════════════════════
# 6. FREE CHAT
# ═══════════════════════════════════════════════════════════════════
def free_chat(api_key, provider, model, messages,
              resume_text="", jd_text="") -> str:
    is_ollama = provider == "ollama"
    history = ""
    for m in messages[-10:]:
        role = "You" if m["role"] == "user" else "Coach Alex"
        history += f"{role}: {m['content']}\n\n"

    resume_ctx = _ollama_trim(resume_text, "chat") if is_ollama else _trim(resume_text, 1000)
    jd_ctx     = _ollama_trim(jd_text,     "chat") if is_ollama else _trim(jd_text, 700)
    max_tok    = _OLLAMA_TOKENS["chat"] if is_ollama else _CHAT_TOKENS

    sys = (
        "You are Alex, a senior interview coach with 15 years of experience. "
        "You speak like a real human being — warm, occasionally direct or even blunt, "
        "sometimes self-deprecating. You use phrases like 'here's what I'd actually do', "
        "'honestly', 'in my experience', 'the thing most candidates miss here'. "
        "You give specific actionable advice. When you give answer examples you write them "
        "in first person as the candidate would say them. "
        "You genuinely want this person to succeed — it comes through in how you respond. "
        "Keep replies to 2-4 paragraphs. Be a real coach, not a chatbot."
    )
    prompt = (
        ("CANDIDATE RESUME:\n" + resume_ctx + "\n\n" if resume_ctx else "") +
        ("TARGET JOB:\n"       + jd_ctx     + "\n\n" if jd_ctx     else "") +
        "CONVERSATION:\n" + history + "Coach Alex:"
    )
    return call_llm(api_key, provider, model, prompt,
                    system_prompt=sys, temperature=0.72, max_tokens=max_tok)


# ═══════════════════════════════════════════════════════════════════
# 7. VOICE TRANSCRIPTION  (Groq Whisper / OpenAI Whisper)
# ═══════════════════════════════════════════════════════════════════
def transcribe_audio(audio_bytes: bytes, api_key: str, provider: str) -> str:
    """
    Transcribe raw audio bytes (WebM/Opus from browser MediaRecorder) to text.

    Backends:
      groq       → Groq Whisper large-v3-turbo (free, fast, best quality)
      openai     → OpenAI Whisper-1
      ollama     → faster-whisper on CPU (needs: pip install faster-whisper)
      anthropic / openrouter → raises helpful error (no audio API)

    Returns the transcribed text string.
    """
    import tempfile, os

    # Write audio to temp file — all Whisper APIs need a file path
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=(os.path.basename(tmp_path), f.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text",
                    language="en",
                )
            return str(result).strip()

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=f, model="whisper-1", response_format="text",
                )
            return str(result).strip()

        elif provider == "ollama":
            # Try faster-whisper (local CPU, no API key needed)
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel("base", device="cpu", compute_type="int8")
                segments, _ = model.transcribe(tmp_path, language="en")
                return " ".join(s.text.strip() for s in segments).strip()
            except ImportError:
                raise ValueError(
                    "Ollama doesn't include speech-to-text. "
                    "Install faster-whisper for local transcription: "
                    "pip install faster-whisper  — or switch to Groq (free) "
                    "for easy cloud transcription."
                )

        else:
            # anthropic, openrouter — no audio API
            raise ValueError(
                f"{provider.title()} doesn't support audio transcription. "
                "Switch to Groq (free) or OpenAI to use voice recording."
            )

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
