"""Prompt construction for evaluation."""

from eval.config import NO_DEMAND_TOKEN

# ---- Three prompt levels ----

_INTERVENTION_TYPES = """Types of valuable proactive interventions:
- The user is about to make a decision but is missing key information or hasn't considered an important trade-off
- Something said now connects to or contradicts a specific point from earlier in the conversation, and the user likely hasn't noticed
- There is a hidden risk, pitfall, or common mistake that the user hasn't accounted for
- The user shows a clear misconception or factual error that would lead them astray
- Scattered information across the conversation needs to be tied together for the user to see the big picture"""

_OUTPUT_RULES = f"""Output rules:
- If you speak up: output your response directly. Be concise, specific, and actionable. Match the conversation's language.
- If no intervention needed: output exactly: {NO_DEMAND_TOKEN}
- Do NOT explain your reasoning. Output ONLY your response or {NO_DEMAND_TOKEN}"""

SYSTEM_PROMPTS = {
    "encouraging": """You are a proactive AI assistant serving {{primary_user}}.

Your job: actively look for opportunities to help {{primary_user}}. Whenever you see a chance to add value — a missing piece of information, a potential risk, a useful connection to something said earlier, or a way to clarify or deepen the discussion — speak up. It is better to offer a helpful nudge than to stay silent and let the user miss something.

{intervention_types}

{output_rules}""",

    "neutral": """You are a proactive AI assistant serving {{primary_user}}.

Your job: at each turn, decide whether to proactively speak up to help {{primary_user}} — only when there is genuine value. Most turns need NO intervention. Only speak up when you are confident the user would clearly benefit.

{intervention_types}

{output_rules}""",

    "suppressing": """You are a proactive AI assistant serving {{primary_user}}.

Your job: stay silent unless there is a critical need for {{primary_user}}. The vast majority of turns require NO intervention. Only speak up when you identify a clear, significant issue that the user is very likely to miss and that would materially affect their outcome. When in doubt, do NOT intervene.

{intervention_types}

{output_rules}""",
}

# Category-specific context hints
CATEGORY_HINTS = {
    "W": """You are observing a work/business conversation. Pay attention to:
- Decisions being made without sufficient data or with overlooked risks
- Resource, timeline, or priority conflicts that the speaker hasn't flagged
- Earlier commitments, metrics, or constraints that are relevant to the current discussion""",

    "L": """You are observing a learning/educational conversation (lecture, tutorial, discussion). Pay attention to:
- Clear misconceptions or factual errors that would hinder understanding
- Connections between the current topic and concepts introduced earlier that would deepen learning
- Note: someone simply explaining a topic is NOT a reason to intervene — only step in when there's a genuine gap or error""",

    "D": """You are observing a casual/daily conversation (personal discussion, tool usage, general knowledge). Pay attention to:
- Practical tips or warnings relevant to the user's situation that they haven't considered
- Connections to earlier points in the conversation that the user might have forgotten
- Emotional cues that suggest the user might benefit from acknowledgment or a different perspective""",
}


def _build_scene_header(session: dict) -> str:
    """Build scene + characters header from session."""
    scene = session.get("scene", {})
    if not scene:
        return ""
    parts = []
    if scene.get("scene"):
        parts.append(f"Scene: {scene['scene']}")
    chars = scene.get("characters", [])
    if chars:
        char_lines = []
        for c in chars:
            desc = f"{c['name']} ({c.get('role', '')})"
            if c.get("background"):
                desc += f" — {c['background']}"
            char_lines.append(f"  - {desc}")
        parts.append("Characters:\n" + "\n".join(char_lines))
    pu = scene.get("primary_user", {})
    if pu:
        parts.append(f"You are serving: {pu.get('name', '?')} — {pu.get('reason', '')}")
    return "\n".join(parts)


def _format_turn(turn: dict) -> str:
    """Format a single turn as 'speaker: text'."""
    return f"{turn.get('speaker', '?')}: {turn.get('text', '')}"


def build_prompt(session: dict, turn_idx: int, level: str = "neutral") -> list[dict]:
    """Build [system, user] messages for evaluating one turn."""
    turns = session["turns"]
    memory = session.get("memory")
    subcat = session.get("subcategory", "")
    scene = session.get("scene", {})

    # Primary user name for system prompt
    pu = scene.get("primary_user", {})
    pu_name = pu.get("name", "the user")

    # Build system prompt
    template = SYSTEM_PROMPTS[level]
    system_base = template.replace("{{primary_user}}", pu_name).format(
        intervention_types=_INTERVENTION_TYPES,
        output_rules=_OUTPUT_RULES,
    )
    category_key = subcat[0] if subcat else ""
    hint = CATEGORY_HINTS.get(category_key, "")
    system_msg = f"{system_base}\n\n{hint}" if hint else system_base

    # Build user message: scene + memory + conversation + current turn
    ctx_parts = []
    scene_header = _build_scene_header(session)
    if scene_header:
        ctx_parts.append(f"[Scene]\n{scene_header}")
    if memory:
        ctx_parts.append(f"[Memory from earlier conversation]\n{memory}")
    if turns[:turn_idx]:
        conv_lines = [_format_turn(t) for t in turns[:turn_idx]]
        ctx_parts.append("[Conversation]\n" + "\n".join(conv_lines))
    context = "\n\n".join(ctx_parts) if ctx_parts else "(conversation start)"

    current = _format_turn(turns[turn_idx])

    user_msg = f"{context}\n\n[Current turn]\n{current}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
