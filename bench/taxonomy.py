"""
Benchmark taxonomy definitions for IntentFlow proactive demand detection.

Hierarchy:
  3 main categories (W/L/D) × 3-4 subcategories = 10 subcategories
  Subcategories are data-driven, clustered from actual podcast data.
  Each subcategory has associated demand types + "other"

Demand types:
  Req (Requirement): explicit, clear-trigger demands
  Ins (Insight):     implicit, subjective, interpretive demands
"""

# ------------------------------------------------------------------
# Category / subcategory definitions (data-driven)
# ------------------------------------------------------------------

CATEGORIES = {
    "Work": {
        "code": "W",
        "zh": "工作",
        "subcategories": {
            "W1": {"name": "BusinessMetrics",    "zh": "业绩指标与目标",     "desc": "revenue/KPI/GMV discussion, target setting, performance review, forecast alignment"},
            "W2": {"name": "ProductStrategy",     "zh": "产品与市场策略",     "desc": "product roadmap, pricing, go-to-market, partnerships, fundraising, business model"},
            "W3": {"name": "TechEngineering",     "zh": "技术研发与工程",     "desc": "model experiments, code/data processing, algorithms, system implementation, R&D discussion"},
            "W4": {"name": "WorkplaceCollab",     "zh": "职场沟通与协作",     "desc": "meeting coordination, document/PPT preparation, task assignment, cross-team communication, HR/career"},
        },
    },
    "Learning": {
        "code": "L",
        "zh": "学习",
        "subcategories": {
            "L1": {"name": "STEMLecture",         "zh": "理工科与AI课程",     "desc": "STEM lectures: math, physics, chemistry, ML/RL/diffusion models, theoretical explanation"},
            "L2": {"name": "ProgrammingTutorial",  "zh": "编程与系统实操",     "desc": "hands-on coding, debugging, repo/file structure, Linux/tools, IDE operation"},
            "L3": {"name": "HumanitiesBusiness",   "zh": "人文社科与语言学习", "desc": "history, art, economics, policy, product/market analysis, language learning methods"},
        },
    },
    "Daily": {
        "code": "D",
        "zh": "日常",
        "subcategories": {
            "D1": {"name": "PersonalLife",         "zh": "个人生活与情绪社交", "desc": "emotions, family, relationships, health, casual chat, personal stories"},
            "D2": {"name": "ToolsWorkflow",        "zh": "工具产品与工作流",   "desc": "AI/tool selection, app/plugin usage, API integration, workflow optimization, schedule management"},
            "D3": {"name": "ContentKnowledge",     "zh": "内容消费与知识探索", "desc": "news, politics, economics, science, math, media/entertainment content discussion"},
        },
    },
}

# ------------------------------------------------------------------
# Demand type definitions (shared across subcategories)
# ------------------------------------------------------------------

DEMAND_TYPES = {
    # Req-leaning (clear external triggers)
    "decision_support":    {"type": "Req", "zh": "决策支持",   "desc": "explicit request for recommendation or decision help"},
    "information_lookup":  {"type": "Req", "zh": "信息查询",   "desc": "fact lookup, definition, data retrieval"},
    "task_planning":       {"type": "Req", "zh": "任务规划",   "desc": "step-by-step planning, scheduling, to-do creation"},
    "problem_solving":     {"type": "Req", "zh": "问题解决",   "desc": "debugging, troubleshooting, concrete issue resolution"},
    "summarization":       {"type": "Req", "zh": "内容总结",   "desc": "summarize prior content, key points extraction"},

    # Ins-leaning (implicit, interpretive)
    "trend_insight":       {"type": "Ins", "zh": "趋势洞察",   "desc": "identify patterns, emerging trends without explicit ask"},
    "risk_warning":        {"type": "Ins", "zh": "风险预警",   "desc": "proactively surface hidden risks or concerns"},
    "sentiment_analysis":  {"type": "Ins", "zh": "情感分析",   "desc": "emotional state reading, mood tracking"},
    "knowledge_gap":       {"type": "Ins", "zh": "知识缺口",   "desc": "detect user's knowledge gap and offer proactive fill"},
    "callback_reminder":   {"type": "Ins", "zh": "回调提醒",   "desc": "current turn connects to/contradicts/builds upon a specific prior point; proactively remind the user of that link"},
    "context_synthesis":   {"type": "Ins", "zh": "全局综合",   "desc": "conversation has accumulated fragmented info and user may have lost the big picture; proactively synthesize key points"},

    "other":               {"type": "other", "zh": "其他",     "desc": "demand that doesn't fit above categories"},
}

# Demand types expected per subcategory (for annotation prompts)
SUBCATEGORY_DEMANDS = {
    "W1": ["decision_support", "task_planning", "summarization", "risk_warning", "callback_reminder", "other"],
    "W2": ["decision_support", "trend_insight", "risk_warning", "task_planning", "callback_reminder", "other"],
    "W3": ["problem_solving", "information_lookup", "knowledge_gap", "task_planning", "callback_reminder", "other"],
    "W4": ["task_planning", "summarization", "sentiment_analysis", "risk_warning", "callback_reminder", "other"],

    "L1": ["knowledge_gap", "summarization", "information_lookup", "callback_reminder", "context_synthesis", "other"],
    "L2": ["problem_solving", "knowledge_gap", "task_planning", "callback_reminder", "information_lookup", "other"],
    "L3": ["knowledge_gap", "information_lookup", "trend_insight", "callback_reminder", "context_synthesis", "other"],

    "D1": ["sentiment_analysis", "decision_support", "risk_warning", "callback_reminder", "context_synthesis", "other"],
    "D2": ["task_planning", "decision_support", "information_lookup", "callback_reminder", "trend_insight", "other"],
    "D3": ["trend_insight", "knowledge_gap", "summarization", "callback_reminder", "context_synthesis", "other"],
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_all_subcategory_codes() -> list[str]:
    codes = []
    for cat in CATEGORIES.values():
        codes.extend(cat["subcategories"].keys())
    return codes


def get_demand_types_for(subcat_code: str) -> list[str]:
    return SUBCATEGORY_DEMANDS.get(subcat_code, list(DEMAND_TYPES.keys()))


def describe_subcategory(subcat_code: str) -> str:
    for cat in CATEGORIES.values():
        if subcat_code in cat["subcategories"]:
            sub = cat["subcategories"][subcat_code]
            return f"{subcat_code} {sub['name']} ({sub['zh']}): {sub['desc']}"
    return subcat_code


def describe_demand_type(dtype: str) -> str:
    d = DEMAND_TYPES.get(dtype, {})
    return f"{dtype} ({d.get('zh', '')}): {d.get('desc', '')}"
