# config.py - 配置管理（UTF-8 编码）
# 敏感配置通过环境变量注入，LLM 配置通过 llm_settings.json 持久化
import json
import os


# ========== LLM 设置 ==========
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_SETTINGS_FILE = os.environ.get("LLM_SETTINGS_FILE", os.path.join(_CONFIG_DIR, "llm_settings.json"))

# 支持的 LLM 提供商（provider → 信息）
SUPPORTED_PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "description": "DeepSeek 官方 API",
        "default_model": "deepseek-chat",
        "default_base_url": "",
    },
    "openai": {
        "label": "OpenAI",
        "description": "OpenAI 官方 API（gpt-4o 等）",
        "default_model": "gpt-4o",
        "default_base_url": "https://api.openai.com/v1",
    },
    "anthropic": {
        "label": "Anthropic",
        "description": "Anthropic Claude API",
        "default_model": "claude-sonnet-4-20250514",
        "default_base_url": "https://api.anthropic.com",
    },
    "openai_compatible": {
        "label": "OpenAI 兼容",
        "description": "Sensenova 等 OpenAI 协议兼容的第三方服务",
        "default_model": "sensenova-6.8-flash-lite",
        "default_base_url": "https://token.sensenova.cn/v1",
    },
}

DEFAULT_LLM_SETTINGS = {
    "provider": "deepseek",
    "api_key": "",
    "base_url": "",
    "model": "deepseek-chat",
    "temperature": 0.5,
}


def load_llm_settings():
    """
    读取 LLM 设置文件。

    文件格式（新版，支持多配置）：
    {
        "configs": [
            {
                "id": "auto",
                "name": "主配置",
                "provider": "openai",
                "model": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-...",
                "temperature": 0.5,
                "active": true,
                "created_at": "2026-08-24T06:00:00"
            },
            ...
        ]
    }

    兼容旧版单配置格式（自动迁移）：
    { "provider": "...", "api_key": "...", ... }
    """
    if os.path.exists(LLM_SETTINGS_FILE):
        try:
            with open(LLM_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 兼容旧版单配置格式：{ "provider": "...", "api_key": "..." }
            if isinstance(data, dict) and "configs" not in data:
                old_config = dict(DEFAULT_LLM_SETTINGS)
                old_config.update(data)
                migrated = {
                    "configs": [{
                        "id": "auto",
                        "name": "主配置",
                        "provider": old_config.get("provider", "deepseek"),
                        "model": old_config.get("model", "deepseek-chat"),
                        "base_url": old_config.get("base_url", ""),
                        "api_key": old_config.get("api_key", ""),
                        "temperature": old_config.get("temperature", 0.5),
                        "active": True,
                        "created_at": data.get("created_at", ""),
                    }]
                }
                # 立即写回新格式
                save_llm_settings(migrated)
                data = migrated

            # 新版格式：configs 列表
            configs = data.get("configs", [])
            if not configs:
                # 空列表时创建默认配置
                configs = [{
                    "id": "auto",
                    "name": "主配置",
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "base_url": "",
                    "api_key": "",
                    "temperature": 0.5,
                    "active": True,
                    "created_at": "",
                }]

            # 确保至少有一条 active 配置
            if not any(c.get("active") for c in configs):
                configs[0]["active"] = True

            return {"configs": configs}

        except (json.JSONDecodeError, OSError):
            pass

    # 文件不存在或读取失败：返回默认单配置
    return {
        "configs": [{
            "id": "auto",
            "name": "主配置",
            "provider": "deepseek",
            "model": "deepseek-chat",
            "base_url": "",
            "api_key": "",
            "temperature": 0.5,
            "active": True,
            "created_at": "",
        }]
    }


def save_llm_settings(settings):
    """持久化 LLM 设置（多配置格式）"""
    with open(LLM_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def get_active_config():
    """获取当前激活的配置"""
    all_configs = load_llm_settings()
    for c in all_configs.get("configs", []):
        if c.get("active"):
            return c
    # 兜底：返回第一条
    if all_configs.get("configs"):
        return all_configs["configs"][0]
    return dict(DEFAULT_LLM_SETTINGS)


# 兼容旧版：如果设置文件里没有 api_key，尝试从环境变量读
# （只在本模块加载时执行一次）
_ENV_LLM_FALLBACK = get_active_config()
if not _ENV_LLM_FALLBACK.get("api_key"):
    _ENV_LLM_FALLBACK["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
    if not _ENV_LLM_FALLBACK["api_key"]:
        _ENV_LLM_FALLBACK["api_key"] = os.environ.get("OPENAI_API_KEY", "")

# 向后兼容：旧代码引用的 DEEPSEEK_API_KEY / DEEPSEEK_MODEL / TEMPERATURE
DEEPSEEK_API_KEY = _ENV_LLM_FALLBACK.get("api_key", "")
DEEPSEEK_MODEL = _ENV_LLM_FALLBACK.get("model", "deepseek-chat")
TEMPERATURE = float(_ENV_LLM_FALLBACK.get("temperature", "0.5"))


# ========== 安全配置 ==========
_ALLOWED_RAW_DIRS = os.environ.get(
    "ALLOWED_WRITE_DIRS", "C:/Users,/home,/tmp,./"
).split(",")
ALLOWED_WRITE_DIRS = [os.path.normpath(d.strip()) for d in _ALLOWED_RAW_DIRS if d.strip()]

DANGEROUS_COMMANDS = os.environ.get(
    "DANGEROUS_COMMANDS", "rm,dd,shutdown,reboot,format,mkfs"
).split(",")
DANGEROUS_COMMANDS = [c.strip() for c in DANGEROUS_COMMANDS if c.strip()]

# 简单 Token 鉴权（留空则跳过）
API_TOKEN = os.environ.get("API_TOKEN", "")


# ========== 文件读写安全 ==========
MAX_FILE_SIZE_BYTES = int(os.environ.get("MAX_FILE_SIZE_BYTES", 5 * 1024 * 1024))  # 5 MB
CMD_TIMEOUT_SECONDS = int(os.environ.get("CMD_TIMEOUT_SECONDS", "15"))


# ========== 日志配置 ==========
LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_FILE = os.environ.get("LOG_FILE", "agent.log")
LOG_MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", 5 * 1024 * 1024))
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "5"))


# ========== 记忆 / 检查点配置 ==========
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "checkpoints.sqlite")
DEFAULT_THREAD_ID = os.environ.get("DEFAULT_THREAD_ID", "1")
WAL_CHECKPOINT_ON_START = True

# ========== 图表目录（Excel 分析生成图表） ==========
CHART_DIR = os.environ.get("CHART_DIR", os.path.join(LOG_DIR, "charts"))

# ========== 知识库图片目录（PDF/DOCX 导入时提取的图片） ==========
KB_IMAGE_DIR = os.environ.get("KB_IMAGE_DIR", os.path.join(LOG_DIR, "kb_images"))

# ========== API 自动重试配置 ==========
MAX_AGENT_RETRIES = int(os.environ.get("MAX_AGENT_RETRIES", "5"))
RETRY_WAIT_BASE = int(os.environ.get("RETRY_WAIT_BASE", "2"))
RETRY_WAIT_MAX = int(os.environ.get("RETRY_WAIT_MAX", "30"))

# ========== Agent 循环防护配置 ==========
MAX_AGENT_ITERATIONS = int(os.environ.get("MAX_AGENT_ITERATIONS", "50"))   # LangGraph 递归上限
MAX_TOOL_CALLS_PER_TURN = int(os.environ.get("MAX_TOOL_CALLS_PER_TURN", "30"))   # 单轮最大工具调用次数
MAX_SAME_TOOL_REPEATS = int(os.environ.get("MAX_SAME_TOOL_REPEATS", "5"))   # 相同工具+参数重复上限

# ========== 上下文窗口管理 ==========
# 不同模型的上下文窗口大小（token），未列出则用 DEFAULT
MODEL_CONTEXT_WINDOWS = {
    "deepseek-chat": 32000, "deepseek-coder": 128000,
    "gpt-4o": 128000, "gpt-4o-mini": 128000, "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 16000,
    "claude-sonnet-4-20250514": 200000, "claude-3": 200000, "claude-3-5": 200000,
    "glm-5.2": 128000, "glm-4": 128000,
}
DEFAULT_CONTEXT_WINDOW = 32000
# 触发裁剪的比例（占窗口的百分比）
CONTEXT_TRIM_RATIO = float(os.environ.get("CONTEXT_TRIM_RATIO", "0.7"))
# 保留最近几轮对话（1 轮 = 1 条用户消息 + 1 条 AI 回复）
CONTEXT_KEEP_RECENT_TURNS = int(os.environ.get("CONTEXT_KEEP_RECENT_TURNS", "8"))
# 是否启用 LLM 摘要（关闭则直接丢弃旧消息）
CONTEXT_SUMMARY_ENABLED = os.environ.get("CONTEXT_SUMMARY_DISABLED", "0") != "1"

# ========== 检查点瘦身配置 ==========
# BoundedSqliteSaver：每个 thread 保留的 checkpoints 行数上限；
# 超出时按最老→最新自动删除（保留 head + 最近 keep-1 条，维持 parent 链完整）
MAX_CHECKPOINTS_PER_THREAD = int(os.environ.get("MAX_CHECKPOINTS_PER_THREAD", "30"))
# compact_checkpoints_db 的目标 checkpoints 表体积上限（MB）；超过时继续裁剪旧行
CHECKPOINT_SIZE_MB = float(os.environ.get("CHECKPOINT_SIZE_MB", "50.0"))