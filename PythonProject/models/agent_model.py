import sqlite3

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from models.checkpoints import BoundedSqliteSaver

from config import (
    load_llm_settings,
    get_active_config,
    SUPPORTED_PROVIDERS,
    CHECKPOINT_DB,
    DEFAULT_THREAD_ID,
    WAL_CHECKPOINT_ON_START,
    MAX_AGENT_ITERATIONS,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_CHECKPOINTS_PER_THREAD,
)
from models.tools import (
    get_time,
    execute_command,
    read_file,
    write_file,
    list_processes,
    kill_process,
    list_services,
    network_check,
    system_stats,
    search_log,
    list_dir,
    search_kb,
    list_hosts,
    execute_on_host,
    run_on_all_hosts,
    check_host_health,
    excel_summary,
    excel_filter,
    excel_aggregate,
    excel_chart,
)
from services.logger_service import get_logger

logger = get_logger()

# 模块级 SQLite 连接（跨 Agent 重建复用）
sqlite_connection = None


def get_sqlite_connection():
    """返回全局 SQLite 连接（供 WAL checkpoint / Agent 重建复用）"""
    return sqlite_connection

# 系统提示词
SYSTEM_PROMPT = """你是一个桌面运维助手。你可以使用以下工具：
- get_time: 获取当前时间
- execute_command: 执行系统命令（只读类操作，如 df, ls, ping）
- read_file: 智能读取文件内容
- write_file: 智能写入文件（限用户目录）
- list_processes: 列出进程（按内存排序）
- kill_process: 结束指定 PID 进程（危险操作，需用户确认）
- list_services: 列出系统服务状态
- network_check: 网络诊断（DNS + ping）
- system_stats: CPU/内存/磁盘快照
- search_log: 在日志文件中检索关键字
- list_dir: 列出目录内容
- search_kb: 检索知识库中过往解决方案
- list_hosts: 列出已配置的远程主机
- execute_on_host: 在指定远程主机上执行命令
- run_on_all_hosts: 在所有远程主机上并行执行命令
- check_host_health: 检查远程主机连通性
- excel_summary: 分析 Excel 文件结构、列信息、数值统计
- excel_filter: 按条件筛选 Excel 行数据
- excel_aggregate: 分组聚合统计 Excel 数据
- excel_chart: 生成 Excel 数据图表（bar/line/pie）

**重要：最终回复时，请使用 Markdown 格式组织你的回答，让内容清晰易读。**  
要求：
- 使用 `#`、`##` 作为标题分层
- 使用 `-` 列表展示多个要点
- 使用 `**粗体**` 强调关键信息
- 命令输出结果用 ` ``` ` 代码块包裹
- 适当添加空行分隔不同部分
- 回复时尽量少用表情，语句精简准确
- **执行危险操作前（kill_process 等）先说明将做什么，等待用户确认**
- 当用户问题与历史解决方案相关时，先用 search_kb 检索知识库
当用户提问时，请根据需要使用工具，并最终用 Markdown 格式的自然语言回答用户的问题。
"""


# ============================================================
# LLM 动态创建
# ============================================================
def create_llm(settings):
    """
    根据 settings 中的 provider 创建对应的 LLM 实例。

    支持：
    - deepseek       → langchain_deepseek.ChatDeepSeek
    - openai         → langchain_openai.ChatOpenAI
    - anthropic      → langchain_anthropic.ChatAnthropic
    - openai_compatible → langchain_openai.ChatOpenAI（自定义 base_url）
    """
    provider = settings.get("provider", "deepseek")
    api_key = (settings.get("api_key") or "").strip()
    model = (settings.get("model") or "").strip()
    try:
        temperature = float(settings.get("temperature", 0.5))
    except (TypeError, ValueError):
        temperature = 0.5
    base_url = (settings.get("base_url") or "").strip()

    # 未配置 Key / 模型时给出明确提示（而不是 pydantic ValidationError）
    if not api_key:
        raise ValueError(
            "未配置 API Key。请先到「⚙️ 设置」页面选择 LLM 提供商并填写 API Key，保存后即可使用。"
        )
    if not model:
        raise ValueError("未配置模型名称，请到「⚙️ 设置」页面填写。")

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
        logger.info(f"LLM: DeepSeek ({model}, temp={temperature})")

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )
        logger.info(f"LLM: OpenAI ({model}, temp={temperature})")

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url or "https://api.anthropic.com",
        )
        logger.info(f"LLM: Anthropic Claude ({model}, temp={temperature})")

    elif provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url or SUPPORTED_PROVIDERS["openai_compatible"]["default_base_url"],
        )
        logger.info(f"LLM: OpenAI 兼容 ({model}, base={base_url or SUPPORTED_PROVIDERS['openai_compatible']['default_base_url']})")

    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")

    return llm


# ============================================================
# Agent 创建
# ============================================================
def create_agent(use_sqlite: bool = True, existing_conn: sqlite3.Connection | None = None):
    """
    创建并返回 Agent 实例和默认配置。
    从 llm_settings.json 读取当前 LLM 配置。
    """
    logger.info(f"开始创建 Agent，持久化模式: {'SQLite' if use_sqlite else '内存'}")

    settings = get_active_config()
    llm = create_llm(settings)

    global sqlite_connection
    if use_sqlite:
        if existing_conn is not None:
            sqlite_connection = existing_conn
        else:
            sqlite_connection = sqlite3.connect(
                CHECKPOINT_DB, check_same_thread=False, timeout=10
            )
            if WAL_CHECKPOINT_ON_START:
                try:
                    sqlite_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    sqlite_connection.commit()
                    logger.info("SQLite WAL checkpoint 已执行")
                except Exception as e:
                    logger.warning(f"WAL checkpoint 失败（非致命）: {e}")
        checkpointer = BoundedSqliteSaver(sqlite_connection, keep=MAX_CHECKPOINTS_PER_THREAD)
        logger.info(
            f"BoundedSqliteSaver 连接已建立: {CHECKPOINT_DB} "
            f"(keep={MAX_CHECKPOINTS_PER_THREAD}, writes=disabled)"
        )
    else:
        checkpointer = InMemorySaver()
        logger.info("使用内存记忆")

    agent = create_react_agent(
        model=llm,
        tools=[
            get_time,
            execute_command,
            read_file,
            write_file,
            list_processes,
            kill_process,
            list_services,
            network_check,
            system_stats,
            search_log,
            list_dir,
            search_kb,
            list_hosts,
            execute_on_host,
            run_on_all_hosts,
            check_host_health,
            excel_summary,
            excel_filter,
            excel_aggregate,
            excel_chart,
        ],
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": DEFAULT_THREAD_ID,
            "recursion_limit": MAX_AGENT_ITERATIONS,
            "max_tool_calls": MAX_TOOL_CALLS_PER_TURN,
        }
    }
    logger.info(f"Agent 创建成功 (recursion_limit={MAX_AGENT_ITERATIONS}, max_tool_calls={MAX_TOOL_CALLS_PER_TURN})")
    return agent, config


def create_llm_test(settings):
    """仅创建 LLM（不创建 Agent），用于连接测试"""
    return create_llm(settings)
