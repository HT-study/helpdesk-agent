from datetime import datetime
import json
import os
import sqlite3
import threading
from functools import wraps
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ── 可重试异常集合（覆盖 API 断连 / 超时 / 限速 / 服务端错误 / 通用网络错误） ──
try:
    from openai import (
        APIConnectionError, APIStatusError, RateLimitError,
        InternalServerError, APITimeoutError, APIError,
    )
    _RETRYABLE_EXCEPTIONS = (
        APIConnectionError, APIStatusError, RateLimitError,
        InternalServerError, APITimeoutError, APIError,
        TimeoutError, ConnectionError, ConnectionResetError,
        ConnectionAbortedError, BrokenPipeError, OSError,
    )
except ImportError:
    _RETRYABLE_EXCEPTIONS = (
        TimeoutError, ConnectionError, ConnectionResetError,
        ConnectionAbortedError, BrokenPipeError, OSError,
    )

# 最多重试次数
MAX_AGENT_RETRIES = int(os.environ.get("MAX_AGENT_RETRIES", "5"))

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_file
from flask_cors import CORS

from models.agent_model import create_agent, create_llm_test, get_sqlite_connection
from services.logger_service import get_logger
from services.audit_service import init_audit_db, get_audit_logs, clear_audit_logs
from services.kb_service import init_kb_db, list_kb, save_kb, delete_kb, search_kb as kb_search, search_semantic as kb_search_semantic, force_rebuild_index as kb_rebuild_index
from services.host_service import init_hosts_db, list_hosts, save_host, delete_host, execute_on_host, check_host_health as host_health
from config import (
    DEFAULT_THREAD_ID,
    CHECKPOINT_DB,
    API_TOKEN,
    LLM_SETTINGS_FILE,
    SUPPORTED_PROVIDERS,
    load_llm_settings,
    save_llm_settings,
    get_active_config,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_SAME_TOOL_REPEATS,
    MODEL_CONTEXT_WINDOWS,
    DEFAULT_CONTEXT_WINDOW,
    CONTEXT_TRIM_RATIO,
    CONTEXT_KEEP_RECENT_TURNS,
    CONTEXT_SUMMARY_ENABLED,
)

# ── 聊天文件上传目录 ──
CHAT_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_files")
os.makedirs(CHAT_UPLOAD_DIR, exist_ok=True)

logger = get_logger()

# ============================================================
# 延迟初始化 Agent：启动时不要求配置 API Key，
# 首次聊天或保存设置时才创建/重建 Agent
# ============================================================
_agent_state = {"agent": None, "config": None}

# ── 用户手动中断标志 ──
_stop_flag = {"active": False, "reason": ""}
_stop_lock = threading.Lock()


def request_stop(reason: str = "用户手动停止"):
    """请求停止当前任务"""
    with _stop_lock:
        _stop_flag["active"] = True
        _stop_flag["reason"] = reason
    logger.warning(f"🛑 收到停止请求: {reason}")


def check_stop() -> bool:
    """检查是否需要停止。返回 True 表示需要停止。"""
    with _stop_lock:
        return _stop_flag["active"]


def clear_stop():
    """清除停止标志"""
    with _stop_lock:
        _stop_flag["active"] = False
        _stop_flag["reason"] = ""


def get_stop_reason() -> str:
    """获取停止原因"""
    with _stop_lock:
        return _stop_flag["reason"]

# 启动时做一次独立的 SQLite WAL checkpoint（不依赖 Agent）
try:
    _conn = sqlite3.connect(CHECKPOINT_DB, timeout=10)
    _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    _conn.commit()
    _conn.close()
    logger.info("✅ 启动时 SQLite WAL checkpoint 完成")
except Exception as e:
    logger.warning(f"启动时 WAL checkpoint 失败（非致命）: {e}")

# 首次启动时压缩 checkpoints 库（禁用中间 writes 后，把历史 writes 与旧 checkpoints 一次清理）
# 后续每次启动都会跑，但因为 writes 已被 BoundedSqliteSaver 禁用，不会再膨胀；
# 裁剪 checkpoints 表到 CHECKPOINT_SIZE_MB 目标体积，并 VACUUM 回收磁盘。
try:
    from models.checkpoints import compact_checkpoints_db
    _compact_result = compact_checkpoints_db(vacuum=True)
    logger.info(f"💾 启动时 checkpoints 库压缩完成: {_compact_result}")
except Exception as e:
    logger.warning(f"启动时 checkpoints 库压缩失败（非致命，仅跳过瘦身）: {e}")

# 初始化审计日志表 / 知识库表 / 远程主机表
try:
    init_audit_db()
    init_kb_db()
    init_hosts_db()
except Exception as e:
    logger.error(f"初始化数据表失败: {e}", exc_info=True)


def get_agent():
    """获取 Agent 实例；尚未创建则延迟初始化。
    若 API Key / 模型未配置，抛出 ValueError，
    由调用方（聊天路由）转成友好错误响应。
    """
    if _agent_state["agent"] is None:
        logger.info("🚀 首次调用，正在初始化 Agent...")
        agent, config = create_agent(use_sqlite=True, existing_conn=get_sqlite_connection())
        _agent_state["agent"] = agent
        _agent_state["config"] = config
        logger.info("✅ Agent 初始化完成")
    return _agent_state["agent"], _agent_state["config"]


def recreate_agent():
    """重建 Agent（设置变更后调用）；失败时清空状态并抛出明确错误"""
    _agent_state["agent"] = None
    _agent_state["config"] = None
    try:
        logger.info("🔄 正在重建 Agent（设置已变更）...")
        agent, config = create_agent(use_sqlite=True, existing_conn=get_sqlite_connection())
        _agent_state["agent"] = agent
        _agent_state["config"] = config
        logger.info("✅ Agent 重建完成")
        return agent, config
    except Exception as e:
        logger.error(f"Agent 重建失败: {e}", exc_info=True)
        raise ValueError(f"Agent 重建失败: {e}") from e


# 启动预创建 Agent：若已配置 API Key 则提前初始化（避免首次请求等待 5-8 秒）
try:
    _pre = get_active_config()
    if _pre.get("api_key"):
        logger.info("🚀 检测到已配置 API Key，启动预创建 Agent...")
        get_agent()
except Exception as e:
    logger.warning(f"启动预创建 Agent 跳过（非致命）: {e}")

app = Flask(__name__)
CORS(app)


# ============================================================
# 鉴权
# ============================================================
def require_token(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not API_TOKEN:
            return view(*args, **kwargs)
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != API_TOKEN:
            logger.warning(f"拒绝未授权请求: {request.remote_addr}")
            return jsonify({"error": "未授权"}), 401
        return view(*args, **kwargs)

    return wrapper


# ============================================================
# 孤儿 tool_calls 修复 + 检查点断线恢复
# ============================================================
def _is_ai_message(msg):
    return hasattr(msg, "type") and msg.type == "ai"


def _get_tool_calls_from_msg(msg):
    """从 Message 中读取 tool_calls，兼容新旧两种序列化格式。"""
    if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls") is not None:
        return list(msg.tool_calls)
    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("tool_calls"):
        return list(msg.additional_kwargs["tool_calls"])
    return []


def _get_tool_call_id(tc):
    """从单个 tool_call 里取 id，兼容 dict / 对象两种形式。"""
    if isinstance(tc, dict):
        return tc.get("id") or tc.get("tool_call_id")
    return getattr(tc, "id", None) or getattr(tc, "tool_call_id", None)


def _get_msg_content(msg):
    """安全获取消息内容字符串"""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, (list, dict)):
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    return str(content) if content else ""


def _is_partial_ai_message(msg):
    """
    判断 AI 消息是否是不完整的（API 断线导致的部分生成消息）。
    特征：
    1. 有 tool_calls 但没有对应 ToolMessage（被 fix_orphan_tool_calls 检测）
    2. 有 tool_calls 但 tool_calls 为空列表（模型生成了空调用）
    3. content 为空但消息类型为 ai（模型生成了空回复）
    4. content 以 "..." 或 "正在" 等不完整结尾（模型正在生成时被中断）
    """
    if not _is_ai_message(msg):
        return False

    content = _get_msg_content(msg).strip()
    tool_calls = _get_tool_calls_from_msg(msg)

    # 有 tool_calls 的 AI 消息需要检查是否有对应的 ToolMessage
    # 这部分由 fix_orphan_tool_calls 负责检测
    # 这里只检测其他不完整模式

    # 空 content 的 AI 消息
    if not content and not tool_calls:
        return True

    # content 以不完整标记结尾（API 断线常见特征）
    incomplete_suffixes = ["...", "正在", "处理中", "正在思考", "正在分析"]
    if content and any(content.rstrip().endswith(sfx) for sfx in incomplete_suffixes):
        if not tool_calls:  # 有工具调用的消息可能只是还没完成
            return True

    return False


def _repair_checkpoint_interruption(agent, config):
    """
    外科手术式修复断线造成的检查点脏数据。

    与 _reset_checkpoint() 不同，此函数只移除断线产生的不完整消息，
    保留完整的对话历史（用户消息 + 已完成的 AI 回复）。

    处理场景：
    1. 有 tool_calls 但无对应 ToolMessage 的 AI 消息（孤立 tool_calls）
    2. 不完整的 AI 消息（content 以 "..." 结尾等）
    3. 连续的不完整 AI 消息链

    策略：
    - 从消息链尾部开始扫描
    - 跳过所有不完整消息（保留用户消息和完整 AI 消息）
    - 用清理后的消息列表更新检查点
    """
    try:
        state = agent.get_state(config)
        if not state or "messages" not in state.values:
            return False

        messages = list(state.values["messages"])
        if not messages:
            return False

        # 收集所有已完成的 ToolMessage 的 tool_call_id
        existing_tool_call_ids = set()
        for msg in messages:
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id:
                existing_tool_call_ids.add(tc_id)

        # 从消息链尾部开始，找到最后一个"安全"的消息索引
        # 安全消息 = 用户消息 / 完整 AI 消息 / 有效 ToolMessage
        dirty_from = len(messages)  # 默认全部安全

        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]

            # 用户消息永远安全
            if msg.type == "human":
                break  # 用户消息之前都是历史，安全

            # ToolMessage：检查是否有对应的 tool_call_id
            if msg.type == "tool":
                # ToolMessage 本身是安全的（只要它对应的 tool_call 存在）
                # 但如果没有对应的 tool_call，它也是脏数据
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id not in existing_tool_call_ids:
                    dirty_from = i
                    continue
                # 有效 ToolMessage 是安全的
                break

            # AI 消息：检查是否有孤立 tool_calls
            if _is_ai_message(msg):
                has_orphan = False
                for tc in _get_tool_calls_from_msg(msg):
                    tc_id = _get_tool_call_id(tc)
                    if tc_id and tc_id not in existing_tool_call_ids:
                        has_orphan = True
                        break

                is_partial = _is_partial_ai_message(msg)

                if has_orphan or is_partial:
                    dirty_from = i
                    continue
                # 完整 AI 消息是安全的
                break

        if dirty_from >= len(messages):
            return False  # 没有脏数据

        # 截断到脏数据之前
        clean_messages = messages[:dirty_from]

        if not clean_messages:
            # 如果截断后没有消息了，保留最后一条用户消息
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].type == "human":
                    clean_messages = [messages[i]]
                    break
            if not clean_messages:
                clean_messages = []

        if not clean_messages:
            return False  # 无法安全截断

        dirty_count = len(messages) - dirty_from

        # 更新检查点
        agent.update_state(config, {"messages": clean_messages})
        logger.warning(
            f"🔧 检查点修复：移除 {dirty_count} 条断线脏数据，保留 {len(clean_messages)} 条消息历史"
        )
        return True

    except Exception as e:
        logger.error(f"_repair_checkpoint_interruption 异常: {e}", exc_info=True)
        return False


# ============================================================
# 上下文窗口管理
# ============================================================
def _estimate_tokens(text: str) -> int:
    """粗估 token 数：中文约 1.5 字符/token，英文约 4 字符/token，混合取 ~2.5"""
    if not text:
        return 0
    # 简单启发式：非 ASCII 字符按 1.5 计，ASCII 按 4 计
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ascii_count = len(text) - non_ascii
    return int(non_ascii / 1.5 + ascii_count / 4) + 4  # +4 消息开销


def _estimate_msg_tokens(msg) -> int:
    """估算单条消息的 token 数"""
    base = 10  # 消息头开销
    content = _get_msg_content(msg)
    base += _estimate_tokens(content)
    # tool_calls 的参数也算
    for tc in _get_tool_calls_from_msg(msg):
        tc_input = ""
        if isinstance(tc, dict):
            tc_input = str(tc.get("args", "") or tc.get("input", ""))
        else:
            tc_input = str(getattr(tc, "args", "") or getattr(tc, "input", ""))
        base += _estimate_tokens(tc_input)
    return base


def _estimate_total_tokens(messages) -> int:
    """估算消息列表总 token 数"""
    return sum(_estimate_msg_tokens(m) for m in messages)


def _get_context_window(model: str) -> int:
    """根据模型名获取上下文窗口大小"""
    model_lower = (model or "").lower()
    for key, size in MODEL_CONTEXT_WINDOWS.items():
        if key.lower() in model_lower:
            return size
    return DEFAULT_CONTEXT_WINDOW


def _find_safe_trim_index(messages, min_keep: int) -> int:
    """
    在 messages 中找到安全的裁剪起点索引。
    保留 index >= trim_index 的消息。
    约束：
    - 不裁剪 SystemMessage（保留在最前）
    - 不切断 (AIMessage with tool_calls) ↔ (ToolMessage) 配对
    - 至少保留 min_keep 条消息
    """
    n = len(messages)
    if n <= min_keep:
        return n  # 全部保留

    start = max(min_keep, 1)  # 至少保留 1 条，且跳过 index 0（通常是 system）

    # 从 start 向后找，确保不在 tool 配对中间切割
    for i in range(start, n):
        msg = messages[i]
        msg_type = getattr(msg, "type", "")

        # 如果是 ToolMessage，它必须前面有对应的 AIMessage
        # 所以裁剪点不能在 ToolMessage 上（除非它前面有完整的 AIMessage）
        if msg_type == "tool":
            # 跳过，找下一个安全点
            continue

        # 如果是 AIMessage 且有 tool_calls，它后面的 ToolMessage 不能被裁掉
        if _is_ai_message(msg):
            tool_calls = _get_tool_calls_from_msg(msg)
            if tool_calls:
                # 检查后面紧跟着的 ToolMessage 是否完整
                # 如果保留此 AIMessage，也必须保留后面的所有 ToolMessage
                # 所以裁剪点应该在这个配对之前
                continue

        # 安全点：HumanMessage 或无 tool_calls 的 AIMessage
        if msg_type in ("human", "ai") and not _get_tool_calls_from_msg(msg):
            return i

    # 兜底：全部保留
    return n


def _build_summary_message(old_messages, llm=None) -> str:
    """
    用 LLM 将被裁剪的旧消息压缩为摘要文本。
    如果 llm 为 None 或摘要失败，返回简单的文本拼接摘要。
    """
    # 提取对话文本
    parts = []
    for msg in old_messages:
        msg_type = getattr(msg, "type", "")
        content = _get_msg_content(msg)
        if not content:
            # 跳过纯 tool_call 消息
            continue
        role = {"human": "用户", "ai": "助手", "system": "系统", "tool": "工具结果"}.get(msg_type, msg_type)
        parts.append(f"[{role}] {content[:500]}")

    if not parts:
        return ""

    dialog_text = "\n".join(parts[:40])  # 最多 40 条，防止太长

    # 尝试用 LLM 生成摘要
    if llm and CONTEXT_SUMMARY_ENABLED:
        try:
            prompt = (
                "请将以下对话历史压缩为简洁的摘要（不超过 300 字），"
                "保留关键信息、已做出的决策、重要的技术细节和结论：\n\n"
                f"{dialog_text}"
            )
            result = llm.invoke(prompt, max_tokens=400)
            summary = (result.content if hasattr(result, "content") else str(result)).strip()
            if summary:
                return summary
        except Exception as e:
            logger.warning(f"LLM 摘要失败，降级为文本摘要: {e}")

    # 降级：简单的文本截断摘要
    return "（对话历史摘要）\n" + dialog_text[:800]


def manage_context_window(agent, config, model: str = "") -> dict:
    """
    上下文窗口管理主入口。
    在 agent.stream() 之前调用，检测消息总量是否超限，
    超限时裁剪旧消息并生成摘要注入。

    返回: {
        "trimmed": bool,          # 是否做了裁剪
        "before_tokens": int,     # 裁剪前 token
        "after_tokens": int,      # 裁剪后 token
        "trimmed_count": int,     # 裁剪掉的消息数
        "summary_generated": bool,# 是否生成了摘要
    }
    """
    result = {"trimmed": False, "before_tokens": 0, "after_tokens": 0,
              "trimmed_count": 0, "summary_generated": False}

    try:
        state = agent.get_state(config)
        if not state or "messages" not in state.values:
            return result

        messages = list(state.values["messages"])
        if not messages:
            return result

        total_tokens = _estimate_total_tokens(messages)
        result["before_tokens"] = total_tokens

        window = _get_context_window(model)
        threshold = int(window * CONTEXT_TRIM_RATIO)

        if total_tokens < threshold:
            return result  # 未超限，无需裁剪

        logger.info(f"📏 上下文窗口管理：当前 ~{total_tokens} tokens，窗口 {window}，阈值 {threshold}，触发裁剪")

        # 计算保留消息数（最近 K 轮 ≈ K*2 条 + 可能的 tool 消息）
        min_keep = CONTEXT_KEEP_RECENT_TURNS * 3  # 每轮 ~3 条（用户+AI+可能工具）

        # 找安全裁剪点
        trim_idx = _find_safe_trim_index(messages, min_keep)

        if trim_idx >= len(messages):
            # 无法安全裁剪（消息太少或全部是配对）
            logger.warning(f"无法安全裁剪：min_keep={min_keep}, total={len(messages)}")
            return result

        # 分离系统消息（通常在 index 0）
        system_messages = []
        dialog_messages = []
        for i, msg in enumerate(messages):
            if getattr(msg, "type", "") == "system" and i == 0:
                system_messages.append(msg)
            else:
                dialog_messages.append(msg)

        # 在 dialog_messages 中重新找安全裁剪点
        # dialog_messages 不含 system，所以 min_keep 不变
        dialog_trim_idx = _find_safe_trim_index(dialog_messages, min_keep)

        if dialog_trim_idx >= len(dialog_messages):
            logger.warning("对话消息无法安全裁剪")
            return result

        old_messages = dialog_messages[:dialog_trim_idx]
        kept_messages = dialog_messages[dialog_trim_idx:]

        if not kept_messages:
            return result

        # 生成摘要
        summary_text = ""
        summary_msg = None
        if old_messages:
            try:
                llm = None
                if CONTEXT_SUMMARY_ENABLED:
                    active = get_active_config()
                    llm = create_llm_test(active)
                summary_text = _build_summary_message(old_messages, llm)
                if summary_text:
                    from langchain_core.messages import SystemMessage
                    summary_msg = SystemMessage(content=f"[历史对话摘要]\n{summary_text}")
                    result["summary_generated"] = True
            except Exception as e:
                logger.warning(f"摘要生成失败: {e}")

        # 组装新消息列表
        new_messages = list(system_messages)
        if summary_msg:
            new_messages.append(summary_msg)
        new_messages.extend(kept_messages)

        # 更新 checkpoint
        agent.update_state(config, {"messages": new_messages})

        after_tokens = _estimate_total_tokens(new_messages)
        result["after_tokens"] = after_tokens
        result["trimmed"] = True
        result["trimmed_count"] = len(old_messages)

        logger.info(
            f"✂️ 上下文裁剪完成：{len(messages)} → {len(new_messages)} 条消息，"
            f"~{total_tokens} → ~{after_tokens} tokens（摘要: {'是' if result['summary_generated'] else '否'}）"
        )

    except Exception as e:
        logger.error(f"上下文窗口管理异常: {e}", exc_info=True)

    return result


def _reset_checkpoint(thread_id: str) -> None:
    """兜底方案：直接删除该 thread 的检查点，恢复到干净状态。
    仅在外科手术修复失败时使用。"""
    conn = sqlite3.connect(CHECKPOINT_DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()


def fix_orphan_tool_calls(agent, config):
    """
    检测"孤立 tool_calls"（有调用请求、无对应 ToolMessage 结果）并修复。

    上一轮对话中断（异常/超时/断网）会导致 checkpoint 残留这类脏数据，
    LangGraph 的 _validate_chat_history 会在下一轮直接拒绝执行。

    策略（已升级）：
    - 优先尝试外科手术式修复（_repair_checkpoint_interruption），保留对话历史
    - 修复失败时兜底重置整个 thread
    """
    try:
        state = agent.get_state(config)
        if not state or "messages" not in state.values:
            return False

        messages = state.values["messages"]
        if not messages:
            return False

        # 收集所有已完成的 ToolMessage 的 tool_call_id
        existing_tool_call_ids = set()
        for msg in messages:
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id:
                existing_tool_call_ids.add(tc_id)

        # 遍历所有 AIMessage，找孤立 tool_calls
        orphan_count = 0
        for msg in messages:
            if not _is_ai_message(msg):
                continue
            for tc in _get_tool_calls_from_msg(msg):
                tc_id = _get_tool_call_id(tc)
                if tc_id and tc_id not in existing_tool_call_ids:
                    orphan_count += 1

        if orphan_count == 0:
            # 没有孤立 tool_calls，但仍可能有不完整的 AI 消息（无 tool_calls 但 content 不完整）
            has_partial = any(
                _is_partial_ai_message(msg) for msg in messages if _is_ai_message(msg)
            )
            if not has_partial:
                return False

        # ── 优先尝试外科手术式修复（保留对话历史）──
        if _repair_checkpoint_interruption(agent, config):
            logger.warning(
                f"🔧 检测到 {orphan_count} 个孤立 tool_calls，已修复（保留对话历史）"
            )
            return True

        # ── 兜底：全量重置（极端情况）──
        thread_id = config.get("configurable", {}).get("thread_id", DEFAULT_THREAD_ID)
        _reset_checkpoint(thread_id)
        logger.error(f"⚠️ 外科手术修复失败，已全量重置会话 (thread_id={thread_id})")
        return True

    except Exception as e:
        logger.error(f"fix_orphan_tool_calls 异常: {e}", exc_info=True)
        return False


def _coerce_str(v) -> str:
    """把 SSE 事件字段统一转成字符串，避免前端 escapeHtml 收到对象崩溃。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)


def _extract_usage(msg) -> dict | None:
    """从 AIMessage 中提取 token 用量：输入/输出/缓存命中/未命中。
    兼容新版 usage_metadata 与旧版 response_metadata.token_usage。
    """
    try:
        usage = None
        # 新版 langchain-core: msg.usage_metadata
        um = getattr(msg, "usage_metadata", None)
        if um:
            usage = {
                "input": int(um.get("input_tokens") or 0),
                "output": int(um.get("output_tokens") or 0),
                "total": int(um.get("total_tokens") or 0),
                "cache_hit": int((um.get("input_token_details") or {}).get("cache_read") or 0),
                "cache_miss": int((um.get("input_token_details") or {}).get("cache_creation") or 0),
            }
        # 旧版: msg.response_metadata.token_usage（DeepSeek 风格）
        rm = getattr(msg, "response_metadata", None) or {}
        tu = rm.get("token_usage") if isinstance(rm, dict) else None
        if not usage and tu:
            usage = {
                "input": int(tu.get("prompt_tokens") or 0),
                "output": int(tu.get("completion_tokens") or 0),
                "total": int(tu.get("total_tokens") or 0),
                "cache_hit": int(tu.get("prompt_cache_hit_tokens") or 0),
                "cache_miss": int(tu.get("prompt_cache_miss_tokens") or 0),
            }
        if not usage or usage["total"] == 0:
            return None
        return usage
    except Exception:
        return None


def _build_user_message(text: str, images: list | None):
    """构造用户消息。有图片时返回 LangChain HumanMessage（带 image_url），
    无图片时返回普通 dict（兼容原有逻辑）。
    非 vision 模型（DeepSeek / openai_compatible）会忽略图片并降级为纯文本。
    """
    if not images:
        return {"role": "user", "content": text}

    # 检查当前模型是否支持图片识别
    settings = get_active_config()
    provider = (settings.get("provider") or "").lower()
    model = (settings.get("model") or "").lower()

    # 支持 vision 的模型列表
    vision_providers = {"openai", "anthropic"}
    vision_models = [
        "gpt-4o", "gpt-4-turbo", "gpt-4-vision",
        "claude-3", "claude-3-5", "claude-sonnet", "claude-opus",
        "qwen-vl", "gemini", "vision",
    ]
    is_vision = provider in vision_providers or any(
        k in model for k in vision_models
    )

    if not is_vision:
        logger.warning(
            f"⚠️ 当前模型 {model} ({provider}) 不支持图片识别，已忽略图片内容"
        )
        return {"role": "user", "content": text or "请分析这张图片"}

    from langchain_core.messages import HumanMessage

    content = [{"type": "text", "text": text or "请分析这张图片"}]
    for img in images:
        if isinstance(img, str) and img.startswith("data:image"):
            content.append({"type": "image_url", "image_url": {"url": img}})
    return HumanMessage(content=content)


def _repair_checkpoint_vision(agent, config):
    """修复检查点中已存在的 image_url 消息，将其转为纯文本。
    非 vision 模型加载含 image_url 的历史消息时会报 400 错误。"""
    try:
        state = agent.get_state(config)
        if not state or "messages" not in state.values:
            return False
        messages = list(state.values["messages"])
        repaired = False
        for i, msg in enumerate(messages):
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                # 提取纯文本部分，移除 image_url
                text_parts = [
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                new_text = " ".join(filter(None, text_parts)).strip()
                msg.content = new_text or "[图片已忽略：当前模型不支持图片识别]"
                repaired = True
            elif isinstance(content, dict):
                # OpenAI 兼容格式的部分序列化
                msg.content = str(content.get("text", "") or content.get("content", ""))
                repaired = True
        if repaired:
            # 用 update_state 将修复后的消息写回检查点
            agent.update_state(config, {"messages": messages})
            logger.info("✅ 已修复检查点中不兼容的图片消息")
            return True
        return False
    except Exception as e:
        logger.warning(f"检查点修复跳过（非致命）: {e}")
        return False


# ============================================================
# 路由：首页
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# 路由：设置页面
# ============================================================
@app.route("/settings")
def settings_page():
    return render_template("settings.html")


@app.route("/settings/providers", methods=["GET"])
def settings_providers():
    """返回支持的 LLM 提供商列表"""
    return jsonify({"providers": SUPPORTED_PROVIDERS})


@app.route("/settings/load", methods=["GET"])
def settings_load():
    """返回当前 LLM 设置（多配置格式，兼容旧前端）"""
    all_settings = load_llm_settings()
    configs = all_settings.get("configs", [])
    active = None
    for c in configs:
        if c.get("active"):
            active = c
            break
    if not active and configs:
        active = configs[0]

    # 脱敏 API Key
    masked_configs = []
    for c in configs:
        mc = dict(c)
        key = mc.get("api_key", "")
        if key:
            mc["api_key"] = key[:4] + "****" + key[-4:] if len(key) > 8 else key[:2] + "****"
        mc["has_key"] = bool(c.get("api_key", ""))
        masked_configs.append(mc)

    # 返回多配置列表（新前端）
    result = {
        "configs": masked_configs,
        "active_config": dict(active) if active else {},
    }

    # 兼容旧前端：也返回旧格式字段
    if active:
        key = active.get("api_key", "")
        result["provider"] = active.get("provider", "")
        result["api_key"] = key[:4] + "****" + key[-4:] if len(key) > 8 else key[:2] + "****"
        result["base_url"] = active.get("base_url", "")
        result["model"] = active.get("model", "")
        result["temperature"] = active.get("temperature", 0.5)
        result["has_key"] = bool(key)

    result["settings_file"] = LLM_SETTINGS_FILE
    return jsonify(result)


@app.route("/settings/configs", methods=["GET"])
def settings_configs_list():
    """返回所有配置列表（脱敏）"""
    all_settings = load_llm_settings()
    configs = all_settings.get("configs", [])
    masked = []
    for c in configs:
        mc = dict(c)
        key = mc.get("api_key", "")
        if key:
            mc["api_key"] = key[:4] + "****" + key[-4:] if len(key) > 8 else key[:2] + "****"
        mc["has_key"] = bool(c.get("api_key", ""))
        masked.append(mc)
    return jsonify({"configs": masked})


@app.route("/settings/configs", methods=["POST"])
def settings_configs_create():
    """新增一条 API 配置"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "无效的 JSON"}), 400

        provider = data.get("provider", "").strip()
        if provider not in SUPPORTED_PROVIDERS:
            return jsonify({"error": f"不支持的 LLM 提供商: {provider}"}), 400

        api_key = data.get("api_key", "").strip()
        if not api_key:
            return jsonify({"error": "API Key 不能为空"}), 400

        model = data.get("model", "").strip()
        if not model:
            return jsonify({"error": "模型名称不能为空"}), 400

        name = (data.get("name") or "").strip() or f"{provider}-{model}"
        base_url = data.get("base_url", "").strip()
        temperature = float(data.get("temperature", 0.5))

        new_config = {
            "id": f"cfg_{int(time.time() * 1000)}",
            "name": name,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "active": False,
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }

        all_settings = load_llm_settings()
        all_settings.setdefault("configs", []).append(new_config)
        save_llm_settings(all_settings)

        logger.info(f"📝 新增 API 配置: {name} ({provider}/{model})")
        return jsonify({"status": "ok", "config": {k: v for k, v in new_config.items() if k != "api_key"}})
    except Exception as e:
        logger.error(f"新增配置失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route("/settings/configs/<config_id>", methods=["PUT"])
def settings_configs_update(config_id):
    """更新一条 API 配置（编辑）"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "无效的 JSON"}), 400

        all_settings = load_llm_settings()
        configs = all_settings.get("configs", [])
        target = None
        for c in configs:
            if c.get("id") == config_id:
                target = c
                break

        if not target:
            return jsonify({"error": f"配置不存在: {config_id}"}), 404

        provider = data.get("provider", target.get("provider", ""))
        if provider not in SUPPORTED_PROVIDERS:
            return jsonify({"error": f"不支持的 LLM 提供商: {provider}"}), 400

        api_key = data.get("api_key", "").strip()
        # 空 key 保留旧值
        if not api_key:
            api_key = target.get("api_key", "")

        model = data.get("model", target.get("model", "")).strip()
        if not model:
            return jsonify({"error": "模型名称不能为空"}), 400

        name = (data.get("name") or target.get("name", "")).strip() or target.get("name", "")
        base_url = data.get("base_url", target.get("base_url", "")).strip()
        temperature = float(data.get("temperature", target.get("temperature", 0.5)))

        target.update({
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "name": name,
        })

        save_llm_settings(all_settings)
        logger.info(f"✏️ 更新 API 配置: {name} ({config_id})")
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"更新配置失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route("/settings/configs/<config_id>", methods=["DELETE"])
def settings_configs_delete(config_id):
    """删除一条 API 配置"""
    try:
        all_settings = load_llm_settings()
        configs = all_settings.get("configs", [])

        # 至少保留一条
        if len(configs) <= 1:
            return jsonify({"error": "至少需要保留一条配置"}), 400

        target = None
        for c in configs:
            if c.get("id") == config_id:
                target = c
                break

        if not target:
            return jsonify({"error": f"配置不存在: {config_id}"}), 404

        # 如果删除的是 active 配置，先将其他配置设为 active
        if target.get("active"):
            for c in configs:
                if c.get("id") != config_id:
                    c["active"] = True
                    break

        all_settings["configs"] = [c for c in configs if c.get("id") != config_id]
        save_llm_settings(all_settings)
        logger.info(f"🗑️ 删除 API 配置: {target.get('name', config_id)}")
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"删除配置失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route("/settings/configs/<config_id>/activate", methods=["PATCH"])
def settings_configs_activate(config_id):
    """激活（切换）一条 API 配置"""
    try:
        all_settings = load_llm_settings()
        configs = all_settings.get("configs", [])
        found = False

        for c in configs:
            if c.get("id") == config_id:
                c["active"] = True
                found = True
            else:
                c["active"] = False

        if not found:
            return jsonify({"error": f"配置不存在: {config_id}"}), 404

        save_llm_settings(all_settings)
        # 重建 Agent 使用新配置
        recreate_agent()

        logger.info(f"🔀 激活 API 配置: {config_id}")
        return jsonify({"status": "ok", "message": "配置已激活，Agent 已重建"})
    except Exception as e:
        logger.error(f"激活配置失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route("/settings/save", methods=["POST"])
def settings_save():
    """保存当前编辑的配置（向后兼容旧前端）"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "无效的 JSON"}), 400

        provider = data.get("provider", "")
        if provider not in SUPPORTED_PROVIDERS:
            return jsonify({"error": f"不支持的 LLM 提供商: {provider}"}), 400

        api_key = data.get("api_key", "").strip()
        # 如果 api_key 为空或掩码值，从当前激活配置保留旧值
        if not api_key:
            active = get_active_config()
            api_key = active.get("api_key", "")

        model = data.get("model", "").strip()
        base_url = data.get("base_url", "").strip()
        temperature = float(data.get("temperature", 0.5))

        if not model:
            return jsonify({"error": "模型名称不能为空"}), 400
        if not api_key:
            return jsonify({"error": "API Key 不能为空，请填写后保存（首次配置必须填写）"}), 400

        # 获取当前激活的配置并更新
        all_settings = load_llm_settings()
        configs = all_settings.get("configs", [])
        target = None
        for c in configs:
            if c.get("active"):
                target = c
                break

        if not target and configs:
            target = configs[0]

        if not target:
            return jsonify({"error": "无可用配置"}), 400

        target.update({
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
        })

        save_llm_settings(all_settings)
        recreate_agent()

        return jsonify({"status": "ok", "message": "设置已保存，Agent 已重建"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"保存设置失败: {str(e)}", exc_info=True)
        return jsonify({"error": f"保存失败: {str(e)}"}), 500


@app.route("/settings/test", methods=["POST"])
def settings_test():
    """测试 LLM 连接：发一个极小的测试请求"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "无效的 JSON"}), 400

        provider = data.get("provider", "")
        if provider not in SUPPORTED_PROVIDERS:
            return jsonify({"error": f"不支持的提供商: {provider}"}), 400

        api_key = data.get("api_key", "").strip()
        config_id = data.get("config_id", "")

        # 前端没传 key 时按优先级回退：
        # 1. 指定 config_id 的配置
        # 2. 匹配 provider 的配置（含新建的未激活配置）
        # 3. 当前激活配置
        if not api_key:
            all_configs = load_llm_settings().get("configs", [])
            # 优先：指定 config_id
            if config_id:
                for c in all_configs:
                    if c.get("id") == config_id and c.get("api_key"):
                        api_key = c["api_key"]
                        break
            # 其次：匹配 provider
            if not api_key:
                for c in all_configs:
                    if c.get("provider") == provider and c.get("api_key"):
                        api_key = c["api_key"]
                        break
            # 最后：激活配置
            if not api_key:
                api_key = get_active_config().get("api_key", "")

        if not api_key:
            return jsonify({"error": "API Key 不能为空，请先保存配置后再测试"}), 400

        model = data.get("model", "").strip()
        base_url = data.get("base_url", "").strip()
        temperature = float(data.get("temperature", 0.5))

        settings = {
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
        }

        llm = create_llm_test(settings)

        # 极简测试：只让模型输出一个字符
        result = llm.invoke("回复一个单词：OK", max_tokens=5)
        content = result.content if hasattr(result, "content") else str(result)
        return jsonify({
            "status": "ok",
            "message": f"连接成功 ✅ — {provider}/{model}",
            "test_reply": content,
        })

    except Exception as e:
        err_msg = str(e)
        # 识别认证错误，给出更明确的提示
        if "401" in err_msg or "Authentication" in err_msg or "auth" in err_msg.lower() or "invalid" in err_msg.lower():
            logger.warning(f"LLM 连接测试：API Key 无效 — {provider}/{model}")
            return jsonify({
                "status": "error",
                "message": f"API Key 无效，请检查后重新输入（当前配置: {provider}/{model}）",
            }), 400
        logger.error(f"LLM 连接测试失败: {err_msg}", exc_info=True)
        return jsonify({"status": "error", "message": f"连接失败: {err_msg}"}), 500


# ============================================================
# 路由：审计日志
# ============================================================
@app.route("/audit")
def audit_page():
    return render_template("audit.html")


@app.route("/audit/list", methods=["GET"])
def audit_list():
    limit = min(int(request.args.get("limit", 100)), 500)
    tool_name = request.args.get("tool_name", "") or None
    return jsonify({"logs": get_audit_logs(limit=limit, tool_name=tool_name)})


@require_token
@app.route("/audit/clear", methods=["POST"])
def audit_clear():
    clear_audit_logs()
    return jsonify({"status": "ok", "message": "审计日志已清空"})


# ============================================================
# 路由：知识库
# ============================================================
# 知识库智能沉淀：AI 提取 Q&A
# ============================================================
_EXTRACT_QA_PROMPT = """你是一个知识库助手。请从以下对话中提取一条"问题-解决方案"对。

用户问题：{user_question}
Agent 回复：{agent_answer}

要求：
1. "问题" 字段：用简洁的一句话概括用户的问题（10-50 字），不要包含具体命令或路径
2. "解决方案" 字段：整理 Agent 回复中的核心解决方案步骤（最多 500 字），使用 Markdown 格式
3. "标签" 字段：用逗号分隔的中文标签（1-3 个），描述该知识条目的主题分类

输出 JSON 格式（不要加 ``` 标记）：
{{"question": "...", "solution": "...", "tags": "..."}}
"""


def _extract_qa_suggestion(user_input: str, agent_response: str) -> dict | None:
    """
    使用 LLM 从一次完整的问答中提取知识库条目。
    返回 {"question": str, "solution": str, "tags": str} 或 None。
    异常时返回 None（不阻塞主流程）。
    """
    try:
        settings = get_active_config()
        llm = create_llm_test(settings)
        prompt = _EXTRACT_QA_PROMPT.format(
            user_question=user_input[:300],
            agent_answer=agent_response[:1000],
        )
        result = llm.invoke(prompt, max_tokens=400)
        content = result.content if hasattr(result, "content") else str(result)

        # 尝试解析 JSON
        import re as _re
        content = content.strip()
        # 移除可能的代码块标记
        content = _re.sub(r"^```json\s*", "", content)
        content = _re.sub(r"```$", "", content)
        content = _re.sub(r"^```\s*", "", content)
        data = json.loads(content)

        question = (data.get("question") or "").strip()
        solution = (data.get("solution") or "").strip()
        tags = (data.get("tags") or "").strip()

        if question and solution:
            logger.info(f"✅ AI 提取到知识条目: {question[:40]}...")
            return {"question": question, "solution": solution, "tags": tags}
    except Exception as e:
        logger.debug(f"AI 知识提取失败（非致命）: {e}")
    return None


# ============================================================
# 路由：知识库智能沉淀 — 保存建议
# ============================================================
@app.route("/kb/save-suggestion", methods=["POST"])
def kb_save_suggestion():
    """
    用户确认 AI 提取的知识条目，保存到知识库。
    安全功能，不要求 Token 鉴权（用户主动操作）。
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "无效的 JSON"}), 400

    question = (data.get("question") or "").strip()
    solution = (data.get("solution") or "").strip()
    tags = (data.get("tags") or "").strip()

    if not question or not solution:
        return jsonify({"error": "问题和解决方案不能为空"}), 400

    save_kb(question, solution, tags)
    logger.info(f"✅ 用户确认保存知识条目: {question[:40]}...")
    return jsonify({"status": "ok", "message": "已保存到知识库"})


# ============================================================
# 路由：知识库
# ============================================================
@app.route("/kb")
def kb_page():
    return render_template("kb.html")


@app.route("/kb/list", methods=["GET"])
def kb_list():
    return jsonify({"items": list_kb()})


@require_token
@app.route("/kb/save", methods=["POST"])
def kb_save():
    data = request.get_json()
    if not data:
        return jsonify({"error": "无效的 JSON"}), 400
    question = (data.get("question") or "").strip()
    solution = (data.get("solution") or "").strip()
    tags = (data.get("tags") or "").strip()
    if not question or not solution:
        return jsonify({"error": "问题和解决方案不能为空"}), 400
    save_kb(question, solution, tags)
    return jsonify({"status": "ok", "message": "已保存到知识库"})


@require_token
@app.route("/kb/delete", methods=["POST"])
def kb_delete():
    data = request.get_json()
    if not data or not data.get("id"):
        return jsonify({"error": "缺少 id"}), 400
    delete_kb(int(data["id"]))
    return jsonify({"status": "ok", "message": "已删除"})


@app.route("/kb/search", methods=["GET"])
def kb_search_route():
    """
    知识库搜索。支持两种模式：
    - mode=semantic（语义搜索，默认）：基于 TF-IDF + 余弦相似度
    - mode=keyword（关键词搜索）：SQL LIKE 模糊匹配
    """
    query = request.args.get("query", "").strip()
    mode = request.args.get("mode", "semantic")  # semantic | keyword
    limit = min(int(request.args.get("limit", 10)), 50)

    if mode == "keyword":
        results = kb_search(query, limit=limit)
    else:
        results = kb_search_semantic(query, limit=limit)

    return jsonify({
        "results": results,
        "mode": mode,
        "query": query,
        "count": len(results),
    })


@app.route("/kb/index/rebuild", methods=["POST"])
def kb_rebuild_endpoint():
    """手动触发语义索引重建"""
    try:
        idx = kb_rebuild_index()
        return jsonify({
            "status": "ok",
            "message": f"语义索引已重建: {idx.count} 条文档, 词汇表 {idx.vocab.size}",
            "doc_count": idx.count,
            "vocab_size": idx.vocab.size,
        })
    except Exception as e:
        logger.error(f"语义索引重建失败: {e}", exc_info=True)
        return jsonify({"error": f"重建失败: {e}"}), 500


@require_token
@app.route("/kb/import", methods=["POST"])
def kb_import():
    """批量导入知识库条目（支持 JSON/CSV/TXT/PDF/DOCX）"""
    import csv
    import io
    import tempfile

    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    filename = file.filename.lower()
    raw = file.read()
    entries = []

    try:
        # ---- 纯文本格式（JSON / CSV / TXT）----
        if filename.endswith(".json"):
            text = raw.decode("utf-8", errors="replace")
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    q = (item.get("question") or item.get("q") or "").strip()
                    s = (item.get("solution") or item.get("s") or item.get("answer") or item.get("a") or "").strip()
                    t = (item.get("tags") or item.get("tag") or "").strip()
                    if q and s:
                        entries.append((q, s, t))
            elif isinstance(data, dict):
                for q, s in data.items():
                    if isinstance(s, str) and q.strip() and s.strip():
                        entries.append((q.strip(), s.strip(), ""))
        elif filename.endswith(".csv"):
            text = raw.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                q = (row.get("question") or row.get("q") or "").strip()
                s = (row.get("solution") or row.get("s") or row.get("answer") or row.get("a") or "").strip()
                t = (row.get("tags") or row.get("tag") or "").strip()
                if q and s:
                    entries.append((q, s, t))
        elif filename.endswith(".txt"):
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            q, s_lines = "", []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("Q:") or stripped.startswith("问题：") or stripped.startswith("问："):
                    if q and s_lines:
                        entries.append((q.strip(), "\n".join(s_lines).strip(), ""))
                    q = stripped[2:].strip()
                    s_lines = []
                elif stripped.startswith("A:") or stripped.startswith("答案：") or stripped.startswith("答："):
                    s_lines.append(stripped[2:].strip())
                elif q:
                    s_lines.append(stripped)
            if q and s_lines:
                entries.append((q.strip(), "\n".join(s_lines).strip(), ""))

        # ---- 文档格式（PDF / DOCX）：智能解析 ----
        elif filename.endswith(".docx") or filename.endswith(".pdf"):
            # 写入临时文件后交给 document_parser 处理
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=filename, prefix="kb_import_")
            try:
                with os.fdopen(tmp_fd, "wb") as tmp_f:
                    tmp_f.write(raw)
                from services.document_parser import parse_document_to_entries
                entries = parse_document_to_entries(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        else:
            return jsonify({
                "error": f"不支持的文件格式：{filename}，支持 .json / .csv / .txt / .pdf / .docx"
            }), 400
    except Exception as e:
        logger.error(f"知识库导入解析失败: {e}", exc_info=True)
        return jsonify({"error": f"解析失败: {e}"}), 400

    if not entries:
        return jsonify({"error": "未解析到有效条目"}), 400

    count = 0
    for q, s, t in entries:
        save_kb(q, s, t)
        count += 1
    logger.info(f"批量导入知识库 {count} 条（格式: {filename.split('.')[-1]}）")
    return jsonify({"status": "ok", "message": f"成功导入 {count} 条知识条目", "count": count})


# ============================================================
# 路由：远程主机管理
# ============================================================
@app.route("/hosts")
def hosts_page():
    return render_template("hosts.html")


@app.route("/hosts/list", methods=["GET"])
def hosts_list():
    return jsonify({"hosts": list_hosts()})


@require_token
@app.route("/hosts/save", methods=["POST"])
def hosts_save():
    data = request.get_json()
    if not data:
        return jsonify({"error": "无效的 JSON"}), 400
    label = (data.get("label") or "").strip()
    hostname = (data.get("hostname") or "").strip()
    port = int(data.get("port", 22))
    username = (data.get("username") or "").strip()
    auth_type = data.get("auth_type", "password")
    credential = data.get("credential", "").strip()
    if not label or not hostname or not username or not credential:
        return jsonify({"error": "标签、主机、用户名、认证凭据不能为空"}), 400
    save_host(label, hostname, port, username, auth_type, credential)
    return jsonify({"status": "ok", "message": "已保存"})


@require_token
@app.route("/hosts/delete", methods=["POST"])
def hosts_delete():
    data = request.get_json()
    if not data or not data.get("id"):
        return jsonify({"error": "缺少 id"}), 400
    delete_host(int(data["id"]))
    return jsonify({"status": "ok", "message": "已删除"})


@require_token
@app.route("/hosts/test", methods=["POST"])
def hosts_test():
    data = request.get_json()
    if not data or not data.get("id"):
        return jsonify({"error": "缺少 id"}), 400
    hosts = list_hosts()
    target = next((h for h in hosts if h["id"] == data["id"]), None)
    if not target:
        return jsonify({"status": "error", "message": "未找到该主机"}), 404
    ok, msg = host_health(target["label"])
    return jsonify({"status": "ok", "success": ok, "message": msg})


# ============================================================
# 路由：图表文件服务
# ============================================================
@app.route("/file/chart/<filename>")
def serve_chart(filename):
    from config import CHART_DIR
    import os as _os
    safe = _os.path.basename(filename)
    path = _os.path.join(CHART_DIR, safe)
    if not _os.path.exists(path):
        return jsonify({"error": "文件不存在"}), 404
    return _send_file(path, mimetype="image/png")


# ============================================================
# 路由：聊天
# ============================================================
@retry(
    stop=stop_after_attempt(MAX_AGENT_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
)
def _invoke_agent(agent, payload, config):
    return agent.invoke(payload, config=config)


def _is_retryable_error(exc: BaseException) -> bool:
    """判断异常是否适合自动重试（API 断连类问题）"""
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    # 兜底：通过异常类型名和消息匹配常见网络 / API 错误
    exc_name = type(exc).__name__.lower()
    exc_msg = str(exc).lower()
    retry_keywords = [
        "connection", "timeout", "timed out", "reset", "aborted",
        "rate limit", "rate_limit", "429", "500", "502", "503", "504",
        "internal server error", "service unavailable",
        "broken pipe", "errno", "temporary failure",
        "api unavailable", "llm_request_failed", "server error",
    ]
    return any(kw in exc_name or kw in exc_msg for kw in retry_keywords)


# ============================================================
# 循环检测器
# ============================================================
class LoopDetector:
    """
    检测 Agent 工具调用中的循环模式：
    1. 总工具调用次数超过上限 → 循环
    2. 相同 (工具名, 参数) 组合重复超过上限 → 循环
    3. 相同工具连续调用超过上限（参数不同也算）→ 循环
    """

    def __init__(self):
        self.total_calls = 0
        self.tool_call_history = []        # list of (tool_name, tool_input_str)
        self.recent_calls = []             # last N calls (for consecutive check)
        self.max_tool_calls = MAX_TOOL_CALLS_PER_TURN
        self.max_same_repeats = MAX_SAME_TOOL_REPEATS
        self._loop_msg = ""

    def record_call(self, tool_name: str, tool_input: str) -> tuple[bool, str]:
        """记录一次工具调用。返回 (is_loop, msg)。"""
        self.total_calls += 1

        # 工具 1：总调用次数超限
        if self.total_calls > self.max_tool_calls:
            return True, (
                f"检测到工具调用循环：单轮已执行 {self.total_calls} 次工具调用"
                f"（上限 {self.max_tool_calls}），任务将终止以避免无限循环。"
            )

        # 工具 2：相同 (工具名, 参数) 重复调用
        key = (tool_name, tool_input)
        self.tool_call_history.append(key)

        # 统计该 key 出现次数
        same_count = sum(1 for k in self.tool_call_history if k[0] == tool_name and k[1] == tool_input)
        if same_count > self.max_same_repeats:
            return True, (
                f"检测到循环：工具 `{tool_name}` 以相同参数重复调用了 {same_count} 次"
                f"（上限 {self.max_same_repeats}），任务将终止以避免无限循环。"
            )

        # 工具 3：相同工具连续调用（参数不同也算，用于检测死循环）
        self.recent_calls.append(tool_name)
        # 保留最近 20 次调用
        if len(self.recent_calls) > 20:
            self.recent_calls = self.recent_calls[-20:]

        # 检查最近 N 次是否全部是同一工具
        for window_size in range(4, min(len(self.recent_calls) + 1, 10)):
            window = self.recent_calls[-window_size:]
            if len(set(window)) == 1:
                tool = window[0]
                return True, (
                    f"检测到循环：工具 `{tool}` 连续调用了 {window_size} 次，"
                    f"任务将终止以避免无限循环。"
                )

        return False, ""

    def reset(self):
        self.total_calls = 0
        self.tool_call_history.clear()
        self.recent_calls.clear()


# ============================================================
# 路由：文件上传
# ============================================================
@app.route("/chat/upload-file", methods=["POST"])
def chat_upload_file():
    """
    聊天文件上传端点。上传文件到 chat_files 目录，
    返回文件路径供聊天消息引用。

    最大文件大小：20 MB
    支持任意文件类型（由 Agent 的 read_file 工具处理）
    """
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 安全检查：移除路径分隔符，防止目录穿越
    safe_name = os.path.basename(file.filename)
    # 过滤掉路径分隔符和其他不安全字符
    safe_name = safe_name.replace("\\", "_").replace("/", "_").replace("..", "_")
    if not safe_name or safe_name == ".txt":
        safe_name = "uploaded_file.txt"

    ext = os.path.splitext(safe_name)[1].lower() or ".txt"
    filename = f"{int(time.time() * 1000)}_{safe_name}"
    filepath = os.path.join(CHAT_UPLOAD_DIR, filename)

    # 文件大小限制：20 MB
    MAX_CHAT_FILE_SIZE = 20 * 1024 * 1024

    try:
        # 直接读取整个文件（比 iter_chunks 兼容性好）
        content = file.read()
        file_size = len(content)

        if file_size > MAX_CHAT_FILE_SIZE:
            return jsonify({"error": f"文件过大：{file_size // 1024 // 1024} MB，超过上限 {MAX_CHAT_FILE_SIZE // 1024 // 1024} MB"}), 400

        with open(filepath, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"文件上传失败: {e}", exc_info=True)
        return jsonify({"error": f"文件写入失败: {e}"}), 500

    logger.info(f"📎 聊天文件上传: {safe_name} ({file_size} bytes)")
    return jsonify({
        "status": "ok",
        "file_path": filepath,
        "file_name": safe_name,
        "file_size": file_size,
        "file_size_mb": round(file_size / 1024 / 1024, 2),
    })


@require_token
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "无效的 JSON 数据"}), 400

        user_input = data.get("message", "").strip()
        images = data.get("images") or []
        file_path = (data.get("file_path") or "").strip()

        if not user_input and not images and not file_path:
            return jsonify({"error": "请输入内容或上传文件"}), 400

        # ── 如果有上传的文件，将文件路径注入到用户输入中 ──
        if file_path and os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            user_input = f"[📎 已上传文件: {file_name}（路径: {file_path}）]\n\n{user_input}"
            logger.info(f"📎 聊天消息附加文件: {file_name}")

        logger.info(f"📨 用户输入: {user_input[:100]}")

        # ── 新请求开始时清除停止标志 ──
        clear_stop()

        try:
            agent, config = get_agent()
        except ValueError as e:
            logger.warning(f"Agent 未就绪: {e}")
            return jsonify({"error": str(e), "need_config": True}), 400

        fix_orphan_tool_calls(agent, config)
        _repair_checkpoint_vision(agent, config)

        # ── 调用前检查是否已被停止 ──
        if check_stop():
            reason = get_stop_reason()
            return jsonify({"response": f"🛑 {reason}", "stopped": True}), 200

        response = _invoke_agent(agent, {"messages": [_build_user_message(user_input, images)]}, config)
        reply = response["messages"][-1].content
        usage = _extract_usage(response["messages"][-1])
        logger.info(f"🤖 Agent 回复: {reply[:100]}...")
        result = {"response": reply, "usage": usage}

        # ── AI 智能沉淀（非流式）──
        # 使用原始用户输入（不包含文件路径）做沉淀
        clean_user_input = user_input.split("[📎 已上传文件")[0].strip() if file_path else user_input
        if reply and len(reply) > 20:
            suggestion = _extract_qa_suggestion(clean_user_input, reply)
            if suggestion:
                suggestion["suggestion_id"] = str(int(time.time() * 1000))
                result["suggestion"] = suggestion
        return jsonify(result)
    except ValueError as e:
        err_msg = str(e)
        logger.error(f"❌ 聊天历史校验错误: {e}", exc_info=True)

        # 兜底硬重置：直接删检查点，完全绕过 Agent
        if "tool_calls" in err_msg or "ToolMessage" in err_msg or "INVALID_CHAT_HISTORY" in err_msg:
            logger.warning("⚠️ 触发兜底硬重置")
            try:
                _reset_checkpoint(DEFAULT_THREAD_ID)
                agent, config = get_agent()
                response = _invoke_agent(agent, {"messages": [_build_user_message(user_input, images)]}, config)
                return jsonify({"response": response["messages"][-1].content})
            except Exception as e2:
                logger.error(f"❌ 硬重置后仍失败: {e2}", exc_info=True)
                return jsonify({"error": "会话状态异常，已硬重置但仍无法恢复，请手动点击「新对话」"}), 500

        try:
            agent, config = get_agent()
            fix_orphan_tool_calls(agent, {"configurable": {"thread_id": DEFAULT_THREAD_ID}})
            _repair_checkpoint_vision(agent, {"configurable": {"thread_id": DEFAULT_THREAD_ID}})
            response = _invoke_agent(agent, {"messages": [_build_user_message(user_input, images)]}, config)
            return jsonify({"response": response["messages"][-1].content, "usage": _extract_usage(response["messages"][-1])})
        except Exception as e2:
            logger.error(f"❌ 重试也失败: {e2}", exc_info=True)
            return jsonify({"error": "会话状态异常，已尝试自动修复但仍失败，请刷新页面或点击「新对话」"}), 500
    except Exception as e:
        logger.error(f"❌ 处理请求异常: {str(e)}", exc_info=True)
        return jsonify({"error": "服务内部错误，请稍后重试"}), 500


def _process_stream_chunk(chunk, last_usage, last_chunk_id):
    """处理单个 stream chunk，返回 (yielded_events, updated_last_usage, new_chunk_id)。
    用于重试时跳过已处理的 chunk（避免重复输出）。
    """
    events = []
    new_last_usage = last_usage
    new_chunk_id = last_chunk_id

    for node_name, update in chunk.items():
        if "messages" not in update:
            continue
        messages = update["messages"]
        if not messages:
            continue
        last_msg = messages[-1]
        msg_type = getattr(last_msg, "type", None)

        # 收集 token 用量（最后一条 AI 消息为准）
        if msg_type == "ai":
            usage = _extract_usage(last_msg)
            if usage:
                new_last_usage = usage

        event_data = {
            "type": "message",
            "content": "",
            "tool_name": None,
            "tool_input": None,
            "tool_output": None,
        }

        if msg_type == "ai":
            if hasattr(last_msg, "content") and last_msg.content:
                event_data["content"] = last_msg.content
            tool_calls = _get_tool_calls_from_msg(last_msg)
            if tool_calls:
                tool = tool_calls[0]
                tool_name = (
                    tool.get("function", {}).get("name")
                    if isinstance(tool, dict)
                    else getattr(tool, "name", None)
                ) or (tool.get("name") if isinstance(tool, dict) else None)
                tool_input = (
                    tool.get("function", {}).get("arguments")
                    if isinstance(tool, dict)
                    else getattr(tool, "arguments", None)
                ) or (tool.get("args") if isinstance(tool, dict) else None)
                event_data["type"] = "tool_call"
                event_data["tool_name"] = tool_name
                event_data["tool_input"] = _coerce_str(tool_input)
            else:
                event_data["type"] = "message"

        elif msg_type == "tool":
            event_data["type"] = "tool_result"
            event_data["tool_name"] = getattr(last_msg, "name", "unknown")
            event_data["tool_output"] = _coerce_str(last_msg.content)

        elif msg_type == "human":
            continue

        else:
            if hasattr(last_msg, "content") and last_msg.content:
                event_data["content"] = last_msg.content
            else:
                continue

        if event_data["content"] or event_data["tool_name"]:
            events.append(event_data)

    return events, new_last_usage, new_chunk_id


@require_token
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "无效的 JSON 数据"}), 400

        user_input = data.get("message", "").strip()
        images = data.get("images") or []
        file_path = (data.get("file_path") or "").strip()

        if not user_input and not images and not file_path:
            return jsonify({"error": "请输入内容或上传文件"}), 400

        # ── 如果有上传的文件，将文件路径注入到用户输入中 ──
        if file_path and os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            user_input = f"[📎 已上传文件: {file_name}（路径: {file_path}）]\n\n{user_input}"
            logger.info(f"📎 聊天消息附加文件: {file_name}")

        logger.info(f"📨 流式用户输入: {user_input[:100]} (图片 {len(images)} 张)")
        try:
            agent, config = get_agent()
        except ValueError as e:
            logger.warning(f"Agent 未就绪: {e}")
            return jsonify({"error": str(e), "need_config": True}), 400

        def generate():
            # 每次新任务开始时清除之前的停止标志
            clear_stop()

            retry_count = 0
            last_usage = None
            loop_detector = LoopDetector()
            timeout_sec = 300  # 5 分钟总超时
            # 收集 Agent 回复内容（用于后续 AI 知识沉淀）
            assistant_response_parts = []

            while True:
                # ── 每轮重试前检查停止标志 ──
                if check_stop():
                    reason = get_stop_reason()
                    logger.warning(f"🛑 检测到停止请求，放弃重试: {reason}")
                    yield f"data: {json.dumps({'type': 'stopped', 'content': reason})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break

                try:
                    # ── 每次重试前重置检查点（防止残留状态导致循环）──
                    if retry_count > 0:
                        logger.warning(f"🔄 重试前重置检查点")
                        _reset_checkpoint(DEFAULT_THREAD_ID)

                    fix_orphan_tool_calls(agent, config)
                    _repair_checkpoint_vision(agent, config)

                    # ── 上下文窗口管理：超限时裁剪旧消息 + 摘要注入 ──
                    _active_cfg = get_active_config()
                    _ctx_result = manage_context_window(agent, config, _active_cfg.get("model", ""))
                    if _ctx_result.get("trimmed"):
                        yield f"data: {json.dumps({'type': 'context_trimmed', 'data': _ctx_result})}\n\n"

                    loop_detector.reset()

                    if retry_count == 0:
                        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

                    user_msg = _build_user_message(user_input, images)
                    stream = agent.stream(
                        {"messages": [user_msg]},
                        config=config,
                        stream_mode="updates",
                    )

                    # 设置任务超时（防止 Agent 无限运行）
                    import signal as _signal
                    timeout_flag = {"expired": False}

                    def _timeout_handler(signum, frame):
                        timeout_flag["expired"] = True

                    if hasattr(_signal, 'SIGALRM'):
                        _signal.signal(_signal.SIGALRM, _timeout_handler)
                        _signal.alarm(timeout_sec)

                    try:
                        for chunk in stream:
                            # ── 每个 chunk 前检查停止标志 ──
                            if check_stop():
                                reason = get_stop_reason()
                                raise StopIteration(f"user_stopped:{reason}")

                            if timeout_flag["expired"]:
                                raise TimeoutError(f"任务执行超时（{timeout_sec}s），已自动终止")

                            events, last_usage, _ = _process_stream_chunk(chunk, last_usage, 0)

                            for ev in events:
                                # ── 收集 Agent 回复内容（用于 AI 知识沉淀）──
                                if ev.get("type") == "message" and ev.get("content"):
                                    assistant_response_parts.append(ev["content"])

                                # ── 循环检测：只检测 tool_call 事件 ──
                                if ev.get("type") == "tool_call":
                                    tool_name = ev.get("tool_name", "unknown")
                                    tool_input = ev.get("tool_input", "") or ""
                                    is_loop, loop_msg = loop_detector.record_call(tool_name, tool_input)
                                    if is_loop:
                                        logger.warning(f"⚠️ 检测到循环: {loop_msg}")
                                        yield f"data: {json.dumps({'type': 'loop_detected', 'content': loop_msg})}\n\n"
                                        raise StopIteration(f"loop:{loop_msg}")

                                # ── 每个事件前也检查一次停止标志 ──
                                if check_stop():
                                    reason = get_stop_reason()
                                    raise StopIteration(f"user_stopped:{reason}")

                                yield f"data: {json.dumps(ev)}\n\n"

                    finally:
                        if hasattr(_signal, 'SIGALRM'):
                            _signal.alarm(0)

                    # 成功完成
                    if last_usage:
                        yield f"data: {json.dumps({'type': 'usage', **last_usage})}\n\n"

                    # ── AI 智能沉淀：提取 Q&A 建议 ──
                    full_response = "".join(assistant_response_parts).strip()
                    if full_response and len(full_response) > 20:
                        suggestion = _extract_qa_suggestion(user_input, full_response)
                        if suggestion:
                            suggestion["suggestion_id"] = str(int(time.time() * 1000))
                            yield f"data: {json.dumps({'type': 'knowledge_suggestion', **suggestion})}\n\n"

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break

                except StopIteration:
                    # StopIteration 可能是循环检测或用户停止触发的
                    reason = str(StopIteration.__cause__) if getattr(StopIteration, '__cause__', None) else ""
                    if isinstance(StopIteration, type):
                        # StopIteration 异常对象可能带有值
                        pass
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break

                except ValueError as e:
                    err_msg = str(e)
                    logger.error(f"流式聊天历史校验错误: {str(e)}", exc_info=True)

                    # 检查是否已停止
                    if check_stop():
                        reason = get_stop_reason()
                        yield f"data: {json.dumps({'type': 'stopped', 'content': reason})}\n\n"
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break

                    if "tool_calls" in err_msg or "ToolMessage" in err_msg or "INVALID_CHAT_HISTORY" in err_msg:
                        _reset_checkpoint(DEFAULT_THREAD_ID)
                        logger.warning("⚠️ SSE 流触发硬重置，尝试重新连接")

                        retry_count += 1
                        if retry_count > MAX_AGENT_RETRIES:
                            yield f"data: {json.dumps({'type': 'error', 'content': '会话状态异常，已多次尝试恢复但失败，请刷新页面重试'})}\n\n"
                            break

                        wait_time = min(2 ** retry_count, 30)
                        logger.warning(f"🔄 API 异常第 {retry_count} 次重试（等待 {wait_time}s）")
                        yield f"data: {json.dumps({'type': 'retrying', 'count': retry_count, 'max': MAX_AGENT_RETRIES, 'message': '会话历史异常，已重置并恢复中...'})}\n\n"
                        time.sleep(wait_time)

                    else:
                        yield f"data: {json.dumps({'type': 'error', 'content': '会话状态异常，请重试'})}\n\n"
                        break

                except _RETRYABLE_EXCEPTIONS as e:
                    # 检查是否已停止
                    if check_stop():
                        reason = get_stop_reason()
                        yield f"data: {json.dumps({'type': 'stopped', 'content': reason})}\n\n"
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break

                    retry_count += 1
                    if retry_count > MAX_AGENT_RETRIES:
                        logger.error(f"❌ API 连续失败 {MAX_AGENT_RETRIES} 次，放弃重试: {str(e)}", exc_info=True)
                        yield f"data: {json.dumps({'type': 'error', 'content': f'API 连接持续失败（已重试 {MAX_AGENT_RETRIES} 次），请稍后重试'})}\n\n"
                        break

                    wait_time = min(2 ** retry_count, 30)
                    logger.warning(f"🔄 API 异常第 {retry_count} 次重试（等待 {wait_time}s）: {type(e).__name__}")
                    yield f"data: {json.dumps({'type': 'retrying', 'count': retry_count, 'max': MAX_AGENT_RETRIES, 'message': f'API 连接中断，{wait_time}秒后自动重试...'})}\n\n"
                    time.sleep(wait_time)

                except Exception as e:
                    if _is_retryable_error(e):
                        # 检查是否已停止
                        if check_stop():
                            reason = get_stop_reason()
                            yield f"data: {json.dumps({'type': 'stopped', 'content': reason})}\n\n"
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            break

                        retry_count += 1
                        if retry_count > MAX_AGENT_RETRIES:
                            logger.error(f"❌ 连续失败 {MAX_AGENT_RETRIES} 次，放弃重试: {str(e)}", exc_info=True)
                            yield f"data: {json.dumps({'type': 'error', 'content': '任务连续失败，已多次重试仍无法完成，请稍后重试'})}\n\n"
                            break

                        wait_time = min(2 ** retry_count, 30)
                        logger.warning(f"🔄 未知异常第 {retry_count} 次重试（等待 {wait_time}s）: {str(e)[:100]}")
                        yield f"data: {json.dumps({'type': 'retrying', 'count': retry_count, 'max': MAX_AGENT_RETRIES, 'message': f'执行异常，{wait_time}秒后自动重试...'})}\n\n"
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ 不可重试错误: {str(e)}", exc_info=True)
                        yield f"data: {json.dumps({'type': 'error', 'content': '任务执行失败'})}\n\n"
                        break

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f"chat_stream 视图异常: {str(e)}", exc_info=True)
        return jsonify({"error": "服务器内部错误"}), 500


# ============================================================
# 路由：用户手动停止任务（安全功能，不要求 Token 鉴权）
# ============================================================
@app.route("/stop", methods=["POST"])
def stop_task():
    """
    用户手动中断当前正在执行的任务。
    设置全局停止标志，正在运行的 Agent 会在下一个检查点读取该标志并退出。
    """
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "用户手动停止").strip() or "用户手动停止"
    request_stop(reason)
    return jsonify({
        "status": "ok",
        "message": f"🛑 已请求停止任务: {reason}",
        "reason": reason,
    })


@require_token
@app.route("/reset", methods=["POST"])
def reset_conversation():
    try:
        conn = sqlite3.connect(CHECKPOINT_DB)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (DEFAULT_THREAD_ID,))
        conn.commit()
        conn.close()
        logger.info(f"已重置会话 thread_id: {DEFAULT_THREAD_ID}")
        return jsonify({"status": "ok", "message": "会话已重置"})
    except Exception as e:
        logger.error(f"重置会话失败: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    try:
        conn = sqlite3.connect(CHECKPOINT_DB)
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False
    settings = get_active_config()
    return jsonify({
        "status": "ok",
        "thread_id": DEFAULT_THREAD_ID,
        "db_ok": db_ok,
        "agent_ready": bool(settings.get("api_key", "")),
        "llm_provider": settings.get("provider", ""),
        "llm_model": settings.get("model", ""),
    })


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info(f"🚀 启动服务: http://{host}:{port}, debug={debug}")
    app.run(debug=debug, host=host, port=port)
