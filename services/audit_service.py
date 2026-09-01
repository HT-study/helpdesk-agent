# services/audit_service.py - 工具调用审计日志
import sqlite3
import json
from datetime import datetime

from services.logger_service import get_logger

logger = get_logger()

AUDIT_TABLE = "tool_audit"
_DB = None  # 复用主连接（可选）


def _conn():
    from config import CHECKPOINT_DB
    return sqlite3.connect(CHECKPOINT_DB, timeout=10)


def init_audit_db():
    """初始化审计日志表（幂等）"""
    conn = _conn()
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            args TEXT,
            result TEXT,
            success INTEGER NOT NULL,
            duration_ms INTEGER
        )"""
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_audit_ts ON {AUDIT_TABLE}(ts)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_audit_tool ON {AUDIT_TABLE}(tool_name)")
    conn.commit()
    conn.close()
    logger.info("✅ 审计日志表已就绪")


def audit_log(tool_name, args, result, success, duration_ms):
    """记录一次工具调用。result 会被截断到 2000 字符。"""
    try:
        conn = _conn()
        conn.execute(
            f"INSERT INTO {AUDIT_TABLE}(ts, tool_name, args, result, success, duration_ms) VALUES (?,?,?,?,?,?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tool_name,
                json.dumps(args, ensure_ascii=False) if args else "",
                (result or "")[:2000],
                1 if success else 0,
                int(duration_ms),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"审计日志写入失败: {e}")


def get_audit_logs(limit=100, tool_name=None):
    """查询审计日志，按 id 倒序"""
    conn = _conn()
    if tool_name:
        cur = conn.execute(
            f"SELECT id, ts, tool_name, args, result, success, duration_ms FROM {AUDIT_TABLE} WHERE tool_name=? ORDER BY id DESC LIMIT ?",
            (tool_name, limit),
        )
    else:
        cur = conn.execute(
            f"SELECT id, ts, tool_name, args, result, success, duration_ms FROM {AUDIT_TABLE} ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    rows = cur.fetchall()
    conn.close()
    cols = ["id", "ts", "tool_name", "args", "result", "success", "duration_ms"]
    return [dict(zip(cols, r)) for r in rows]


def clear_audit_logs():
    """清空审计日志"""
    conn = _conn()
    conn.execute(f"DELETE FROM {AUDIT_TABLE}")
    conn.commit()
    conn.close()
