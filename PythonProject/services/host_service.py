# services/host_service.py - 远程主机管理 + SSH 执行
import sqlite3
import socket
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

from services.logger_service import get_logger

logger = get_logger()

HOST_TABLE = "remote_hosts"


def _conn():
    from config import CHECKPOINT_DB
    return sqlite3.connect(CHECKPOINT_DB, timeout=10)


def init_hosts_db():
    conn = _conn()
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {HOST_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            hostname TEXT NOT NULL,
            port INTEGER DEFAULT 22,
            username TEXT NOT NULL,
            auth_type TEXT DEFAULT 'password',
            credential TEXT,
            created_at TEXT
        )"""
    )
    conn.commit()
    conn.close()
    logger.info("✅ 远程主机表已就绪")


def list_hosts():
    conn = _conn()
    cur = conn.execute(
        f"SELECT id, label, hostname, port, username, auth_type, created_at FROM {HOST_TABLE} ORDER BY id"
    )
    rows = [
        {
            "id": r[0],
            "label": r[1],
            "hostname": r[2],
            "port": r[3],
            "username": r[4],
            "auth_type": r[5],
            "created_at": r[6],
        }
        for r in cur.fetchall()
    ]
    conn.close()
    return rows


def save_host(label, hostname, port, username, auth_type, credential):
    conn = _conn()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        f"INSERT OR REPLACE INTO {HOST_TABLE}(label, hostname, port, username, auth_type, credential, created_at) VALUES(?,?,?,?,?,?,?)",
        (label, hostname, port, username, auth_type, credential, now),
    )
    conn.commit()
    conn.close()
    logger.info(f"主机已保存: {label} ({hostname})")


def delete_host(host_id):
    conn = _conn()
    conn.execute(f"DELETE FROM {HOST_TABLE} WHERE id=?", (host_id,))
    conn.commit()
    conn.close()


def _get_host_by_label(label):
    """按 label 查询完整主机信息（含 credential）"""
    conn = _conn()
    cur = conn.execute(
        f"SELECT id, label, hostname, port, username, auth_type, credential FROM {HOST_TABLE} WHERE label=?",
        (label,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "label": row[1],
        "hostname": row[2],
        "port": row[3],
        "username": row[4],
        "auth_type": row[5],
        "credential": row[6],
    }


def _ssh_connect_and_exec(host, cmd, timeout=15):
    """通过 SSH 在远程主机执行命令，返回 (success, output)"""
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if host["auth_type"] == "key":
            # 尝试多种密钥类型
            key = None
            for cls_name in ("RSAKey", "Ed25519Key", "ECDSAKey"):
                try:
                    cls = getattr(paramiko, cls_name)
                    key = cls.from_private_key(StringIO(host["credential"]))
                    break
                except Exception:
                    continue
            if key is None:
                return False, "无法解析 SSH 密钥（支持 RSA/ECDSA/Ed25519）"
            client.connect(
                host["hostname"],
                port=host["port"],
                username=host["username"],
                pkey=key,
                timeout=10,
                allow_agent=False,
                look_for_keys=False,
            )
        else:
            client.connect(
                host["hostname"],
                port=host["port"],
                username=host["username"],
                password=host["credential"],
                timeout=10,
                allow_agent=False,
                look_for_keys=False,
            )
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        stdin.close()
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        client.close()
        text = (output + "\n" + error).strip()
        return True, text
    except Exception as e:
        return False, str(e)


def execute_on_host(label, cmd, timeout=15):
    """在指定远程主机上执行命令（面向 Agent 工具）"""
    host = _get_host_by_label(label)
    if not host:
        return False, f"❌ 未找到主机: {label}，请先用 list_hosts 查看可用主机"
    return _ssh_connect_and_exec(host, cmd, timeout)


def run_on_all_hosts(cmd, max_workers=5):
    """在所有远程主机上并行执行命令，返回 (host_label, success, output) 列表"""
    hosts = list_hosts()
    if not hosts:
        return [("", False, "❌ 未配置远程主机，请先到 /hosts 页面添加")]

    full_hosts = [_get_host_by_label(h["label"]) for h in hosts]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda h: (h["label"], *_ssh_connect_and_exec(h, cmd)), full_hosts))
    return results


def check_host_health(label):
    """检查主机的连通性（TCP 端口检测 + SSH 尝试）"""
    host = _get_host_by_label(label)
    if not host:
        return False, "未找到该主机"
    # 先 TCP 检测
    try:
        s = socket.create_connection((host["hostname"], host["port"]), timeout=5)
        s.close()
        tcp_ok = True
    except Exception:
        tcp_ok = False
        return False, f"端口 {host['port']} 不通"
    # 再 SSH 尝试
    success, output = _ssh_connect_and_exec(host, "echo ok", timeout=8)
    if success and output.strip() == "ok":
        return True, "在线 ✅"
    return False, f"SSH 连接失败: {output[:200]}"