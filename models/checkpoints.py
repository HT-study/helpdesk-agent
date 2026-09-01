# models/checkpoints.py - 检查点数据库瘦身与边界写入
#
# 目标：把 LangGraph SqliteSaver 的中间 writes 表禁用 + 按需修剪旧 checkpoints。
#
# 关键洞察：
# - 当前 app.py 的 SSE 流式使用 stream_mode="updates"，不读取 writes 表；
# - manage_context_window() / fix_orphan_tool_calls() 通过 agent.update_state() 写 checkpoints 表，
#   不经过 put_writes()；
# - put_writes() 每轮 tool_call 都会写入中间 blob（每条约 100 KB~2 MB），是 checkpoints.sqlite
#   膨胀到 800+ MB 的元凶。
#
# 因此：
# 1. 重写 put_writes() 为 no-op —— 阻止新中间态继续落盘；
# 2. 在 put() 之后做 checkpoint 行修剪（保留每个 thread 的最近 N 行），防止 checkpoints 表
#    也随轮次累积；
# 3. 提供 compact_checkpoints_db() 一次压库：删全量 writes + 按 size mb 上限裁剪历史行 + VACUUM。

from __future__ import annotations

import logging
import os
import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver

from config import (
    CHECKPOINT_DB,
    MAX_CHECKPOINTS_PER_THREAD,
    CHECKPOINT_SIZE_MB,
    WAL_CHECKPOINT_ON_START,
)

logger = logging.getLogger(__name__)


class BoundedSqliteSaver(SqliteSaver):
    """只保留 turn 边界的 checkpoints、不写中间 writes 的 SqliteSaver。

    行为：
    - put_writes() → 直接 return（no-op）：SSE 流式 (stream_mode="updates") 不需要 writes 表。
    - put() → 保留原行为（写 checkpoints 行），之后触发 checkpoints 表修剪。
    - delete_thread() / get_tuple() / list() / get_delta_channel_history()
      保持与 SqliteSaver 完全一致。

    线程安全：父类已用 self.lock 包住 cursor 事务；我们在同一锁内做修剪。
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        keep: int | None = None,
        serde=None,
    ) -> None:
        self.keep: int = keep if keep is not None else MAX_CHECKPOINTS_PER_THREAD
        super().__init__(conn, serde=serde)

    # -- 1) 禁用中间 writes ------------------------------------------------
    def put_writes(self, config, writes, task_id: str, task_path: str = "") -> None:  # type: ignore[override]
        """重写为 no-op：不持久化中间 tool_call 输出。

        说明：app 的 SSE 流式使用 stream_mode="updates"，chunk 直接来自图运行时
        state 变更，不读 writes 表。中间态写入只用于回放 / debug，
        对本应用无价值，且是 DB 膨胀主因。
        """
        return None

    # -- 2) 写 checkpoints 后不自动修剪 -------------------------------------
    #    checkpoint_id 是 UUID，ORDER BY checkpoint_id DESC 与插入顺序无关，
    #    盲目按字符串删除中间节点会切断 parent 链。
    #    因此 put() 保留父类行为；如需瘦身，显式调用 compact_checkpoints_db()
    #    （启动时自动 + 可通过 /compact API 手动触发）。
    def put(self, config, checkpoint, metadata, new_versions):
        return super().put(config, checkpoint, metadata, new_versions)


def _sqlite_conn() -> sqlite3.Connection:
    """打开 checkpoints 数据库连接（供静态工具函数使用）。"""
    return sqlite3.connect(CHECKPOINT_DB, timeout=15)


def prune_checkpoints(
    *,
    keep: int | None = None,
    all_threads: bool = True,
    verbose: bool = True,
) -> dict:
    """按需删除 checkpoints 表里每个 thread 的旧行（保留最近 keep 条），
    并同步清理 writes 表中对应 checkpoint_id 的中间行。

    说明：checkpoint_id 是 UUID，ORDER BY 不反映插入顺序。
    因此「最近 K 条」通过 parent 链回溯得到：从 head 沿 parent_checkpoint_id
    一路回溯 K 步，把这一路径上的所有祖先标为「保留」，剩余行全部删除。
    """
    keep = keep if keep is not None else MAX_CHECKPOINTS_PER_THREAD
    if keep <= 0:
        return {"deleted_checkpoints": 0, "deleted_writes": 0}

    conn = _sqlite_conn()
    try:
        cur = conn.cursor()
        if all_threads:
            threads = cur.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        else:
            threads = cur.execute(
                "SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''"
            ).fetchall()

        deleted_cp = 0
        deleted_w = 0
        for (thread_id,) in threads:
            head = cur.execute(
                """SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints
                   WHERE thread_id = ?
                   ORDER BY checkpoint_id DESC LIMIT 1""",
                (thread_id,),
            ).fetchone()
            if not head:
                continue

            # 沿 parent 链回溯 keep 步，收集要保留的 checkpoint_id 集合
            keep_ids: set[str] = set()
            cid = head[0]
            for _ in range(keep):
                if cid is None:
                    break
                keep_ids.add(str(cid))
                parent = cur.execute(
                    "SELECT parent_checkpoint_id FROM checkpoints WHERE checkpoint_id = ?",
                    (cid,),
                ).fetchone()
                cid = parent[0] if parent else None

            # 找出要删除的（不在保留集合中的）
            to_del = cur.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchall()
            to_del_ids = [r[0] for r in to_del if r[0] not in keep_ids]
            if not to_del_ids:
                continue

            ph = ",".join("?" for _ in to_del_ids)
            cur.execute(
                f"DELETE FROM writes WHERE thread_id = ? AND checkpoint_id IN ({ph})",
                [thread_id, *to_del_ids],
            )
            deleted_w += cur.rowcount
            cur.execute(
                f"DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id IN ({ph})",
                [thread_id, *to_del_ids],
            )
            deleted_cp += cur.rowcount
        conn.commit()
        if verbose:
            logger.info(
                f"🧹 prune_checkpoints 完成: 删除 checkpoints={deleted_cp}, writes={deleted_w}"
            )
        return {"deleted_checkpoints": deleted_cp, "deleted_writes": deleted_w}
    finally:
        conn.close()


def compact_checkpoints_db(
    *,
    target_mb: float | None = None,
    keep: int | None = None,
    vacuum: bool = True,
) -> dict:
    """一次压库：删除全量 writes + 按目标体积裁剪 checkpoints + VACUUM。

    用于：
    - 首次上线把 800+ MB 的旧库压回健康大小；
    - 定期维护（可选挂定时任务）。

    参数：
        target_mb: checkpoints 表目标大小（MB）。超出时按最老→最新删除行直到达标。
                   为 None 或 ≤ 0 则跳过。
        keep:      每个 thread 至少保留的 checkpoints 行数（prune_checkpoints 用）。
        vacuum:    是否执行 VACUUM 回收磁盘（会锁库，建议在非运行期调用）。

    返回: {deleted_writes, deleted_checkpoints, freed_mb, vacuumed}
    """
    target_mb = target_mb if target_mb is not None else CHECKPOINT_SIZE_MB
    conn = _sqlite_conn()
    try:
        sz_before = os.path.getsize(CHECKPOINT_DB)
        cur = conn.cursor()

        # 1) 清空全量 writes（对当前 stream_mode 无影响）
        cur.execute("DELETE FROM writes")
        deleted_writes = cur.rowcount
        conn.commit()

        # 2) 裁剪 checkpoints：先按 keep 修剪，再按 size 上限裁剪
        if keep is not None:
            prune = prune_checkpoints(keep=keep, verbose=False)
            deleted_checkpoints = prune["deleted_checkpoints"]
        else:
            deleted_checkpoints = 0

        if target_mb and target_mb > 0:
            # 循环删除「最老叶子」checkpoint（parent 链最深、但不在最后 K 条内）
            # 直到总 size 达标。删除前先走 parent 链确认它是安全的 leaf。
            keep_floor = keep if keep is not None else 3
            while True:
                info = cur.execute("PRAGMA page_count").fetchone()
                pg = cur.execute("PRAGMA page_size").fetchone()
                size_mb = (info[0] * pg[0]) / (1024 * 1024)
                if size_mb <= target_mb:
                    break
                rows = cur.execute("SELECT count(*) FROM checkpoints").fetchone()
                if rows[0] <= keep_floor:
                    break
                # 按 parent 链长度 + checkpoint_id 升序，取最老的一条 leaf 删除
                oldest = cur.execute(
                    """SELECT thread_id, checkpoint_id FROM checkpoints
                       WHERE checkpoint_id NOT IN (
                           SELECT parent_checkpoint_id FROM checkpoints WHERE parent_checkpoint_id IS NOT NULL
                       )
                       ORDER BY checkpoint_id ASC LIMIT 1"""
                ).fetchone()
                if not oldest:
                    break
                thread_id, cid = oldest
                cur.execute("DELETE FROM writes WHERE thread_id = ? AND checkpoint_id = ?", (thread_id, cid))
                cur.execute("DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?", (thread_id, cid))
                conn.commit()
                deleted_checkpoints += 1

        sz_after = os.path.getsize(CHECKPOINT_DB)

        # 3) WAL checkpoint（写回主库）
        if WAL_CHECKPOINT_ON_START:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.commit()
            except Exception as e:
                logger.warning(f"WAL checkpoint 失败: {e}")

        # 4) VACUUM 回收磁盘
        vacuumed = False
        if vacuum:
            try:
                conn.execute("VACUUM")
                conn.commit()
                vacuumed = True
            except Exception as e:
                logger.warning(f"VACUUM 失败: {e}")

        sz_final = os.path.getsize(CHECKPOINT_DB)
        freed_mb = (sz_before - sz_final) / (1024 * 1024)
        logger.info(
            f"💾 compact_checkpoints_db: writes=-{deleted_writes}, checkpoints=-{deleted_checkpoints}, "
            f"size {sz_before / (1024 * 1024):.1f} → {sz_final / (1024 * 1024):.1f} MB "
            f"(freed {freed_mb:.1f} MB, vacuumed={vacuumed})"
        )
        return {
            "deleted_writes": deleted_writes,
            "deleted_checkpoints": deleted_checkpoints,
            "freed_mb": round(freed_mb, 2),
            "size_before_mb": round(sz_before / (1024 * 1024), 2),
            "size_after_mb": round(sz_final / (1024 * 1024), 2),
            "vacuumed": vacuumed,
        }
    finally:
        conn.close()


def checkpoint_db_size_mb() -> float:
    """返回 checkpoints.sqlite 的当前大小（MB）。"""
    try:
        return round(os.path.getsize(CHECKPOINT_DB) / (1024 * 1024), 2)
    except OSError:
        return 0.0
