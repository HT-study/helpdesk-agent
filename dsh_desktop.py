"""
DSH Desktop Launcher — 双击启动 AI Agent 桌面应用
- 后台启动 Flask 服务 (port 5000)
- 自动打开浏览器
- 小窗口显示运行状态
"""
import os
import sys
import threading
import webbrowser
import time

# ── 确定 exe 所在目录（PyInstaller 打包后用 sys.executable，开发时用 __file__） ──
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后：exe 所在目录
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录：优先用 exe 同目录，确保配置/数据库持久化
DATA_DIR = APP_DIR

os.chdir(DATA_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# ── 环境变量：把数据文件路径指向 exe 同目录（避免写到临时解压目录） ──
os.environ.setdefault("FLASK_PORT", "5000")
os.environ.setdefault("API_TOKEN", "dsh-admin")
os.environ.setdefault("LLM_SETTINGS_FILE", os.path.join(DATA_DIR, "llm_settings.json"))
os.environ.setdefault("LOG_DIR", os.path.join(DATA_DIR, "logs"))
os.environ.setdefault("LOG_FILE", "agent.log")
os.environ.setdefault("CHECKPOINT_DB", os.path.join(DATA_DIR, "checkpoints.sqlite"))

PORT = int(os.environ.get("FLASK_PORT", "5000"))
URL = f"http://127.0.0.1:{PORT}"


def start_server():
    """在守护线程中启动 Flask 应用"""
    try:
        from app import app
        # 打包后确保模板路径正确：优先 exe 同目录的 templates，其次临时解压目录
        import flask
        tmpl_candidates = [
            os.path.join(DATA_DIR, "templates"),
            os.path.join(sys._MEIPASS, "templates") if getattr(sys, 'frozen', False) else None,
        ]
        for tp in tmpl_candidates:
            if tp and os.path.isdir(tp):
                app.template_folder = tp
                app.static_folder = tp  # 模板里内联 CSS/JS，static 也指向同目录
                break
        # 关闭 Flask reloader（避免 PyInstaller 下双进程）
        app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
    except Exception as e:
        print(f"[ERROR] 启动失败: {e}")
        sys.exit(1)


def open_browser():
    """延迟 2 秒打开浏览器，等待服务就绪"""
    time.sleep(2)
    webbrowser.open(URL)


def run_tray():
    """系统托盘图标（使用 tkinter 实现，无需第三方依赖）"""
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        # 无 tkinter 则退化为控制台等待
        print(f"\n✅ 服务已启动: {URL}")
        print("按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            os._exit(0)
        return

    root = tk.Tk()
    root.title("DSH AI Agent")
    root.geometry("320x180")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    # 居中
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 320) // 2
    y = (root.winfo_screenheight() - 180) // 2
    root.geometry(f"320x180+{x}+{y}")

    label = tk.Label(root, text="🤖 DSH AI Agent 运行中", font=("Microsoft YaHei", 14))
    label.pack(pady=20)

    url_label = tk.Label(root, text=f"访问地址: {URL}", font=("Consolas", 10), fg="#6366f1", cursor="hand2")
    url_label.pack()
    url_label.bind("<Button-1>", lambda e: webbrowser.open(URL))

    def do_open():
        webbrowser.open(URL)

    def do_stop():
        if messagebox.askokcancel("退出", "确定停止服务并退出？"):
            os._exit(0)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=12)
    tk.Button(btn_frame, text="打开页面", width=10, command=do_open, bg="#6366f1", fg="white").pack(side="left", padx=8)
    tk.Button(btn_frame, text="停止退出", width=10, command=do_stop, bg="#ef4444", fg="white").pack(side="left", padx=8)

    root.protocol("WM_DELETE_WINDOW", lambda: root.iconify())  # 关闭窗口不退出，最小化
    root.mainloop()


def main():
    print("=" * 50)
    print("  DSH AI Agent Desktop Launcher")
    print(f"  服务地址: {URL}")
    print("=" * 50)

    # 启动 Flask（守护线程）
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # 打开浏览器
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # 主线程跑 GUI / 托盘
    run_tray()


if __name__ == "__main__":
    main()
