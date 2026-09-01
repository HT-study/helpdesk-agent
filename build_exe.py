# -*- coding: utf-8 -*-
"""使用 PyInstaller API 直接打包 dsh.exe"""
import sys
import os

# 确保能导入 PyInstaller
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PyInstaller import __main__ as pyi_main

# 构建参数
args = [
    'dsh_desktop.py',
    '--name', 'dsh',
    '--onefile',
    '--noconsole',
    '--add-data', f'templates{os.pathsep}templates',
    '--hidden-import', 'sqlite3',
    '--hidden-import', 'openpyxl',
    '--hidden-import', 'xlrd',
    '--hidden-import', 'tkinter',
    '--hidden-import', 'app',
    '--hidden-import', 'config',
    '--hidden-import', 'models.agent_model',
    '--hidden-import', 'models.tools',
    '--hidden-import', 'services.audit_service',
    '--hidden-import', 'services.kb_service',
    '--hidden-import', 'services.host_service',
    '--hidden-import', 'services.logger_service',
    '--collect-all', 'langgraph',
    '--collect-all', 'langchain_core',
    '--collect-all', 'langchain_openai',
    '--collect-all', 'langchain_community',
    '--collect-all', 'paramiko',
    '--collect-all', 'pandas',
    '--collect-all', 'matplotlib',
    '--noconfirm',
]

print("=" * 60)
print("  DSH AI Agent - PyInstaller 打包")
print("  " + " ".join(args[:6]) + " ...")
print("=" * 60)

try:
    pyi_main.run(args)
    print("\n✅ 打包完成！")
    print(f"exe: {os.path.abspath('dist/dsh.exe')}")
except Exception as e:
    print(f"\n❌ 打包失败: {e}")
    sys.exit(1)