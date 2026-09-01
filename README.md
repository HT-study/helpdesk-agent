# 🤖 DSH — AI Agent Desktop 桌面运维助手

> **基于 LangGraph ReAct Agent + Flask + SQLite 的本地 AI 运维桌面应用**
>
> 支持 DeepSeek / OpenAI / Anthropic 及 OpenAI 兼容服务，提供流式对话、智能工具调用、知识库检索、远程主机管理等完整功能。

---

## 📐 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         浏览器 (Browsersync / Chrome)                │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐          │
│  │  聊天    │  知识库  │ 审计日志  │ 远程主机  │   设置   │          │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘          │
│                         │   SSE / Fetch + JSON                     │
├─────────────────────────┼──────────────────────────────────────────┤
│  Flask (port 5000)      │  /chat /chat/stream /settings /kb /audit  │
│                         │  /hosts /health /stop /reset              │
├─────────────────────────┼──────────────────────────────────────────┤
│  LangGraph ReAct Agent  │  SQLite Checkpoint (对话状态持久化)       │
│  ┌─────────────────────┐│  ┌────────────────────────────────────┐  │
│  │ Agent 主循环         ││  │ Knowledge Base (Q&A 表 + 语义索引) │  │
│  │ ┌─────────────┐     ││  │ Host Service (远程主机 SSH)        │  │
│  │ │ ReAct       │     ││  │ Audit Log (管理操作审计)           │  │
│  │ │ Reasoning   │     ││  │ Context Window (裁剪+摘要)         │  │
│  │ └──────┬──────┘     ││  └────────────────────────────────────┘  │
│  │        │            ││                                           │
│  │ 20+ Tools           ││  ┌────────────────────────────────────┐  │
│  │ ├─ execute_command  ││  │ Config (多 API 配置，热切换)       │  │
│  │ ├─ read_file / write││  └────────────────────────────────────┘  │
│  │ ├─ list_processes   ││                                           │
│  │ ├─ system_stats     ││  ┌────────────────────────────────────┐  │
│  │ ├─ network_check    ││  │ Retry (tenacity 指数退避)          │  │
│  │ ├─ search_log       ││  │ Loop Detection (同工具参数重复拦截) │  │
│  │ ├─ search_kb        ││  │ Manual Stop (用户手动中断)         │  │
│  │ └─ ... (更多工具)   ││  │ Checkpoint Repair (断线恢复)        │  │
│  └─────────────────────┘│  └────────────────────────────────────┘  │
├─────────────────────────┼──────────────────────────────────────────┤
│  services/              │  embed: TF-IDF numpy 向量检索             │
│  ├─ embedding_service   │  kb:   SQLite + full-text + semantic     │
│  ├─ kb_service          │  doc:  pymupdf / python-docx → Q&A       │
│  ├─ audit_service       │  host: paramiko SSH → 远程命令执行       │
│  ├─ host_service        │                                           │
│  └─ logger_service      │                                           │
└─────────────────────────┴──────────────────────────────────────────┘
```

---

## 🛠 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **前端** | HTML5 + CSS3 + Vanilla JS + Font Awesome 6.5 | 单页应用，无构建工具 |
| **后端** | Python 3.11 + Flask 3.x | 轻量级 Web 框架 |
| **Agent** | LangGraph + LangChain | ReAct Agent 主循环 |
| **LLM** | LangChain-OpenAI / DeepSeek / Anthropic | 多提供商热切换 |
| **持久化** | SQLite | Checkpoint + 知识库 + 主机 + 审计 |
| **重试** | tenacity | 指数退避自动重试 |
| **文档** | pymupdf / pdfplumber / python-docx | PDF/DOCX 解析 |
| **嵌入** | numpy TF-IDF | 轻量级语义检索，无外部依赖 |
| **SSH** | paramiko | 远程主机连接与执行 |
| **打包** | PyInstaller | 一键打包为桌面 .exe |

---

## ⚡ 功能清单

> 版本标签：✅ 已实现 | 🔄 进行中 | 🔜 计划 | ❌ 已放弃

### 1. LLM 多模型配置

| 功能 | 状态 | 说明 |
|------|------|------|
| **多提供商支持** | ✅ | DeepSeek / OpenAI / Anthropic / OpenAI 兼容（Sensenova 等） |
| **多配置管理** | ✅ | 创建、编辑、删除、激活多个 API 配置，热切换 |
| **API Key 安全存储** | ✅ | 本地 JSON 文件，前端脱敏显示 |
| **配置测试** | ✅ | 一键测试 API Key 连通性，返回友好错误信息 |
| **配置自动迁移** | ✅ | 旧版单配置格式自动迁移到新版多配置结构 |

### 2. Agent 核心能力

| 功能 | 状态 | 说明 |
|------|------|------|
| **ReAct Agent 主循环** | ✅ | LangGraph 驱动的 Reasoning → Act → Observe 循环 |
| **20+ 内置工具** | ✅ | 系统运维、文件操作、知识库、远程主机、Excel 分析 |
| **工具调用可视化** | ✅ | 展开/折叠面板显示工具输入/输出 |
| **SSE 流式输出** | ✅ | 逐 token 流式渲染，支持 typing 动画 |
| **系统提示词** | ✅ | 内置角色定位（桌面运维助手）与工具说明 |
| **自定义工具热加载** | 🔜 | 用户可注册自定义 Python 工具（规划中） |

### 3. 稳定性与容错

| 功能 | 状态 | 说明 |
|------|------|------|
| **API 自动重试** | ✅ | tenacity 指数退避，最多 5 次重试，网络超时/429/5xx 自动恢复 |
| **循环检测** | ✅ | 相同工具+参数连续重复 5 次自动阻断，防止 Agent 死循环 |
| **手动中断** | ✅ | 用户可一键停止正在运行的任务 |
| **任务超时** | ✅ | 5 分钟硬超时（SIGALRM），防止 Agent 无限运行 |
| **断线恢复** | ✅ | 检测并修复：孤立 tool_calls、残缺 AI 消息、丢失的图片数据 |
| **上下文窗口管理** | ✅ | 超过窗口 70% 时自动裁剪旧消息，LLM 摘要注入系统提示 |
| **检查点瘦身** | ✅ | 禁用中间 writes 写入 + 启动自动压库，DB 从 812 MB 稳定到 ~0.5 MB |

#### 3.1 检查点瘦身原理

> 默认 `SqliteSaver` 每轮 tool_call 都会把完整的中间状态 blob 写入 `writes` 表（每条约 100 KB ~ 2 MB），
> 长期运行后 `checkpoints.sqlite` 会膨胀到数百 MB。

DSH 通过自定义 `BoundedSqliteSaver`（`models/checkpoints.py`）解决：

| 措施 | 说明 |
|------|------|
| **禁用中间 writes** | `put_writes()` 重写为 no-op。DSH 使用 `stream_mode="updates"`，chunk 直接来自图运行时状态变更，不依赖 writes 表回放。 |
| **保留 turn 边界** | `put()` 保持原样，只写 checkpoints 表（每轮最终态约几十 KB），可正常 `get_state` / 恢复。 |
| **启动自动压库** | 每次启动调用 `compact_checkpoints_db()`：清空历史 writes + 按 parent 链裁剪旧 checkpoints + `VACUUM` 回收磁盘。 |
| **API 端点** | `POST /compact`：手动触发一次压库，返回 `{size_before_mb, size_after_mb, freed_mb}`。 |
| **parent 链安全** | checkpoint_id 是 UUID，不能按字符串排序；裁剪时沿 parent 链回溯 keep 步确定保留集合。 |

实测效果：`812 MB → 0.5 MB`（一次压库），后续长期稳定在 `CHECKPOINT_SIZE_MB` 上限内。

#### 3.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_CHECKPOINTS_PER_THREAD` | `30` | 每个 thread 保留的 checkpoints 行数上限 |
| `CHECKPOINT_SIZE_MB` | `50.0` | 目标数据库体积（MB），超出时启动自动裁剪旧行 |

### 4. 知识库

| 功能 | 状态 | 说明 |
|------|------|------|
| **PDF 导入** | ✅ | pymupdf 提取文本 + 图片，自动分割 Q&A |
| **DOCX 导入** | ✅ | python-docx 提取文本 + 图片 |
| **批量导入** | 🔄 | 拖拽目录批量解析（规划中） |
| **语义搜索** | ✅ | numpy TF-IDF 向量索引，无需外部向量数据库 |
| **关键词搜索** | ✅ | SQLite FTS 全文检索 |
| **手动编辑** | ✅ | 在线新增/修改/删除 Q&A 条目 |
| **索引重建** | ✅ | 一键重建语义索引 |
| **导入可视化** | ✅ | 进度条 + 文件列表展示 |

### 5. AI 知识沉淀

| 功能 | 状态 | 说明 |
|------|------|------|
| **自动问答建议** | ✅ | 对话结束后，Agent 分析回复内容，自动提取可沉淀的 Q&A |
| **一键保存到知识库** | ✅ | 前端弹窗确认，确认后保存到 KB |
| **沉淀质量过滤** | ✅ | 过滤过短、无实质内容的建议 |

### 6. 聊天功能

| 功能 | 状态 | 说明 |
|------|------|------|
| **多会话管理** | ✅ | 新建/重命名/删除会话，SQLite 持久化 |
| **图片消息** | ✅ | 拖拽/粘贴/选择上传图片，多模态对话 |
| **文件上传** | ✅ | 聊天中附加文件，路径注入到 Agent 上下文 |
| **快捷指令** | ✅ | 可配置的快捷输入，点击即发送 |
| **Markdown 渲染** | ✅ | 代码块语法高亮、表格、列表等 |
| **复制消息** | ✅ | 单条消息一键复制 |
| **整段导出** | 🔜 | Markdown / PDF 导出（规划中） |
| **流式打字效果** | ✅ | 逐字符渲染 AI 回复 |

### 7. 远程主机管理

| 功能 | 状态 | 说明 |
|------|------|------|
| **主机管理** | ✅ | 添加/编辑/删除 SSH 主机，保存凭据 |
| **远程命令执行** | ✅ | Agent 通过 `execute_on_host` 工具远程执行命令 |
| **批量执行** | ✅ | `run_on_all_hosts` 同时向所有主机执行 |
| **健康检查** | ✅ | `check_host_health` 探测主机存活状态 |
| **连通性测试** | ✅ | 一键测试 SSH 连通性 |
| **批量回滚** | 🔜 | 执行计划预览 + 滚动回滚（规划中） |

### 8. 审计与安全

| 功能 | 状态 | 说明 |
|------|------|------|
| **操作审计日志** | ✅ | 记录管理操作（配置变更、文件操作等） |
| **日志查询** | ✅ | 分页查询、按类型/时间筛选 |
| **日志导出** | ✅ | 一键清空 |
| **Token 认证** | 🔄 | 前端注入 Bearer Token（本地部署场景，规划中） |
| **危险命令二次确认** | 🔜 | `rm -rf` / `format` 等危险操作弹窗确认（规划中） |
| **文件路径白名单** | 🔜 | 限制 Agent 读写路径范围（规划中） |

### 9. 上下文窗口管理

| 功能 | 状态 | 说明 |
|------|------|------|
| **Token 估算** | ✅ | 中文字符/1.5 + ASCII/4 启发式估算 |
| **模型感知阈值** | ✅ | 根据模型名自动匹配上下文窗口大小（DeepSeek 32K/128K, GPT-4o 128K, Claude 200K 等） |
| **安全裁剪** | ✅ | 保留 SystemMessage + 最近 N 轮，不切断 tool_call ↔ tool_result 配对 |
| **LLM 摘要注入** | ✅ | 被裁剪的旧消息经 LLM 压缩为摘要，注入系统提示 |
| **降级策略** | ✅ | LLM 摘要失败时自动降级为文本截断摘要 |
| **实时显示** | ✅ | 进度条显示当前 token 用量，颜色随用量变化（绿→黄→红） |

### 10. 界面与体验

| 功能 | 状态 | 说明 |
|------|------|------|
| **响应式布局** | ✅ | 侧边栏 + 主内容区，适配不同屏幕尺寸 |
| **深色/浅色主题** | ✅ | CSS 变量驱动，一键切换 |
| **引导配置界面** | ✅ | 未配置 API 时显示专业引导卡片（步骤说明 + 品牌色图标） |
| **上下文指示器** | ✅ | 模型栏显示当前上下文 token 使用量进度条 |
| **Toast 通知** | ✅ | 重试/中断/循环/上下文裁剪等状态实时通知 |
| **错误友好提示** | ✅ | API 配置错误显示精致内联卡片（🔑 图标 + 前往设置按钮） |
| **模型栏信息** | ✅ | 显示当前 Provider · Model，快速跳转设置页 |
| **快捷指令** | ✅ | 侧边栏可配置的常用快捷指令 |
| **移动端适配** | 🔜 | 抽屉侧边栏 + 触摸手势（规划中） |

---

## 🔧 工具列表

Agent 可调用的全部工具：

| 工具 | 功能 | 备注 |
|------|------|------|
| `get_time` | 获取当前时间和日期 | |
| `execute_command` | 执行本地 Shell 命令 | 有安全校验 |
| `read_file` | 读取文件内容 | 支持大文件分段 |
| `write_file` | 写入文件 | 路径白名单校验 |
| `list_processes` | 列出系统进程 | Windows/Linux 适配 |
| `kill_process` | 终止进程 | |
| `list_services` | 列出系统服务 | |
| `network_check` | 网络连通性测试 | ping / traceroute |
| `system_stats` | 系统资源统计 | CPU / 内存 / 磁盘 |
| `search_log` | 日志文件搜索 | 正则表达式支持 |
| `list_dir` | 列出目录内容 | |
| `search_kb` | 知识库语义搜索 | TF-IDF 向量检索 |
| `list_hosts` | 列出远程主机列表 | |
| `execute_on_host` | 在指定主机执行命令 | SSH |
| `run_on_all_hosts` | 批量执行命令到所有主机 | 并行 SSH |
| `check_host_health` | 检测主机健康状态 | SSH 可达性 |
| `excel_summary` | Excel 文件统计摘要 | 列统计、行数 |
| `excel_filter` | Excel 数据过滤 | 条件筛选 |
| `excel_aggregate` | Excel 聚合计算 | 分组统计 |
| `excel_chart` | Excel 数据图表 | 导出图片到 charts/ |

---

## 📡 API 参考

| 方法 | 路径 | 说明 |
|------|------|------|
| **GET** | `/` | 聊天主页 |
| **GET** | `/settings` | 设置页 |
| **GET** | `/settings/providers` | 获取支持的提供商列表 |
| **GET** | `/settings/load` | 加载活跃配置 |
| **GET** | `/settings/configs` | 获取所有配置列表 |
| **POST** | `/settings/configs` | 新建配置 |
| **PUT** | `/settings/configs/<id>` | 更新配置 |
| **DELETE** | `/settings/configs/<id>` | 删除配置 |
| **PATCH** | `/settings/configs/<id>/activate` | 激活配置 |
| **POST** | `/settings/save` | 保存旧格式配置（兼容） |
| **POST** | `/settings/test` | 测试 API Key 连通性 |
| **GET** | `/audit` | 审计日志页 |
| **GET** | `/audit/list` | 获取审计日志列表 |
| **POST** | `/audit/clear` | 清空审计日志 |
| **POST** | `/kb/save-suggestion` | 保存 AI 问答建议 |
| **GET** | `/kb` | 知识库页 |
| **GET** | `/kb/list` | 获取知识库条目 |
| **POST** | `/kb/save` | 保存 Q&A 条目 |
| **POST** | `/kb/delete` | 删除 Q&A 条目 |
| **GET** | `/kb/search` | 知识库搜索 |
| **POST** | `/kb/index/rebuild` | 重建语义索引 |
| **POST** | `/kb/import` | 导入文档（PDF/DOCX） |
| **GET** | `/hosts` | 远程主机页 |
| **GET** | `/hosts/list` | 获取主机列表 |
| **POST** | `/hosts/save` | 保存主机 |
| **POST** | `/hosts/delete` | 删除主机 |
| **POST** | `/hosts/test` | 测试主机连通性 |
| **GET** | `/file/chart/<filename>` | 获取导出的图表图片 |
| **POST** | `/chat/upload-file` | 上传聊天附件（图片/文件） |
| **POST** | `/chat` | 发送聊天消息（非流式） |
| **POST** | `/chat/stream` | 发送聊天消息（SSE 流式） |
| **POST** | `/stop` | 停止当前 Agent 任务 |
| **POST** | `/reset` | 重置对话 |
| **GET** | `/health` | 健康检查 |
| **POST** | `/compact` | 手动触发检查点库压缩（返回 size_before_mb / size_after_mb / freed_mb） |

---

## ⚙️ 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FLASK_PORT` | `5000` | Flask 服务端口 |
| `API_TOKEN` | `dsh-admin` | API 认证 Token |
| `LLM_SETTINGS_FILE` | `<app_dir>/llm_settings.json` | LLM 配置文件路径 |
| `LOG_DIR` | `<app_dir>/logs` | 日志目录 |
| `LOG_FILE` | `agent.log` | 日志文件名 |
| `CHECKPOINT_DB` | `<app_dir>/checkpoints.sqlite` | 对话检查点数据库 |
| `MAX_AGENT_RETRIES` | `5` | API 重试最大次数 |
| `RETRY_WAIT_BASE` | `2` | 重试初始等待秒数 |
| `RETRY_WAIT_MAX` | `30` | 重试最大等待秒数 |
| `MAX_AGENT_ITERATIONS` | `50` | LangGraph 递归迭代上限 |
| `MAX_TOOL_CALLS_PER_TURN` | `30` | 单轮最大工具调用次数 |
| `MAX_SAME_TOOL_REPEATS` | `5` | 相同工具+参数重复拦截阈值 |
| `CONTEXT_TRIM_RATIO` | `0.7` | 上下文窗口裁剪触发比例 |
| `CONTEXT_KEEP_RECENT_TURNS` | `8` | 保留最近对话轮数 |
| `CONTEXT_SUMMARY_DISABLED` | `0` | 设为 `1` 禁用 LLM 摘要 |
| `CHART_DIR` | `<log_dir>/charts` | 导出的图表图片目录 |
| `KB_IMAGE_DIR` | `<log_dir>/kb_images` | 知识库导入时提取的图片目录 |

---

## 🚀 快速启动

### 方式一：桌面应用（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 打包为桌面 exe
python build_exe.py

# 3. 双击生成的 DSH.exe 运行
#    - 自动启动 Flask 服务 (http://127.0.0.1:5000)
#    - 自动打开浏览器
#    - 配置保存在 exe 同目录
```

### 方式二：开发模式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置环境变量（可选，有默认值）
set FLASK_PORT=5000
set API_TOKEN=dsh-admin

# 3. 启动开发服务器
python app.py

# 4. 浏览器访问 http://127.0.0.1:5000
#    首次使用请先在 ⚙️ 设置 页配置 API Key
```

### 方式三：直接运行

```bash
python app.py

# 指定端口（通过环境变量）
set FLASK_PORT=8080
python app.py
```

> 运行后访问 `http://127.0.0.1:<端口>`，首次使用请先在 ⚙️ 设置 页配置 API Key。

---

## 📁 项目结构

```
dsh/
├── app.py                 # 主应用：Flask 路由 + Agent 调度 + SSE 流
├── config.py              # 配置管理：多 API 配置、上下文窗口
├── dsh_desktop.py         # 桌面启动器（PyInstaller 打包入口）
├── build_exe.py           # PyInstaller 打包脚本
├── models/
│   ├── agent_model.py     # LangGraph Agent 创建 + LLM 初始化
│   └── tools.py           # 20+ 内置工具实现
├── services/
│   ├── audit_service.py   # 审计日志服务
│   ├── document_parser.py # 文档解析（PDF/DOCX → Q&A）
│   ├── embedding_service.py # 语义嵌入（TF-IDF 向量）
│   ├── host_service.py    # 远程主机管理（SSH）
│   ├── kb_service.py      # 知识库服务
│   └── logger_service.py  # 日志服务
├── templates/
│   ├── index.html         # 聊天主页（JS 单页应用）
│   ├── audit.html         # 审计日志页
│   ├── hosts.html         # 远程主机页
│   ├── kb.html            # 知识库页
│   └── settings.html      # 设置页（多配置管理）
└── data/
    ├── llm_settings.json  # LLM 多配置存储
    ├── checkpoints.sqlite # 对话检查点（LangGraph）
    ├── kb.sqlite          # 知识库 + 语义索引
    └── logs/agent.log     # Agent 运行日志
```

---

## 🔐 安全说明

> ⚠️ **本地部署**：当前设计为本地桌面应用，默认运行在 `127.0.0.1:5000`，通过 `API_TOKEN` 环境变量做简单认证。请勿将服务暴露到公网。

> **生产部署建议**：如需生产环境使用，请务必：
> 1. 配置 HTTPS 反向代理（nginx / Caddy）
> 2. 设置强 `API_TOKEN`
> 3. 限制文件读写路径白名单
> 4. 对危险命令增加二次确认

---

## 🔄 更新日志

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | — | 初始发布：ReAct Agent + 多模型配置 + 知识库 + 远程主机 |
| v1.1 | — | API 自动重试（tenacity）+ 循环检测 + 手动中断 |
| v1.2 | — | AI 知识沉淀 + 语义搜索 + 聊天文件上传 |
| v1.3 | — | 引导配置界面（harness 风格）+ 设置页美化 |
| v1.4 | — | 上下文窗口管理（自动裁剪 + LLM 摘要） |

---

## 📄 License

MIT

---

> **维护说明**：添加新功能时，请在上方功能清单中新增条目，并在更新日志中记录版本和日期。
> 新增环境变量时，在「环境变量配置」表中追加行。
> 新增 API 端点时，在「API 参考」表中追加行。
