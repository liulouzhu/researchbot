# ResearchBot - 智能科研助手

基于 nanobot 改进的智能科研助手，支持论文检索、阅读理解、知识管理和文献综述生成。

## 核心功能

### 1. 论文智能处理
- **论文检索** (`paper_search`): 通过学术数据库检索相关论文
- **论文摘要** (`paper_summarize`): 生成论文摘要和要点总结
- **论文比较** (`paper_compare`): 对多篇论文进行对比分析
- **文献综述** (`paper_review`): 基于主题生成相关工作综述
- **PDF下载** (`paper_pdf`): 自动下载论文PDF
- **论文增强** (`paper_enrich`): 使用 Crossref/OpenAlex 丰富论文元数据
- **Crossref搜索** (`crossref_search`): 直接搜索 Crossref 数据库
- **OpenAlex搜索** (`openalex_search`): 直接搜索 OpenAlex 数据库
- **引文导出** (`paper_cite`): 将论文导出为 BibTeX、RIS、CSL-JSON、APA、MLA、GB/T 7714 格式

### 2. 智能代理
- 支持多种大语言模型 (OpenAI GPT, Anthropic Claude, 本地 Ollama 等)
- 长期记忆管理
- 工具调用和代码执行
- 上下文理解

### 3. 多渠道接入
- **命令行界面**: 直接在终端与助手交互
- **Telegram Bot**: 通过 Telegram 与助手对话
- **微信**: 支持企业微信机器人

### 4. 定时任务
- 支持 Cron 定时任务
- 自动化的定期研究和报告生成

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/liulouzhu/researchbot.git
cd researchbot

# 安装依赖  
pip install -e .

# 或者使用 uv (推荐)
uv pip install -e .
```

### 配置

首次运行需要配置 API 密钥：

```bash
# 初始化配置
researchbot onboard

# 或者手动编辑配置文件
vim ~/.researchbot/config.json
```

配置文件示例：

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "provider": "auto"
    }
  },
  "providers": {
    "anthropic": {
      "apiKey": "your-api-key"
    }
  },
  "literature": {
    "crossref": {
      "mailto": "your-email@example.com"
    },
    "openalex": {
      "apiKey": "your-openalex-api-key"
    }
  }
}
```

### Crossref / OpenAlex 配置

Crossref 和 OpenAlex 是论文元数据来源，可以增强现有论文记录。

**Crossref 配置：**
- `mailto`: 用于 Crossref polite pool（推荐提供）
- `userAgent`: 自定义 User-Agent
- `apiBase`: API 基础地址（默认 https://api.crossref.org）

**OpenAlex 配置：**
- `apiKey`: API 密钥（2026-02-13 起必须使用）
- `apiBase`: API 基础地址（默认 https://api.openalex.org）

环境变量方式：
```bash
export RESEARCHBOT_LITERATURE__CROSSREF__MAILTO="your-email@example.com"
export RESEARCHBOT_LITERATURE__OPENALEX__API_KEY="your-openalex-api-key"
```

### 本地语义检索

对本地保存的论文进行语义搜索，结合关键词和向量相似度。

#### 快速配置

**1. 安装 sqlite-vec（如需向量检索）：**
```bash
uv pip install sqlite-vec
```

**2. 配置（config.json）：**
```json
{
  "literature": {
    "semanticSearch": {
      "sqliteDbPath": "literature/indexes/search.sqlite3",
      "embeddingProvider": "dashscope",
      "embeddingApiKey": "your-api-key",
      "embeddingModel": "text-embedding-v4",
      "embeddingDimension": 1024
    }
  }
}
```

#### 使用

```bash
# 搜索本地论文
researchbot paper search-local "机器学习安全"

# 手动重建索引
researchbot paper index --rebuild
```

论文会在保存时自动索引。

### 使用

#### 命令行对话

```bash
# 启动交互式对话
researchbot agent -m "帮我查找关于大语言模型安全性的最新研究"

# 指定配置文件
researchbot agent -m "Hello" --config /path/to/config.json

# 指定工作目录
researchbot agent -m "Hello" --workspace /path/to/workspace
```

#### 论文工具

```bash
# 搜索论文
researchbot paper search "large language model security"

# 总结论文
researchbot paper summarize "https://arxiv.org/abs/1234.5678"

# 比较论文
researchbot paper compare "paper1.pdf" "paper2.pdf"

# 生成文献综述
researchbot paper review "大语言模型安全"
```

#### 创新点工作流 (`innovation_workflow`)

`innovation_workflow` 是科研创新点发现工具，从研究主题出发，经历"生成候选 → 查新评估 → 审阅评分 → 多轮迭代收敛"四个阶段，帮助从粗想法逐步收敛为有证据支撑的可行创新方向。

**基础用法：**
```bash
# 生成 5 个关于 LLM 安全性的创新点候选
researchbot agent -m "innovation_workflow topic=\"LLM security\" num_candidates=5 top_k=3"

# 开启多轮迭代模式
researchbot agent -m "innovation_workflow topic=\"LLM security\" enable_iteration=True max_rounds=3"

# 覆盖已有结果重新运行
researchbot agent -m "innovation_workflow topic=\"LLM security\" overwrite=True"
```

> 如果你想要的是“再来一批不一样的”，请明确传 `overwrite=True`。默认情况下同一主题会复用已有结果，避免重复计算。

**四阶段工作流：**

| 阶段 | 说明 |
|------|------|
| Stage 1 | 根据主题和本地文献上下文生成 3-8 个创新点候选 |
| Stage 2 | 对每个候选进行本地 + 线上（arXiv）查新，分析新颖性 |
| Stage 3 | 基于证据对每个候选评分（novelty/feasibility/evidence/impact/risk）并给出决策 |
| Stage 4（可选） | 对"revise"候选进行多轮修订和再审阅，逐步收敛 |

**输出文件结构：**
```
innovation/<topic_slug>/
├── workflow.json           # 工作流元数据（版本、参数、阶段状态）
├── candidates.json / .md    # 初始候选列表
├── novelty_report.json / .md
├── review_report.json / .md
├── iterations/
│   ├── round_0/
│   ├── round_1/            # （如有迭代）
│   └── round_N/
├── iteration_report.json     # 迭代汇总（含各轮统计和最终推荐）
└── iteration_report.md     # 人类可读的迭代总结
```

**迭代模式参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_iteration` | `False` | 是否开启多轮迭代（需 `enable_review=True`） |
| `max_rounds` | `2` | 最大迭代轮数（1-5） |
| `min_proceed` | `1` | 达到该数量 `proceed` 候选后提前停止 |
| `revise_top_k` | `3` | 每轮最多修订的候选数量 |
| `stop_if_no_change` | `True` | 本轮无新增 `proceed` 时停止 |

**推荐评分与决策规则：**

评分维度：novelty、feasibility、evidence、impact（各 1-10）；risk（1-10，越高风险越大）。

综合得分：`overall = novelty×0.20 + feasibility×0.25 + evidence×0.15 + impact×0.20 + (10-risk)×0.20`

决策：
- `proceed`：evidence≥6 且 feasibility≥5 且 risk≤4
- `revise`：有潜力但需强化
- `drop`：feasibility≤3 或 evidence≤3 或 risk≥7 或 is_duplicate=true

**迭代停止条件（任一满足即停）：**
- 达到 `max_rounds`
- 累计 `proceed` 候选数 ≥ `min_proceed`
- 本轮无新增 `proceed` 且 `stop_if_no_change=True`
- 无更多可修订的候选

**创新线去重：** 迭代模式下，同一 `parent_candidate`（或原始 title）的多个修订版本会按 lineage 去重，仅保留得分最高者，确保 top-K 推荐覆盖不同创新线。

#### 引文导出

支持 6 种引文格式导出：**BibTeX**、**RIS**、**CSL-JSON**、**APA**、**MLA**、**GB/T 7714**。

```bash
# 在对话中导出单篇论文 BibTeX
paper_cite paper_id="2401.12345" format="bibtex"

# 导出多篇论文为 RIS
paper_cite paper_ids=["2401.12345", "2301.45678"] format="ris"

# 导出全部本地论文为 GB/T 7714 格式
paper_cite format="gbt7714"

# 导出到文件
paper_cite paper_id="2401.12345" format="bibtex" output="file" path="refs.bib"
```

**支持的格式：**
| 格式 | 说明 | 典型用途 |
|------|------|---------|
| `bibtex` | BibTeX 条目 | LaTeX 论文 |
| `ris` | RIS 格式 | EndNote, Zotero, Mendeley |
| `csl-json` | CSL-JSON 对象 | Pandoc, Citeproc |
| `apa` | APA 第 7 版 | 社会科学论文 |
| `mla` | MLA 第 9 版 | 人文学科论文 |
| `gbt7714` | GB/T 7714-2015 | 中文论文 |

**Citekey 规则：** `第一作者姓 + 年份 + 标题关键词`（如 `vaswani2017attention`），DOI 后缀用于去重。

#### 启动网关服务

```bash
# 启动 API 服务
researchbot gateway

# 指定端口
researchbot gateway --port 18799

# 指定工作目录
researchbot gateway --workspace /path/to/workspace
```

## 项目结构

```
researchbot/
├── cli/                  # 命令行界面
├── agent/               # 智能代理核心
│   ├── loop.py         # Agent 循环
│   ├── context.py      # 上下文管理
│   └── tools/          # 内置工具
├── config/             # 配置管理
├── providers/          # LLM 提供商适配器
├── channels/           # 消息渠道集成
├── session/            # 会话管理
├── cron/               # 定时任务
└── templates/          # 工作区模板
```

## 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `RESEARCHBOT_CONFIG` | 配置文件路径 | `~/.researchbot/config.json` |
| `RESEARCHBOT_MAX_CONCURRENT_REQUESTS` | 最大并发请求数 | `3` |


## 配置文件路径

- 新版配置: `~/.researchbot/config.json`


## 致谢

ResearchBot 基于 [nanobot](https://github.com/HKUDS/nanobot) 构建。
