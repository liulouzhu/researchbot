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
