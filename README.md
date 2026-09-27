# 《算法设计与分析》课程智能助教（RAG）

基于 RAG（Retrieval-Augmented Generation）的课程智能助教系统，针对《算法设计与分析》课程，支持可溯源问答、自动出题与学习引导。系统构建了多格式文档解析与 OCR 管线实现课程资料的统一建模，采用 Hybrid Retrieval（稠密检索 + BM25 + RRF）混合检索策略提升复杂查询的召回效果，并提供流式 Web 交互前端。

## 特性

- **多格式文档解析**：支持 PDF / PPTX / DOCX / TXT，PDF 与 PPT 中的图片走 Tesseract OCR 提取文字。
- **统一文本切分**：按页 / 幻灯片分割，DOCX / TXT 按句子边界滑动窗口切分（chunk_size=500, overlap=100）。
- **Hybrid Retrieval**：稠密向量检索（OpenAI 兼容 embedding）+ BM25 稀疏检索（jieba + nltk），用 RRF（Reciprocal Rank Fusion）融合两路结果。
- **可溯源问答**：每个检索片段携带 `filename` + `page_number` 元数据，系统提示词强制引用资料页码，降低幻觉。
- **流式响应**：FastAPI `/chat_stream` SSE 流式返回，前端逐字渲染 Markdown + MathJax 公式。
- **课程助教人设**：内置详细系统提示词，区分知识点讲解、习题引导、自动出题、资料外扩展四类回答逻辑。

## 项目结构

```
Proj2/
├── app.py                    # FastAPI Web 入口，提供 /chat 与 /chat_stream
├── main.py                   # CLI 交互入口
├── rag_agent.py              # RAGAgent：检索 + 提示词 + 流式 / 非流式生成
├── vector_store.py           # ChromaDB 向量库 + Dense / BM25 / Hybrid 检索
├── document_loader.py        # 多格式文档加载与 OCR
├── text_splitter.py          # 文本切分（句子边界 + 滑动窗口）
├── process_data.py           # 数据预处理脚本：加载 → 切分 → 入库
├── download_nltk_punkt.py    # 一次性下载 NLTK punkt / punkt_tab 数据
├── config.py                 # 配置：API、数据路径、检索参数（环境变量优先）
├── requirements.txt          # Python 依赖
├── data/                      # 课程资料（PDF）
│   ├── LectureNotes.pdf
│   ├── Solutions_for_Algorithms_AI2615.pdf
│   └── Slides01-Introduction.pdf ~ Slides22-More-NP-Complete-Problems.pdf
├── vector_db/                 # ChromaDB 持久化目录（已索引 1687 条）
└── frontend/
    └── index.html            # 前端：SSE 流式渲染 + Markdown + MathJax
```

## 环境要求

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) （仅在重新处理 `data/` 中的文档时需要；已有向量库可跳过）
- 阿里云百炼（DashScope / MaaS）API Key，模型使用 `qwen3-max` + `text-embedding-v3`

## 安装

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 下载 NLTK 分词数据（首次运行必需，仅一次）
python download_nltk_punkt.py

# 3. 安装 Tesseract OCR（可选，仅在重新处理含图片的 PDF / PPT 时需要）
#    Windows:  从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装包
#              安装时勾选中文语言包（chi_sim），并将安装目录加入 PATH
#    macOS:    brew install tesseract tesseract-lang
#    Ubuntu:   sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```

## 配置 API Key

本项目**不在代码中硬编码 API Key**，启动时优先读环境变量，未设置则在终端交互式提示输入。

### 方式 A：环境变量（推荐，可持久化）

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="sk-你的key"
$env:OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-你的key"
export OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

> 如果使用千问 MaaS 平台（`maas.qianwenapi.com`）创建的 key，请把 `OPENAI_API_BASE` 换成 `https://maas.qianwenapi.com/compatible-mode/v1`。

### 方式 B：启动时交互输入

不设置环境变量直接运行 `python main.py` 或 `uvicorn app:app --port 8000`，程序会提示：

```
============================================================
检测到尚未配置 OPENAI_API_KEY。
...
请输入 OPENAI_API_KEY:
============================================================
```

输入的 key 仅对本次运行有效，不会写入任何文件。

## 运行

### 步骤 1：数据预处理（首次运行或资料更新时）

```bash
python process_data.py
```

这一步会：加载 `data/` 下所有 PDF / PPTX / DOCX / TXT → 切分 → 调用 embedding API → 写入 `vector_db/`。

> `vector_db/` 已在 `.gitignore` 中排除，**首次使用必须先执行此步骤**生成向量库，之后才能进入步骤 2。

### 步骤 2a：CLI 问答

```bash
python main.py
```

进入交互式命令行，直接输入问题即可。

### 步骤 2b：Web 问答（推荐）

```bash
uvicorn app:app --reload --port 8000
```

浏览器访问 **http://127.0.0.1:8000/** 即可使用前端页面问答（前端 SSE 流式渲染，支持 Markdown 与 LaTeX 数学公式）。

## 技术栈

| 层 | 选型 |
|---|---|
| LLM | qwen3-max（阿里云百炼，OpenAI 兼容接口） |
| Embedding | text-embedding-v3 |
| 向量数据库 | ChromaDB（PersistentClient） |
| 稀疏检索 | 自实现 BM25（jieba 中文分词 + nltk 英文分词） |
| 混合融合 | RRF（Reciprocal Rank Fusion, k=60） |
| 文档解析 | PyPDF2 + PyMuPDF + python-pptx + docx2txt + pytesseract |
| 后端 | FastAPI + Uvicorn（SSE 流式响应） |
| 前端 | 原生 HTML / JS + marked.js + MathJax |

## 检索与生成流程

```
用户问题
   │
   ├──────────────────┬──────────────────────┐
   ▼                  ▼                      │
Dense 检索        BM25 检索                  │
(OpenAI emb)     (jieba + nltk)             │
   │                  │                      │
   └──────┬───────────┘                      │
          ▼                                  │
       RRF 融合                              │
          │                                  │
          ▼                                  │
   Top-K 文档片段（含 filename + page_number）│
          │                                  │
          ▼                                  │
   构造 Prompt（系统提示词 + 资料 + 问题） ◄─┘
          │
          ▼
   qwen3-max 流式生成
          │
          ▼
   前端 SSE 逐字渲染
```

## 注意事项

- `config.py` 中所有非敏感参数（chunk_size、TOP_K、BM25 参数、RRF_K 等）可按需调整。
- 若 DashScope 返回 `AllocationQuota.FreeTierOnly`，说明免费额度已用完，需要在阿里云控制台开通按量付费或充值。
- `vector_db/` 不纳入版本控制，由 `process_data.py` 生成；若重新运行会先清空再写入。
- Tesseract OCR 失败不会中断主流程，对应图片的 OCR 文本会被静默跳过。

## License

本项目仅用于课程教学目的。
