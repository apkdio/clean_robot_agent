# 扫地机器人智能客服（clean_robot_agent）

基于 **RAG（检索增强生成）** 与 **本地大模型** 的扫地机器人智能客服系统。支持自然语言问答、预算内产品推荐、故障排查、使用维护等场景，全部模型本地化部署，无需云端 API。

## 核心特性

- **双路召回**：稠密检索（bge-m3 向量）+ 稀疏检索（BM25 关键词）→ RRF 融合，兼顾语义近似与精确关键词匹配
- **意图分类**：本地深度学习分类头（bge-m3 embedding + Linear 分类器）前置判断用户意图，毫秒级推理
- **结构化查询**：识别"预算 1000 以内"等价格约束，通过 Chroma metadata 过滤精确枚举预算内产品
- **工具调用**：LLM function calling，内置日期计算工具，支持"最近半年""2025年三月"等时间范围新品查询
- **多轮 SOP 引导**：选购推荐、故障排查等场景按标准流程多轮引导，支持追问（比较新/更便宜）与最近发布查询
- **流式输出**：SSE 流式返回答案，前端逐字渲染
- **知识库热更新**：每 30 分钟自动扫描 `data/knowledge/`，检测文件增删改并增量入库
- **多轮友好**：领域外问题礼貌拒答，模糊问题软引导，闲聊自然回应

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Flask + 原生 HTML/CSS/JS |
| 检索 | Chroma（向量库）+ BM25（rank-bm25）+ RRF 融合 |
| Embedding | bge-m3（本地 Ollama） |
| 生成模型 | qwen2.5:7b（本地 Ollama） |
| 意图分类 | bge-m3 + PyTorch 分类头（本地推理） |
| 文档解析 | pypdf + Docling + 标准库多策略 |

## 目录结构

```
clean_robot_agent/
├── config/                      # 配置文件（实际配置 + _template 模板）
│   ├── agent.yaml               # Agent/LLM 配置（含模型名、温度）
│   ├── rag.yaml                 # RAG 配置（分块、检索、RRF 参数）
│   ├── chroma.yaml              # Chroma/embedding 配置
│   ├── prompts.yaml             # Prompt 文件路径
│   └── *_template.yaml          # 对应模板（含注释说明，复制后填值）
├── data/
│   ├── knowledge/               # 知识库源文件（txt/pdf，热更新监控目录）
│   ├── datasets/                # 意图分类训练数据集（JSONL）
│   ├── bgm_model/               # 训练好的分类头模型
│   ├── pkl/                     # BM25 pickle 缓存
│   ├── state/                   # 热更新指纹快照
│   └── vector_store/            # Chroma 持久化向量库
├── prompts/                     # Prompt 模板（system/摘要/报告）
├── tools/                       # 核心工具模块（详见下节）
├── function_tools/              # LLM 工具调用（function calling）工具
│   └── date_tool.py             # 日期计算工具（绝对/相对日期 → 日期范围）
├── sops/                        # SOP 标准操作流程（多轮引导）
│   ├── base.py                  # 会话状态 + 执行器
│   ├── purchase.py              # 选购推荐 SOP
│   └── repair.py                # 故障排查 SOP
├── intent_classifier_training/  # 意图分类模型训练工具
│   ├── build_intent_dataset.py  # 数据集构建（从知识库抽取 + 规则改写）
│   └── train_intent_classifier.py # 分类头训练脚本
├── webapp/
│   ├── app.py                   # Flask 后端（上传/问答/流式接口）
│   └── templates/index.html     # 聊天前端
├── requirements.txt
└── .gitignore
```

## 核心模块说明（tools）

| 模块 | 职责 |
|------|------|
| `agent.py` | RAG 编排：意图路由 → 检索 → 结构化直出或 LLM 生成 |
| `intent_router.py` | 意图分类：本地分类头（bge-m3 + Linear）判 robot/casual/other/unknown |
| `hybrid_retriever.py` | 双路召回编排：dense + sparse → RRF 融合 |
| `vector_store.py` | Chroma 稠密检索、入库、metadata 过滤、稀疏索引构建 |
| `sparse_retriever.py` | BM25 关键词检索（含 pickle 持久化缓存） |
| `rrf_fusion.py` | RRF（倒数排名融合）算法 |
| `metadata_extractor.py` | 结构化元数据：价格提取、预算解析、发布时间提取、型号信息抽取 |
| `entry_splitter.py` | 编号条目分块器（每个问答/型号一个 chunk） |
| `hot_ingest.py` | 知识库热更新：定时扫描 + 增量入库 |
| `file_tools.py` | 文档多策略解析（txt/pdf/csv/docx） |
| `llm_tool.py` | LLM / embedding 工厂（含 function calling） |
| `log_tool.py` | 日志（控制台彩色 + 文件） |
| `config_tool.py` / `path_tool.py` / `prompts_tool.py` | 配置 / 路径 / Prompt 加载 |

## 工具调用模块（function_tools/）

| 模块 | 职责 |
|------|------|
| `date_tool.py` | 日期计算工具：`calc_date_range` 把"最近半年""2025年三月"等表达换算成日期范围 |

## SOP 模块（sops/）

| 模块 | 职责 |
|------|------|
| `base.py` | SOP 基础设施：会话状态 + 执行器（ask/action/reply 三步式状态机）+ 追问处理（比较新/更便宜）+ 最近发布查询 |
| `purchase.py` | 选购推荐 SOP：收集预算（上限/区间）+ 宠物 → 结构化推荐 |
| `repair.py` | 故障排查 SOP：问现象 → 检索 → LLM 生成排查步骤 |

## 快速开始

### 1. 环境要求

- Python 3.13+
- [Ollama](https://ollama.com)（本地已运行）
- 内存 ≥ 16GB（三个模型常驻：bge-m3 + qwen2.5:3b + qwen2.5:7b）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 拉取模型

```bash
ollama pull bge-m3        # embedding 模型
ollama pull qwen2.5:7b    # 生成模型
ollama pull qwen2.5:3b    # 意图分类兜底（可选）
```

### 4. 配置

复制模板为实际配置，填入你的参数：

```bash
cp config/agent_template.yaml  config/agent.yaml
cp config/rag_template.yaml    config/rag.yaml
cp config/chroma_template.yaml config/chroma.yaml
cp config/prompts_template.yaml config/prompts.yaml
```

### 5. 启动

```bash
python webapp/app.py
```

浏览器打开 `http://localhost:5050`。

## 知识库管理

知识库源文件放在 `data/knowledge/` 下，支持 `.txt/.md/.pdf/.csv/.docx/.pptx/.xlsx`。

**文档格式规范**（详见 `data/knowledge_example/格式模板.txt`）：

```
## 入门级（1500 元以下）
1. **米家扫拖机器人 M20**
   - 吸力：2800Pa｜导航：LDS激光｜避障：红外
   - 参考价：899
```

核心规则：
- 每个条目以 `数字. ` 开头
- 型号名/问题用 `**加粗**`
- 参数用 `｜` 分隔，值内不能有空格
- 价格独立一行 `参考价：数字`
- 发布时间独立一行 `发布时间：YYYY-MM-DD`（用于"最近半年""2025年三月"等新品查询）

**热更新**：放入新文件后，系统每 30 分钟自动检测并入库；重启 Flask 会立即触发一次扫描。

## 意图分类模型训练

系统默认用本地分类头做意图分类（速度快），也保留了 3b 模型兜底。

**重新训练**（当需要扩充数据集时）：

```bash
# 1. 构建数据集（从知识库抽取 robot 样本 + 内置 other/casual/unknown）
python intent_classifier_training/build_intent_dataset.py

# 2. 训练分类头（bge-m3 embedding + Linear 分类器）
python intent_classifier_training/train_intent_classifier.py
```

训练产物输出到 `data/bgm_model/`，推理时由 `intent_router.py` 自动加载。

## 配置说明

| 配置 | 关键项 |
|------|--------|
| `agent.yaml` | `llm.model`（生成模型）、`behavior.retrieval_only`（纯检索模式开关） |
| `rag.yaml` | `chunk.chunk_size`、`retrieval.dense_top_k/sparse_top_k/final_top_k`、`rrf.*` |
| `chroma.yaml` | `persist_dir`、`collection_name`、`embedding.model` |

## 问答流程

```
用户提问
  → 提示词注入检测（命中直接拒绝）
  → SOP 会话检查（有活跃 SOP → 继续多轮引导）
  → intent_router（本地分类头）
      ├─ other（领域外）→ 礼貌拒答
      ├─ casual（闲聊）→ 自然回应
      ├─ unknown（模糊）→ 软引导 + RAG
      └─ robot（领域内）
          ├─ 选购意图且无预算 → 进入选购 SOP 多轮引导
          ├─ 含日期 → 日期工具（规则解析 / LLM function calling）→ 日期范围
          ├─ 含预算 → metadata 过滤 → 结构化直出型号列表
          └─ 其他 → 双路召回（dense+sparse→RRF）→ LLM 生成
  → 流式输出（SSE）
```

## 注意事项

- 所有模型本地运行，无云端依赖
- `data/vector_store/`、`data/pkl/`、`data/state/`、`data/bgm_model/` 为运行时产物，已加入 `.gitignore`
- 配置文件 `config/*.yaml`（非 template）含本地环境信息，已加入 `.gitignore`
