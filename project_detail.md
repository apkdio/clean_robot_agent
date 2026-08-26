# 扫地机器人智能客服 · 项目设计文档

> 本文档面向新加入的开发者 / AI 模型，帮助快速理解项目架构、关键设计决策与已知问题。
> 阅读顺序建议：项目概述 → 架构总览 → 模块详解 → 技术决策 → 已知问题 → 未来规划。

---

## 一、项目概述

### 1.1 定位

一个**完全本地化部署**的扫地机器人智能客服系统。用户以自然语言提问，系统通过「意图识别 + 检索增强生成（RAG）」给出回答。

### 1.2 核心能力

| 能力        | 说明                                                          |
|-----------|-------------------------------------------------------------|
| 领域问答      | 扫地机器人使用、故障、维护、选购等问答                                         |
| 预算推荐      | 识别"预算 1000 以内"等约束，精确枚举预算内产品                                 |
| 工具调用      | LLM function calling，内置日期/预算/故障分类工具，支持"最近半年""一千来块"等口语化结构化提取 |
| 多轮 SOP 引导 | 选购推荐等场景按标准流程逐步收集需求（预算/宠物），多轮引导，进入时给开场提示                     |
| 知识库分域     | 按场景（品牌/选购/型号/故障/售后/维护 6 域）定向检索对应知识域，避免跨域词带偏召回              |
| 情绪安抚      | 识别负面情绪（投诉/烦躁等）前置安抚，只安抚不拦截                                   |
| 价格极值查询    | 识别"最贵/最便宜"并全局按价格排序取极值，避免走 RAG 幻觉                            |
| 安全前置      | 识别危险现象（冒烟/烧焦/进水等）前置拦截，立即停机联系售后                               |
| 品牌咨询      | 识别"为什么买/优势/介绍"等品牌咨询走 RAG 品牌介绍域，不触发选购 SOP                      |
| 售后咨询      | 识别"报修/更换零件/保修"等售后咨询走 RAG 售后服务域，不触发维修 SOP                     |
| 领域边界      | 识别非扫地机器人问题（电动车、手机等）并礼貌拒答                                    |
| 闲聊兜底      | 问候、道谢自然回应；模糊问题软引导                                           |
| 流式输出      | 前端逐字渲染回答                                                    |
| 知识库热更新    | 每 30 分钟自动同步 `data/knowledge/` 文件变动                          |
| 多会话与上下文   | 按 session_id 隔离会话，对话持久化到本地（最近6轮），前端可新建/切换历史会话；自由指代（"它怎么样"）通过拼接历史让 LLM 自主消解 |
### 1.3 技术栈

| 层 | 选型 | 说明 |
|----|------|------|
| Web 框架 | Flask | 后端 + SSE 流式接口 |
| 向量库 | Chroma | 稠密检索持久化，支持 metadata 过滤 |
| 稀疏检索 | BM25（rank-bm25） | 关键词检索，弥补向量语义检索的精确匹配盲区 |
| Embedding | bge-m3（Ollama） | 1024 维，多语言，本地运行 |
| 生成模型 | qwen2.5:7b（Ollama） | 主回答模型 |
| 意图分类 | bge-m3 + PyTorch Linear 分类头 | 本地推理，毫秒级 |
| 会话上下文 | jsonl 文件 + 内存缓存 | 按 session_id 持久化最近 6 轮 |
| 文档解析 | pypdf + Docling + 标准库 | 多策略按扩展名分发 |

---

## 二、系统架构

### 2.1 总览

```
┌─────────────┐      ┌──────────────────────────────────────────────┐
│  前端页面    │ HTTP │              Flask 后端 (webapp/app.py)       │
│ index.html  │◄────►│  /api/chat/stream   /api/ingest  /api/reset   │
└─────────────┘ SSE  └───────────────┬──────────────────────────────┘
                                      │ 调用
                          ┌───────────▼────────────┐
                          │   agent.py（编排层）    │
                          └───────────┬────────────┘
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
   ┌──────▼──────┐          ┌─────────▼─────────┐        ┌────────▼────────┐
   │ intent_router│          │ metadata_extractor│        │  hybrid_retriever│
   │ 意图分类      │          │ 价格/预算解析      │        │  双路召回         │
   └─────────────┘          └───────────────────┘        └────────┬────────┘
                                                                  │
                                    ┌─────────────────┬───────────┘
                              ┌─────▼──────┐   ┌──────▼──────┐   ┌────────────┐
                              │ DenseRetriever│  │SparseRetriever│ │ rrf_fusion │
                              │  Chroma 向量  │   │    BM25      │   │   RRF 融合 │
                              └─────┬──────┘   └──────┬──────┘   └─────┬──────┘
                                    │                 │                │
                                    └─────────┬───────┘                │
                                              ▼                        ▼
                                      ┌───────────────┐         ┌──────────┐
                                      │  Chroma 向量库 │         │ 融合结果  │
                                      └───────────────┘         └──────────┘
```

### 2.2 问答数据流

```
用户 query
  → 提示词注入检测（_INJECT_RE，命中直接拒绝）
  → 上下文记录（context_store：记录用户消息，供 RAG 生成拼接历史）
  → 负面情绪安抚（detect_emotion，只安抚不拦截）
  → SOP 会话检查（sops/，按 session_id 隔离）
      ├─ 有活跃 SOP → 继续该 SOP（多轮引导，不经过意图路由）
      └─ 无活跃 SOP → 继续
  → intent_router.route_intent(query)
      （bge-m3 embedding → Linear(1024→4) 分类头，输出四类意图）
      ├─ other（领域外）   → 礼貌拒答，直接返回
      ├─ casual（闲聊）    → 走 LLM 自然回应
      ├─ unknown（模糊）   → 随机软引导语 + 继续 RAG
      └─ robot（领域内）   → 继续
  → SOP 触发（选购意图 + 命中 trigger + 尚未提供预算）→ 进入选购 SOP 多轮引导
  → 日期工具调用（function_tools/date_tool）
      检测时间表达（"最近半年"/"2025年三月"）→ 规则解析或 LLM function calling → 日期范围
  → metadata_extractor.build_filter(query)
      检测预算约束（"1000以内" → {"min_price": {"$lte": 1000}}）
      合并价格 + 日期 filter（$and）
      ├─ 有结构化约束 → search_by_filter 精确枚举 → 代码拼接列表直出
      └─ 无约束 → hybrid_retriever.search(query)
             ├─ DenseRetriever：query 向量化 → Chroma HNSW 检索 top-10
             ├─ SparseRetriever：BM25 关键词打分 top-10
             └─ RRF 融合两路排名 → top-8
  → LLM（qwen2.5:7b）基于检索结果生成自然语言回答
  → SSE 流式返回，前端逐字渲染
```

---

## 三、目录结构

```
clean_robot_agent/
├── config/                      # 配置（实际 + 模板）
│   ├── agent.yaml               # LLM 模型、温度、行为开关
│   ├── rag.yaml                 # 分块、检索、RRF 参数
│   ├── chroma.yaml              # Chroma 持久化、embedding 模型
│   ├── prompts.yaml             # prompt 文件路径映射
│   ├── word_dict_config.py      # 词表配置中心（情绪/退出/域/触发/症状等词表集中管理）
│   └── *_template.yaml          # 模板（含注释，git 提交；实际配置 gitignore）
├── data/
│   ├── knowledge/               # 知识库源文件（热更新监控目录）
│   ├── knowledge_example/       # 知识库格式模板示例
│   ├── datasets/                # 意图分类训练数据集
│   ├── context/                 # 对话上下文 jsonl（按会话分文件，gitignore）
│   ├── bgm_model/               # 训练好的意图分类头（gitignore）
│   ├── pkl/                     # BM25 pickle 缓存（gitignore）
│   ├── state/                   # 热更新指纹快照（gitignore）
│   └── vector_store/            # Chroma 向量库（gitignore）
├── prompts/                     # prompt 模板
│   ├── main_prompts.txt         # 主 system prompt（角色/规则/闲聊/兜底话术）
│   ├── rag_summarize_prompts.txt  # 预留：RAG 摘要模板（当前未接线）
│   └── report_prompts.txt         # 预留：报告生成模板（未接线）
├── tools/                       # 核心模块（详见第四节）
├── function_tools/              # LLM 工具调用（function calling）工具
│   └── date_tool.py             # 日期计算工具（绝对/相对日期 → 日期范围）
├── sops/                        # SOP 标准操作流程（多轮引导）
│   ├── base.py                  # 会话状态 + 执行器（ask/action/reply 步骤）
│   ├── purchase.py              # 选购推荐 SOP（预算 + 宠物槽位）
│   └── repair.py                # 故障排查 SOP（现象改写 + LLM 生成）
├── test_scripts/                # 测试脚本（mock 测试，独立于 Chroma）
│   ├── test_cases.py            # 意图分类 + 端到端泛化测试
│   ├── test_sop.py              # SOP 状态机测试
│   └── test_retrieval.py        # 知识库分域召回测试（域路由/域过滤）
├── intent_classifier_training/  # 意图分类训练工具
│   ├── build_intent_dataset.py  # 数据集构建
│   └── train_intent_classifier.py # 分类头训练
├── webapp/
│   ├── app.py                   # Flask 后端（问答/流式 + 会话 API）
│   └── templates/index.html     # 聊天前端（知识库管理 + 新建/历史对话）
├── Multi_Route_Retrieval/       # 独立 demo（双路召回原型，保留未集成）
├── temp/                        # 数据处理脚本（独立用途，不参与运行）
├── requirements.txt
├── .gitignore
└── README.md / project_detail.md
```

---

## 四、核心模块详解

### 4.1 agent.py —— 编排层

系统入口 `ask_stream(query)`，是一个生成器，按意图分流：

```python
def ask_stream(query, session_id="default"):
    # 上下文：记录用户消息（RAG 生成时拼接历史，LLM 自主消解指代）
    intent = route_intent(query)          # 意图分类
    if intent == "other":                 # 领域外拒答
        yield 拒答话术; return
    if intent == "casual":                # 闲聊
        yield LLM自然回应; return
    if intent == "unknown":               # 模糊引导
        yield 随机引导语 + "\n\n"
    # robot / unknown 继续
    metadata_filter = build_filter(query)  # 预算检测
    if metadata_filter:                    # 预算 → 结构化直出
        models = search_by_filter(...)     # 精确枚举
        yield 拼接的型号列表; return
    domain = _route_domain(query)          # 知识域路由（选购/故障/维护）
    chunks = hr.search(query,              # 双路召回（域定向，识别不准则全库）
                      filter={"file_name": domain} if domain else None)
    # 拼接 prompt → LLM 流式生成
```

**设计要点**：
- 多会话隔离：SOP 会话、推荐结果、退出确认、上下文全部按 session_id 存储，多会话互不串扰
- 意图路由前置，避免领域外问题污染检索
- 预算类问题走「结构化直出」而非 LLM 转述——LLM 会漏型号、混格式、产生幻觉，代码拼接 100% 可靠
- 知识域路由（轻量文档隔离）：按场景定向检索对应知识域，避免"吸力"等跨域词带偏召回；识别不准则全库兜底，是"提纯"而非"硬隔离"

### 4.2 intent_router.py —— 意图分类

当前实现：**本地分类头**（bge-m3 embedding + `nn.Linear(1024→4)`）。

```python
def route_intent(query):
    head, labels = _load_model()          # 懒加载分类头
    vec = embed_documents([query])[0]     # bge-m3 向量
    logits = head(vec)                    # 前向，毫秒级
    return labels[argmax(logits)]         # robot/other/casual/unknown
```

**四类意图**：

| 标签 | 含义 | 处理 |
|------|------|------|
| robot | 扫地机器人领域 | 正常检索 |
| other | 明确领域外 | 拒答 |
| casual | 闲聊问候 | 自然回应 |
| unknown | 无法确定 | 软引导 + RAG |

**演进历史**（重要，理解为什么这样做）：
1. 关键词词典 → 覆盖不全（"飞机"漏判）
2. 3b LLM 分类 → 慢（1-2 秒），专业术语误判（"边刷"判 other）
3. 深度学习分类头 → 快（约 1.1 秒，瓶颈在 embedding），准确率 96.91%

### 4.3 hybrid_retriever.py —— 双路召回

```python
def search(query, filter=None):
    dense_results  = self.dense.search(query, filter=filter)   # 向量 top-10
    sparse_results = self.sparse.search(query, filter=filter)  # BM25 top-10
    fused = reciprocal_rank_fusion(dense_results, sparse_results)  # RRF → top-8
    return [doc for doc, _, _ in fused]
```

**为什么双路**：稠密检索擅长语义近似（"水痕"≈"水渍"），但精确关键词（"边刷"）会被同类别词稀释；BM25 精确命中关键词，但缺乏同义词理解。两者 RRF 融合取长补短。

### 4.4 vector_store.py —— 稠密检索 + 入库

核心函数：

| 函数 | 职责 |
|------|------|
| `ingest_file` | 单文件入库：解析 → 条目分块 → 提取价格 metadata → 向量化写入 Chroma |
| `ingest_data_dir` | 批量入库 `data/knowledge/` |
| `DenseRetriever.search` | 向量检索 + metadata 过滤 |
| `search_by_filter` | **返回所有**符合 metadata 条件的 chunk（无 top-k 限制，用于预算完整枚举） |
| `build_hybrid_index` | 从 Chroma 读 chunk 构建 BM25 索引（支持 pickle 缓存） |

### 4.5 entry_splitter.py —— 编号条目分块

**核心设计**：知识库是"编号条目列表"格式（`1. **标题**`），用字符数硬切（RecursiveCharacterTextSplitter）会把多个条目切进一个 chunk，导致 metadata 无法精确对应。

分块器识别 `数字. ` 边界，**每个条目一个 chunk**：

```
1. **米家扫拖机器人 M20**     → chunk1（价格 899）
   - 吸力：2800Pa
   - 参考价：899
2. **追觅 D10s**              → chunk2（价格 1099）
   - 参考价：1099
```

这样每个 chunk 的 `min_price == max_price == 该型号价格`，metadata 精确对应。

### 4.6 metadata_extractor.py —— 结构化元数据

三个职责：

1. **入库时提取**（`extract_price_metadata`）：从 chunk 文本扫描"参考价：XXX"，写入 `min_price`/`max_price`
2. **检索时解析**（`build_filter` + `resolve_budget_filter` 统一入口）：从 query 提取预算约束（"1000以内" → `{"min_price": {"$lte": 1000}}`），支持中文数字（"一千"→1000）；规则 miss 且含预算 hint 时由 `resolve_budget_filter` 走 3b function calling 兜底
3. **直出时格式化**（`extract_model_info` + `format_model_line`）：从型号 chunk 提取型号名/吸力/导航/避障，拼成统一格式

### 4.7 hot_ingest.py —— 热更新

后台 daemon 线程每 30 分钟：

```
扫描 data/knowledge/ → 计算每个文件 MD5 → 与快照对比
  ├─ added    → ingest_file 入库
  ├─ changed  → 删旧 chunk + 重新入库
  ├─ removed  → 按 file_name 删 chunk
  └─ 无变化    → 跳过
更新快照 + 重建 BM25 索引
```

### 4.8 file_tools.py —— 文档解析

按扩展名多策略分发：

| 格式 | 解析方式 |
|------|---------|
| .txt/.md | 标准库直接读（UTF-8，多编码回退） |
| .pdf | pypdf（文本型 PDF；扫描件回退 Docling OCR） |
| .csv | 标准库 csv |
| .docx/.pptx/.xlsx | Docling |

### 4.9 function_tools/date_tool.py —— 日期计算工具

LLM function calling 的日期工具。核心函数：

| 函数 | 职责 |
|------|------|
| `parse_absolute_date` | 解析绝对日期："2025年三月"/"2025年3月"/"2025年" → 日期范围 |
| `parse_relative_date` | 解析相对日期："最近半年"/"近三个月"/"今年"/"去年" → 日期范围 |
| `parse_date` | 统一入口（先绝对后相对） |
| `calc_date_range` | 工具执行器（LLM 发出 tool_call 后由 agent 调用） |
| `build_date_filter` | 日期范围 → Chroma `where` filter |

**设计要点**：
- 规则解析优先（快、可靠），LLM function calling 兜底（覆盖规则未覆盖的表达）
- publish_date 以 int（YYYYMMDD）存储，因为 Chroma 的 `$gte/$lte` 只接受 int/float，不接受字符串
- 日期范围用 `$and` 组合两个单操作符条件（Chroma 要求每个表达式只能有一个操作符）

**调用流程**（agent.py `_resolve_date_filter`）：
```
query 含时间表达？
  → 规则 parse_date 命中 → 直接得到日期范围
  → 规则未命中 + 含时间 hint → LLM function calling（chat_with_tools + DATE_TOOL_SCHEMA）
       → LLM 返回 tool_call → calc_date_range 执行 → 日期范围
  → build_date_filter → 与预算 filter 合并（$and）
```

### 4.10 function_tools/budget_tool.py —— 预算提取工具（function calling 兜底）

LLM function calling 的预算工具，作为 `build_filter`（规则）未命中时的兜底。核心：

| 项 | 职责 |
|----|------|
| `BUDGET_TOOL_SCHEMA` | 单一 integer 参数 `budget_max` 的 function-calling schema |
| `budget_args_to_filter` | LLM 返回的 `{budget_max}` → Chroma `where` filter（`min_price ≤ budget_max`）|

**设计要点**：
- **规则优先**：`build_filter` 能识别的"1000以内""1000-2000"走 0 延迟规则；只有规则 miss 的口语/模糊表达（"一千来块""1500上下""两千出头""八百多"）才走 LLM 兜底
- **单一 integer 参数**：qwen2.5:3b 的 function calling 只能稳定处理单一 integer 参数——多参数（min/max）会被误填成 `user_input`，string 参数不做数值转换。因此只提取单一上限 `budget_max`，区间下限仍由规则（阿拉伯数字区间）覆盖
- **3b 而非 7b**：兜底用 `qwen2.5:3b`（~2.3s），比 7b 快 2-3 倍；代价是"出头"这类语义较难表达偶尔 miss（tool_calls 空 → 退化为全库检索，不给错误预算）

**调用流程**（统一入口 `metadata_extractor.resolve_budget_filter`，agent.py 直接调用）：
```
query 含预算？
  → 规则 build_filter 命中 → 直接得到 filter
  → 规则未命中 + 含预算 hint（_BUDGET_HINT_RE）→ LLM function calling（3b）
       → LLM 返回 {budget_max} → budget_args_to_filter → filter
  → 无 hint → None（全库兜底）
```

### 4.11 function_tools/symptom_tool.py —— 故障现象分类工具（function calling 兜底）

LLM function calling 的故障分类工具，作为 repair SOP `_SYMPTOM_MAP`（关键词）未命中时的兜底。核心：

| 项 | 职责 |
|----|------|
| `SYMPTOM_QUERY_MAP` | 10 个故障类型编号 → 标准检索 query |
| `SYMPTOM_TOOL_SCHEMA` | 单一 integer 参数 `symptom_id`（0-10）的 function-calling schema |
| `symptom_id_to_query` | 编号 → 标准 query；0/越界/非数字 → None |

**设计要点**：
- **规则优先**：`_SYMPTOM_MAP` 的 21 个关键词（"不动""漏水""异响"…）0 延迟匹配；口语描述（"奇怪的声音""地上有水""边角扫不到"）miss 时用 3b 归类
- **单一 integer 参数**：与 budget_tool 同理，3b 只能稳定处理单一 integer，故用编号而非多字段/自由字符串
- **挂载点**：`_extract_symptom` 只在 repair SOP 的 ask_symptom 步骤调用（此时用户已在描述故障），无需 hint 前置判断
- **miss 安全退化**：3b 归类为"无明确故障"（id=0）→ None → SOP 走 retry 重新问，不给错误答案

### 4.12 sops/ —— SOP 标准操作流程（多轮引导）

**定位**：SOP 是「代码驱动的流程循环」（状态机），与「LLM 驱动的 agent loop」不同——步骤固定、确定性高、不额外调 LLM，延迟可控。适合客服这类有标准流程的场景。

核心文件：

| 文件 | 职责 |
|------|------|
| `base.py` | 会话状态（内存，按 session_id 隔离）+ 执行器：`start_sop`/`continue_sop`/`end_sop`/`match_sop`；知识域定义 + 追问处理（含开场 intro、价格极值）|
| `purchase.py` | 选购推荐 SOP：收集预算（上限/区间）+ 宠物 → 结构化推荐 |
| `repair.py` | 故障排查 SOP：问现象 → 检索（定向故障排除域）→ LLM 生成排查步骤 |

**步骤类型**（三步式状态机）：

| 类型 | 行为 | 执行方 |
|------|------|--------|
| `ask` | 提问收集一个槽位，提取失败用 retry 话术重问 | 代码（确定性）|
| `action` | 调用 skill 执行动作（如按预算检索）| skill |
| `reply` | 输出结果并结束 SOP | 代码/模板 |

**选购推荐 SOP 流程**：

```
触发：robot/unknown 意图 + 命中 trigger + 尚未提供预算
  → ask_budget  问预算（支持"1000以内"上限 / "1000-2000"区间）
  → ask_pet     问是否有宠物
  → do_search   按预算检索（复用 search_by_filter + format_model_line）
  → reply       输出推荐列表
```

**故障排查 SOP 流程**：

```
触发：robot 意图 + 故障现象词（不动/漏水/异响/不充电…）
  → ask_symptom  问现象（_SYMPTOM_MAP 把口语现象改写成标准检索 query）
  → do_search    检索故障条目 + LLM 生成分步排查方案
  → reply        输出排查步骤
```

**关键设计决策**：
- **触发时机**：选购意图 + 命中 trigger 即进 SOP；已含预算的 query（"1000以内推荐"）也进 SOP，由首轮提取预填预算后继续问宠物，推荐更精准
- **guards 排除条件**：SOP 定义自带 `guards`（排除条件），`match_sop` 统一评估——选购咨询命中 guard 就不触发 SOP 走 RAG，消除 agent.py 里的 `sop_id == "purchase"` 硬编码特判
- **首轮提取**：`start_sop` 用 `_run(query)` 而非 `_run(None)`，首轮尝试用触发 query 提取第一个槽位（预填用户已给的信息）；首轮提取失败用 ask 话术（而非 retry 话术），避免"刚说想买就被回预算没听清"
- **开场提示**：进入 SOP 先给一句 intro（"我来帮您推荐～"），让用户知道自己进入了引导环节，再问第一个问题
- **退出机制**：用户说"算了/退出/取消/换个问题"时 `end_sop`，回落到正常流程
- **槽位即上下文**：SOP 的上下文是结构化槽位（预算、宠物），不直接拼用户原话，天然规避一部分提示词注入风险
- **多会话隔离**：会话状态、推荐结果、退出确认、上下文全部按 session_id 存储（前端每次新建对话生成 UUID），多会话互不串扰

**追问与最近发布查询**（`base.py` 的 `handle_followup`）：

SOP 推荐结束后，用户常追问上一轮结果。按语义分两类：

| 类型 | 例子 | 处理 |
|------|------|------|
| 重定向追问 | "有没有比较新的""最近发布""新款" | 全局检索 + 按发布时间倒序（不限上一轮区间）|
| 价格极值 | "最贵/最便宜的是哪款" | 全局按价格排序取极值（不依赖上一轮）|
| 限定追问 | "有没有更便宜的" | 上一轮推荐结果内，按价格升序 |

实现要点：
- 上一轮推荐的结构化型号列表（含名称/价格/发布时间）按 session_id 保存在模块级 dict（`_last_recommends`），供追问筛选
- 笼统"最近"（不带时间单位，如"最近有什么发布的"）也走全局检索；带时间单位（"最近半年"）仍走日期工具，用正则区分
- 追问检测放在 SOP 触发之前，避免"有没有"类追问被误当新选购

**关于追问的架构结论**（重要）：追问检测解决的是「结构化追问」（比较新/更便宜），这是确定性筛选，用代码精确完成；「自由指代」（"它怎么样""那这个呢"）需要拼接对话历史 + LLM 语义理解，是另一层能力，两者互补而非替代。

### 4.13 tools/context_store.py —— 会话上下文（持久化 + 历史拼接）

按 session_id 把对话持久化到 `data/context/<session_id>.jsonl`（每行一条 JSON 消息），内存缓存保留最近 6 轮（12 条，供快速读取）。

| 函数 | 职责 |
|------|------|
| `append_message` | 追加一条消息（写内存缓存 + jsonl 文件），可携带 intent / models 元数据 |
| `get_recent` | 取最近 n 条消息（默认 6 轮），优先内存缓存，未缓存则读文件 |
| `get_last_models` | 从最近消息倒序找第一条带 models 的 assistant 消息（取最近推荐型号列表） |
| `ensure_session_id` | 校验前端 UUID，非法则重新生成（防御异常 session_id 写入文件名） |
| `list_sessions` | 列出所有会话（摘要 + 消息数 + 时间），供前端历史对话列表 |

**用途**：
1. **自由指代消解（历史拼接 LLM）**：用户说"它怎么样/那这个呢"时，RAG 生成把最近 6 轮历史拼进 prompt，让 LLM 自己看上下文理解"它"指谁，不再做代码替换
2. **历史会话回放**：前端会话抽屉（`/api/sessions`、`/api/sessions/<sid>/messages`）列出/加载历史对话；刷新页面保留上下文

**设计要点**：
- 文件全量保留 + 内存只留 6 轮：文件是持久层（重启可恢复），内存是读热层（避免频繁读文件）
- assistant 消息由 webapp/app.py 在流式完成后写入，并附带 `models`（结构化推荐列表），供"更便宜"等结构化追问筛选
- 写文件追加而非覆写，天然支持多轮累积

---

## 五、技术决策记录（ADR）

> 记录关键决策及原因，帮助新成员理解"为什么这么做"。

### ADR-1：embedding 模型 qwen3-embedding → bge-m3

**原因**：qwen3-embedding:0.6b（600MB）对中文语义分辨力不足，检索相似度分数偏低（最低 0.22）。换 bge-m3（1.2GB）后，语义检索质量显著提升。

### ADR-2：分块从字符硬切 → 编号条目分块

**原因**：知识库是编号条目格式，字符硬切（chunk_size=600）把 3-5 个条目塞进一个 chunk，embedding 语义被稀释，metadata 无法精确对应型号价格。改为每条目一个 chunk 后，预算过滤从"模糊区间"变为"精确价格"。

### ADR-3：预算查询用"结构化直出"而非 LLM 转述

**原因**：小模型（≤7b）转述检索结果时会漏型号、混格式、产生幻觉。既然 metadata 已精确，直接用代码拼接型号列表，100% 可靠。这是"结构化问题用代码，非结构化问题用 LLM"的体现。

### ADR-4：意图分类从 LLM → 深度学习分类头

**原因**：意图分类本质是文本分类任务，不需要大模型。3b LLM 分类慢（1-2 秒）且专业术语误判（"边刷"被判 other）。改为 bge-m3 + Linear 分类头后，速度提升、专业术语可通过训练数据解决。

### ADR-5：训练数据必须与推理分布一致

**原因**：最初从知识库抽取的 robot 样本是陈述句（"故障现象：xxx"），而真实用户问的是疑问句（"xx怎么办"），导致训练 100% 但推理严重误判。通过规则改写（"故障现象：xx"→"xx怎么办？"）对齐分布后，推理恢复正常。

### ADR-6：SSE 流式传输用 JSON 编码

**原因**：结构化直出的大 chunk 含 `\n` 换行，与 SSE 事件分隔符 `\n\n` 冲突，导致前端截断。用 `json.dumps` 编码每个 chunk，换行转义成字面量 `\n`，前端 `JSON.parse` 解码，彻底解决。

### ADR-7：日期工具用规则优先 + LLM function calling 兜底

**原因**：日期表达（"最近半年""2025年三月"）高度规则，正则解析又快又准；但用户表达千变万化，规则无法穷举。因此规则解析优先，命中直接返回；未命中且含时间 hint 时，才走 LLM function calling 兜底，兼顾速度与覆盖。

### ADR-8：publish_date 用 int（YYYYMMDD）存储

**原因**：Chroma 的 `$gte/$lte` 过滤只接受 int/float 操作数，不接受字符串。若 publish_date 存 ISO 字符串（"2025-03-15"），日期过滤会报错。改用 int（20250315）后，数值比较天然满足字典序，且符合 Chroma 的类型约束。

### ADR-9：多轮用 SOP（代码驱动）而非 agent loop（LLM 自主）

**原因**：客服的多轮是"引导式"（系统问、用户答），有标准流程，用 SOP 状态机实现确定性高、不额外调 LLM、延迟可控；而 agent loop 让 LLM 自主决定每步动作，7b 模型多步循环稳定性不足（易死循环/错误决策），且延迟线性叠加。评估结论：当前阶段 SOP 优先，agent loop 留到多工具编排需求出现时再做「受控 mini loop」。

### ADR-10：结构化追问用代码筛选，自由指代才用历史拼接

**原因**：追问分两类——结构化追问（"有没有比较新的/更便宜的"）是确定性筛选/排序，用代码（保留结构化型号列表 + 关键词路由）精确完成，符合 ADR-3「结构化用代码」；自由指代（"它怎么样""那这个呢"）才需要拼接对话历史 + LLM 语义理解。若用历史拼接替代代码筛选，会让结构化追问退化成 LLM 转述（漏型号/幻觉），是倒退。结论：两者互补，当前先做代码筛选，自由指代等真正遇到再加。

### ADR-11：轻量文档隔离用"分域过滤"而非"独立向量库"

**背景**：知识库按文件分域（品牌介绍/选购指南/具体型号/常见维修/售后服务），"吸力"这类词跨域出现（选购的"吸力参数"、故障的"吸力下降"），全库检索会把跨域词带偏召回。

**原因**：备选方案有（a）每个域建独立向量库、（b）单库 + metadata `file_name` 过滤、（c）靠 LLM 过滤。选（b）——独立向量库要维护多份索引、跨库合并复杂；LLM 过滤延迟高且不稳定。单库 + `file_name` 过滤成本最低，且保留全库兜底（识别不准则不过滤）。

**域路由设计**：`DOMAIN_MAP`（场景 → 文件名）统一在 `config/word_dict_config.py`，`agent._route_domain` 按关键词把 query 路由到对应域，`repair.py` 定向故障域。**宁缺毋滥**——只对高置信度场景过滤（品牌/选购/型号/故障/售后/维护 6 域），识别不准则全库兜底，是"提纯"而非"硬隔离"。跨域词在错误域仍有少量命中属正常，不影响主要召回。

### ADR-12：槽位提取用"规则优先 + function calling 兜底"，而非整体 function calling 意图路由

**背景**：评估"用 function calling 做意图路由"（让 LLM 一次完成意图 + 槽位提取）迁移到本项目的可行性。

**原因**：整体迁移不划算——意图路由是每个 query 的必经路径，function calling 要额外调 LLM（1~10s），且回答阶段还要再调一次 LLM，延迟翻倍；qwen2.5:7b 的 function calling 稳定性也不如本地分类头（ADR-4/ADR-5 的资产不应丢弃）。**正确姿势是把 function calling 作为分类头/规则之后的兜底层**，与 date_tool（ADR-7「规则优先 + LLM function calling 兜底」）完全同构。

**预算槽位落地**：`build_filter`（规则）优先，规则 miss 且含预算 hint 时，用 `function_tools/budget_tool.py` 的 function calling 兜底提取上限。实测 qwen2.5:3b 只能稳定处理**单一 integer 参数**（多参数被误填 `user_input`、string 参数不做数值转换），故只提取单一上限 `budget_max`，区间下限仍由规则覆盖；3b 比 7b 快 2-3 倍，代价是"出头"这类语义较难表达偶尔 miss（退化为全库检索，不给错误预算）。

**推广到故障现象**：`function_tools/symptom_tool.py` 用同一模式（单一 integer 编号 `symptom_id` 0-10）把口语故障（"奇怪的声音""地上有水"）归类到标准故障 query，挂载在 repair SOP 的 `_extract_symptom`。miss 时 SOP 走 retry 重新问，不给错误答案。

### ADR-13：情绪识别用"纯规则前置"，不用 function calling 或分类头

**背景**：评估增加情绪识别前置模块（识别愤怒/焦虑并安抚）。

**原因**：情绪识别是每个 query 的必经路径（前置），若用 function calling 会违反 ADR-12「必经路径不加 LLM」；而它的 miss 后果很温和（**没安抚 ≠ 给错答案**），不值得训练分类头。因此用**纯规则词表（0 延迟）+ 分级安抚**——强烈负面（投诉/退钱）给正式安抚，轻微负面（烦/急）给轻量安抚，命中即安抚、不拦截。

**关键细节**：词表要规避领域歧义词——"报警"（机器人故障报警）、单独"急"（急停键）、"崩溃"（APP崩溃）都不算情绪词，避免误判。

**演进**：命中安抚后还会**剥离情绪词**再走后续流程——否则 LLM 会把"烂透了"这类抱怨当成独立问题，输出"暂无信息 + 具体答案"的矛盾。剥离后 LLM 只见具体诉求，配合"禁止矛盾输出"的 prompt 规则消除矛盾。

### ADR-14：单轮决策层的"排除条件"收敛到 SOP 的 guards，而非散在编排层

**背景**：意图分类之后的决策层（追问 → SOP 触发）曾是"匹配就拦截"的散落判断——选购 SOP 的"选购咨询不触发""含预算不触发"这两个排除条件，以 `sop_id == "purchase"` 特判形式硬编码在 agent.py 里。

**原因**：把排除条件收敛到 SOP 定义（`guards` 字段），由 `match_sop` 统一评估「trigger 命中 + guards 均未命中 → 触发；任一 guard 命中 → 返回 None 走后续分支」。好处：消除编排层对具体 SOP 的硬编码特判，新增场景只需改 SOP 定义，不动 agent.py。

**边界**：只收敛「场景触发 + 排除条件」这一类判断。追问/价格极值（单轮排序查询）、情绪/注入（全局前置）、领域路由（检索层）、退出词（会话内）语义不同，保持原位，不强行归入 SOP——避免"统一收敛"变成概念混淆。

**演进**：guards 现保留「选购咨询」（is_consulting）和「售后咨询」（is_aftersales，保修/售后/退换货）两类排除条件——"买回来后有没有保修"含"买"会命中 purchase trigger，但语义是售后而非选购。同时移除了过宽的"有没有"触发词："有没有 XX"后面跟什么无法穷举（保修/功能/价格），靠词表必然误伤，不如让它回归正常路由（含预算走结构化直出、否则走 RAG）。

### ADR-15：SOP 首轮用触发 query 提取槽位，含预算也进 SOP（不短路直出）

**背景**：`start_sop` 曾用 `_run(None)` 丢弃触发 query，导致 repair 首轮症状被丢弃（用户重复描述）、purchase 首轮预算被丢弃；同时 has_budget guard 把含预算的 query 短路成纯价格直出，宠物等信息整条丢失。

**原因**：改 `_run(query)` 让首轮也走提取——repair"机器人不动了"一步到位，purchase"2000-2500推荐"预填预算后继续问宠物。配套两处：① 首轮提取失败用 ask 话术（而非 retry），避免"刚说想买就被回预算没听清"；② 移除 has_budget guard，含预算的 query 也进 SOP（预填预算 + 问宠物），代价是"1000以内推荐"从直出变成多问一轮宠物——推荐场景问宠物本就有价值，可接受。

### ADR-16：词表集中到 config/word_dict_config.py 配置中心

**背景**：情绪词、退出词、域映射、触发词、症状映射等 15+ 个词表曾散落在 agent.py / sops/*.py / function_tools/*.py 等 5 个文件里，改一个词要翻多处，并发编辑时还互相覆盖。

**原因**：集中到 `config/word_dict_config.py`，按领域分节（情绪/退出/域/触发/症状/追问），各模块 `from config.word_dict_config import XXX` 引用。边界：只集中"词列表 + 映射"；正则（注入检测、日期/预算提示）和中文数字映射是"解析逻辑"而非"词表"，保留原处；测试脚本的 mock 词表保留（测试独立性）。

### ADR-17：多会话与上下文用"session_id 隔离 + jsonl 持久化"，自由指代用"历史拼接 LLM"

**背景**：v1.8.3 需要支持前端多会话（新建对话/切换历史）+ 自由指代多轮（"它怎么样"指上一轮推荐）。

**原因**：
- **session_id 隔离**：SOP 会话、推荐结果、退出确认原本都是模块级单例（单用户假设），多会话会互相串扰。全部改为按 session_id 存储（`_sessions`/`_last_recommends`/`_pending_exits`），前端用 `crypto.randomUUID()` 生成会话 ID，刷新即新会话，切换会话即恢复上下文。
- **jsonl 持久化**：会话状态要跨刷新存活（前端历史对话列表 + 回放），内存不够、数据库过重——`data/context/<session_id>.jsonl` 每行一条消息，文件全量保留、内存缓存最近 6 轮，读写都是 append/切片，成本极低。
- **自由指代用历史拼接 LLM 而非代码替换**：最初用"指代词 + 上一轮推荐型号名"做确定性替换（`REFERENCE_WORDS`），但暴露两个问题——依赖 `save_recommend` 全覆盖（"最便宜/详细讲讲"等路径漏保存就指代错）、且只能换成型号名无法处理复杂指代。最终回到 ADR-10 的本意：RAG 生成时把最近 6 轮历史拼进 prompt，让 LLM 自己看上下文理解"它"指谁，彻底摆脱对结构化上下文的依赖。

### ADR-18：品牌化"不染一尘"——知识库全量替换，域从 4 扩到 6，售前售后咨询走 RAG

**背景**：项目从通用扫地机器人客服转型为"不染一尘"品牌专属客服，知识库全量替换为品牌文件（品牌介绍/选购指南/具体型号/常见维修/售后服务 5 个），删除 8 个通用文件。

**原因**：
- **域从 4 扩到 6**：新增"品牌介绍"（brand）和"售后服务"（aftersales）两个域，加上选购/型号/故障/维护共 6 域；维护（maintain）与维修（repair）共用"常见维修问题.txt"（品牌文件里维护与维修内容未拆分）。
- **售前售后咨询走 RAG 而非 SOP**：品牌咨询（"为什么买/优势"）和售后咨询（"报修/更换零件/保修"）是事实性咨询（直接回答政策/介绍），不需要多轮引导；只有选购推荐（收集预算/宠物）和故障排查（问现象）才走 SOP。给 purchase SOP 加 brand guard、repair SOP 加 aftersales guard，把咨询类从 SOP 挡出去。
- **安全前置**：危险现象（冒烟/烧焦/鼓包/进水/漏电）在一切改写/路由之前拦截，直接输出停机联系售后的安全话术，避免"冒烟"被 symptom 改写成"充不进电"丢失危险信号。
- **意图分类重训练**：品牌化后分类头不认识"不染一尘"，品牌/售后 query 被误判 unknown。补训练集（`_ROBOT_BRAND`：品牌咨询 + 售后报修样本），并修复 `_extract_title` 逐行匹配（品牌文件带"## 章节"，旧实现只匹配首行导致抽取 0 样本）。重训练后测试准确率 91.5%。
- **退出机制通用化**：所有 SOP 开场语统一追加"（随时可回复「0」退出本环节）"，"0" 精确匹配退出，用 `_with_exit_hint` 通用函数实现。

---

## 六、已知问题与坑

### 6.1 当前已知问题

| 问题 | 影响 | 状态 |
|------|------|------|
| 意图分类对"电池续航"等无领域词边界 case | "电池续航下降怎么办"偶发误判 other（robot 48% vs other 50%）| 语义固有歧义 |
| 自由指代已解决 | 通过 RAG 生成拼接最近 6 轮历史，LLM 自主消解"它/这个"等指代；小模型偶发消解不准属固有局限 | 已实现 |

### 6.2 开发中的坑（避坑指南）

1. **Ollama embedding 批量崩溃**：一次 embed 500+ 条触发 Ollama 内部 tokenize 服务（62633 端口）崩溃，需分批（64 条/批）处理。
2. **端口残留**：Flask 多次启动后 5050 端口被旧进程占用，导致请求打到旧代码。排查时先 `netstat -ano | findstr 5050` 清理残留进程。
3. **CRLF 换行**：Windows 下文件用 CRLF，部分编辑工具按 LF 匹配失败。可用 Python 脚本或完整 Read+Write 重写。
4. **中文编码**：Windows 默认 GBK，读 txt 用 UTF-8 需显式指定；Docling 处理 PDF 在 GBK 环境下会崩，需 `PYTHONUTF8=1`。

---

## 七、未来规划

### 7.1 自由指代（已通过历史拼接 LLM 实现）

引导式多轮（系统问、用户答，如选购推荐 SOP）已通过 `sops/` 落地；自由指代（"它怎么样""那这个呢"）已通过 RAG 生成拼接最近 6 轮历史、由 LLM 自主消解（见 4.13/ADR-17）。

剩余的是更复杂的指代（跨话题、多指代、长程依赖）与历史压缩——当轮次增多、上下文变长时，可引入历史摘要/结构化压缩（见 ADR-17 的压缩评估）。注意：结构化追问（比较新/更便宜）保持代码筛选，不退回 LLM 转述（ADR-10）。

### 7.2 Function Calling / Tool Use（已部分实现）

已落地日期工具（`function_tools/date_tool.py`），qwen2.5:7b 的 function calling 能力已验证可用。后续可扩展更多工具：`query_products(预算)`、`search_faq(query)`、`check_order(订单号)` 等。新增工具统一放 `function_tools/`，提供 `*_TOOL_SCHEMA` + 执行函数即可。

### 7.3 意图分类数据集扩充

当前 658 条（robot 434 + other 103 + casual 60 + unknown 61），robot 占比偏高。需补充"预算推荐""天气"等边界样本，并考虑数据平衡。

---

## 八、开发约定

1. **日志用英文**：logger 消息统一英文（历史中文已清理）
2. **日志分级**：召回详情用 DEBUG，汇总数用 INFO；控制台彩色（INFO 白/WARN 黄/ERROR 红）
3. **配置模板化**：敏感/本地配置写 `*_template.yaml` 提交，实际配置 gitignore
4. **导入风格**：tools 内部用直接导入（`from log_tool import`），webapp 用包导入（`from tools.xxx import`），两者都靠 sys.path 同时包含项目根和 tools 目录
5. **知识库格式**：编号条目（`数字. ` + `**标题**` + `- 参数`），详见 `data/knowledge_example/格式模板.txt`
