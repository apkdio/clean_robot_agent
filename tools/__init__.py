"""核心工具模块包（RAG 管道的全部核心模块）。

模块清单：
  - agent.py                 RAG 编排入口
  - intent_router.py         意图分类（本地 bge-m3 + 分类头）
  - hybrid_retriever.py      双路召回（稠密 + 稀疏 → RRF）
  - vector_store.py          Chroma 稠密检索与入库
  - sparse_retriever.py      BM25 稀疏检索
  - rrf_fusion.py            RRF 倒数排名融合
  - metadata_extractor.py    结构化元数据（价格 / 预算 / 发布时间）
  - entry_splitter.py        编号条目分块器
  - hot_ingest.py            知识库热更新
  - file_tools.py            文档多策略解析
  - llm_tool.py              LLM / embedding 工厂（含 function calling）
  - log_tool.py              日志
  - config_tool.py           配置加载
  - path_tool.py             路径工具
  - prompts_tool.py          Prompt 加载
"""
