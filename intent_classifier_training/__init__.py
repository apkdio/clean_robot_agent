"""意图分类模型训练工具包。

  - build_intent_dataset.py    数据集构建（从知识库抽取 + 规则改写 + 内置种子词表）
  - train_intent_classifier.py 分类头训练（bge-m3 embedding + Linear 分类器）

训练产物输出到 data/bgm_model/，推理时由 tools/intent_router.py 自动加载。
"""
