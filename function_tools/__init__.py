"""工具契约包。

只放主链单次抽取的 schema（`schemas.py`）：模型能填什么参数、参数如何描述。
LLM 调用在 `service/llm_extract.py`，纯规则/校验在 `utils/` 对应业务模块。
AI 工具编排已移出主链，不再保留 agent tool 通道。
"""
