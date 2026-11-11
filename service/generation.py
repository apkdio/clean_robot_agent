"""生成出口：流式生成、失败兜底与答案硬事实校验接线。"""
from __future__ import annotations

import random

from langchain_core.messages import HumanMessage, SystemMessage

from config.word_dict_config import NO_ANSWER_REPLIES
from mapper import trace_store
from utils.log_tool import get_logger
from utils.prompts_tool import load_main_prompts, render
from service.turn_recorder import log_behavior

logger = get_logger(name="generation")


def verify_answer(answer: str, chunks: list, query: str) -> None:
    """核对 RAG 答案里的硬事实；只记日志与 trace，不改变已经输出的内容。"""
    from service.answer_checker import check_answer

    try:
        result = check_answer(answer, chunks, query)
    except Exception as e:  # noqa: BLE001
        logger.warning("[AnswerVerify] 校验异常，跳过：%s", e)
        trace_store.node("answer.verify", status="degraded", out={"error": type(e).__name__})
        return

    if result.get("skipped"):
        logger.warning("[AnswerVerify] 校验降级：%s", ",".join(result["skipped"]))
    trace_store.verify(result["checked"], result["missing"],
                       action="log", exempt=result.get("exempt"),
                       skipped=result.get("skipped"),
                       unchecked=result.get("unchecked"))
    if result.get("unchecked"):
        logger.warning("[AnswerVerify] 型号目录中没有，未核对：%s", ",".join(result["unchecked"]))
    if result["missing"]:
        logger.warning("[AnswerVerify] 召回片段中找不到：%s", ",".join(result["missing"]))


def stream_with_fallback(messages: list, temperature: float = 0.3, stop_check=None,
                         verify=None, stream_chat_func=None):
    """流式生成，失败/零输出时走确定性兜底并留痕。

    verify 只在完整生成（未停止、未异常）后调用，半截答案不参与核对。
    """
    from gateway.llm_tool import get_chat_model_name

    if stream_chat_func is None:
        from gateway.llm_tool import stream_chat as stream_chat_func

    produced = False
    parts = []
    stopped = False
    failed = False
    try:
        for chunk in stream_chat_func(messages, model=get_chat_model_name(),
                                      temperature=temperature):
            if stop_check is not None and stop_check():
                logger.info("[Agent] stopped during generation")
                stopped = True
                return
            produced = True
            parts.append(chunk)
            yield chunk
    except Exception as e:  # noqa: BLE001
        failed = True
        logger.warning("[Generate] 流式生成失败：%s", e)
        trace_store.note_error(e)
        trace_store.node("generate", status="degraded")
    if not produced:
        log_behavior("no_answer_fallback", hits=0, via="generate_failed")
        yield random.choice(NO_ANSWER_REPLIES)
        return
    if verify is not None and not stopped and not failed:
        verify("".join(parts))


def casual_stream(query: str, temperature: float = 0.3, stop_check=None,
                  stream_chat_func=None):
    """闲聊回应：不走检索，但仍走同一套生成兜底。"""
    prompt = render("casual_guard", query=query)
    if prompt is None:
        prompt = f"用户说：{query}"
    messages = [
        SystemMessage(content=load_main_prompts()),
        HumanMessage(content=prompt),
    ]
    yield from stream_with_fallback(messages, temperature=temperature,
                                    stop_check=stop_check,
                                    stream_chat_func=stream_chat_func)


def rag_stream(query: str, history_block: str, chunk_texts: list,
               temperature: float = 0.3, stop_check=None,
               stream_chat_func=None):
    """RAG 生成：参考资料 + 对话历史 + 问题，生成后做硬事实校验。"""
    from gateway.llm_tool import get_chat_model_name

    context_block = "\n\n".join(chunk_texts)
    user_message = render("rag_answer_rules", history_block=history_block,
                          chunk_texts=context_block, query=query)
    if user_message is None:
        user_message = f"参考资料：\n{context_block}\n\n问题：{query}"
    trace_store.step("generate", model=get_chat_model_name())
    yield from stream_with_fallback(
        [SystemMessage(content=load_main_prompts()),
         HumanMessage(content=user_message)],
        temperature=temperature,
        stop_check=stop_check,
        verify=lambda text: verify_answer(text, chunk_texts, query),
        stream_chat_func=stream_chat_func,
    )
