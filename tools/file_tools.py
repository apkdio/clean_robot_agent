import hashlib
import os
import sys

from langchain_core.documents import Document
from log_tool import get_logger

logger = get_logger(name="file_tools")

# 纯文本扩展名：直接用标准库读取，速度快且不会出现 GBK 乱码
_TEXT_EXTS = (".txt", ".md", ".markdown", ".asciidoc", ".adoc")
# PDF：优先使用 pypdf（纯 Python、速度快、无编译器依赖）；失败时回退到 Docling
_PDF_EXTS = (".pdf",)
# 结构化文档扩展名：由 Docling 解析（DOCX/PPTX/XLSX/HTML 等）
_DOC_EXTS = (
    ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".odt", ".ods", ".odp", ".html", ".htm", ".epub",
)


def get_file_md5_hex(file_path: str):
    """计算文件 MD5 哈希。以二进制模式读取，不涉及编码。"""
    if not os.path.isfile(file_path):
        logger.error(f"[MD5 Module] File {file_path} not found.")
        return None
    md5_obj = hashlib.md5()
    chunk_size = 4096  # 4KB
    try:
        # 注意：二进制模式不能传入 encoding 参数
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"[MD5 Module] Calculate MD5 failed. {e}")
        return None


def _read_text_file(file_path: str) -> list[Document]:
    """用标准库读取纯文本文件，显式指定 UTF-8 以避免中文乱码。"""
    docs = []
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": file_path}))
            return docs
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"[Text Reader] Failed to read {file_path}: {e}")
            return docs
    logger.error(f"[Text Reader] Cannot decode {file_path}, please check file encoding.")
    return docs


def _read_csv_file(file_path: str) -> list[Document]:
    """用标准库读取 CSV 文件，将各行合并为单个文本块。"""
    import csv
    docs = []
    try:
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                with open(file_path, "r", encoding=encoding, newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"[CSV Reader] Cannot decode {file_path}.")
            return docs
        content = "\n".join(",".join(row) for row in rows)
        docs.append(Document(page_content=content, metadata={"source": file_path}))
    except Exception as e:
        logger.error(f"[CSV Reader] Failed to read {file_path}: {e}")
    return docs


def _read_with_docling(file_path: str) -> list[Document]:
    """使用 Docling 解析结构化文档（PDF/DOCX/PPTX/XLSX 等）。

    在中文 Windows 上读取文本/PDF 时，Docling 可能会遇到 GBK 编码问题；
    强制开启 UTF-8 模式（PYTHONUTF8）可避免崩溃。
    """
    # 确保进程以 UTF-8 模式运行，避免 torch/docling 的 GBK 解码崩溃
    os.environ.setdefault("PYTHONUTF8", "1")
    if getattr(sys, "flags", None) is not None and not sys.flags.utf8_mode:
        # 启动后无法再开启 UTF-8 模式；至少为子进程设置环境变量
        logger.warning("[Docling] UTF-8 mode not enabled; recommend running with PYTHONUTF8=1.")

    try:
        from langchain_docling.loader import DoclingLoader, ExportType
    except ImportError as e:
        logger.error(f"[Docling] langchain_docling not installed: {e}")
        return []

    try:
        loader = DoclingLoader(
            file_path=file_path,
            export_type=ExportType.MARKDOWN,
        )
        docs = loader.load()
        logger.info(f"[Docling] {file_path} parsed, {len(docs)} chunk(s).")
        return docs
    except Exception as e:
        logger.error(f"[Docling] Failed to parse {file_path}: {e}")
        return []


def _read_pdf_file(file_path: str) -> list[Document]:
    """使用 pypdf 读取 PDF（纯 Python、速度快、无编译器依赖）。

    pypdf 对基于文本的 PDF 效果良好；如果提取不到文本（例如扫描版 PDF），
    则回退到 Docling（需要 OCR，且在没有 C++ 编译器时可能失败）。
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        logger.error(f"[PDF] pypdf not installed: {e}; falling back to Docling.")
        return _read_with_docling(file_path)

    try:
        reader = PdfReader(file_path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        full_text = "\n".join(parts).strip()
        if full_text:
            logger.info(f"[PDF] pypdf extracted {file_path}: {len(reader.pages)} page(s), {len(full_text)} chars.")
            return [Document(page_content=full_text, metadata={"source": file_path})]
        logger.warning(f"[PDF] pypdf extracted no text, possibly a scanned PDF; falling back to Docling.")
        return _read_with_docling(file_path)
    except Exception as e:
        logger.error(f"[PDF] pypdf failed to parse {file_path}: {e}; falling back to Docling.")
        return _read_with_docling(file_path)


def extract_file(file_path: str) -> list[Document]:
    """按文件扩展名选择解析策略，返回 langchain Document 列表。

    - .txt/.md 等：标准库直接读取（快、不乱码）
    - .csv：标准库 csv 读取
    - .pdf：先用 pypdf（快），扫描件回退到 Docling
    - .docx/.pptx 等：Docling 解析
    """
    if not os.path.isfile(file_path):
        logger.error(f"[Extract File] {file_path} does not exist or is not a file.")
        return []

    ext = os.path.splitext(file_path)[1].lower()

    if ext in _TEXT_EXTS:
        logger.info(f"[Extract] Reading text mode: {file_path}")
        return _read_text_file(file_path)

    if ext == ".csv":
        logger.info(f"[Extract] Reading CSV mode: {file_path}")
        return _read_csv_file(file_path)

    if ext in _PDF_EXTS:
        logger.info(f"[Extract] Reading PDF mode: {file_path}")
        return _read_pdf_file(file_path)

    if ext in _DOC_EXTS:
        logger.info(f"[Extract] Parsing with Docling: {file_path}")
        return _read_with_docling(file_path)

    # 未知扩展名，尝试按文本读取
    logger.warning(f"[Extract] Unknown extension {ext}; reading as text: {file_path}")
    return _read_text_file(file_path)
