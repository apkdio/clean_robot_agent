import hashlib
import os
import sys

from langchain_core.documents import Document
from log_tool import get_logger

logger = get_logger(name="file_tools")

# Plain text extensions: read directly via stdlib, fast and no GBK garbling
_TEXT_EXTS = (".txt", ".md", ".markdown", ".asciidoc", ".adoc")
# PDF: prefer pypdf (pure Python, fast, no compiler dependency); fall back to Docling
_PDF_EXTS = (".pdf",)
# Structured document extensions: parsed by Docling (DOCX/PPTX/XLSX/HTML etc.)
_DOC_EXTS = (
    ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".odt", ".ods", ".odp", ".html", ".htm", ".epub",
)


def get_file_md5_hex(file_path: str):
    """Calculate file MD5 hash. Reads in binary mode, no encoding involved."""
    if not os.path.isfile(file_path):
        logger.error(f"[MD5 Module] File {file_path} not found.")
        return None
    md5_obj = hashlib.md5()
    chunk_size = 4096  # 4KB
    try:
        # Note: binary mode must not take an encoding argument
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"[MD5 Module] Calculate MD5 failed. {e}")
        return None


def _read_text_file(file_path: str) -> list[Document]:
    """Read a plain-text file via stdlib with explicit UTF-8 to avoid garbled Chinese."""
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
    """Read a CSV file via stdlib, joining rows into a single text block."""
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
    """Parse a structured document (PDF/DOCX/PPTX/XLSX etc.) using Docling.

    Docling may hit GBK encoding issues on Chinese Windows when reading text/PDF;
    forcing UTF-8 mode (PYTHONUTF8) prevents the crash.
    """
    # Ensure the process runs in UTF-8 mode to avoid torch/docling GBK decode crashes
    os.environ.setdefault("PYTHONUTF8", "1")
    if getattr(sys, "flags", None) is not None and not sys.flags.utf8_mode:
        # Cannot enable UTF-8 mode after startup; set env var for child processes at least
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
    """Read a PDF with pypdf (pure Python, fast, no compiler dependency).

    pypdf works well for text-based PDFs; if it extracts no text (e.g. scanned PDF),
    falls back to Docling (requires OCR and may fail without a C++ compiler).
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
    """Choose a parsing strategy by file extension and return a list of langchain Documents.

    - .txt/.md etc.: stdlib direct read (fast, no garbling)
    - .csv: stdlib csv read
    - .pdf: pypdf first (fast), scanned fallback to Docling
    - .docx/.pptx etc.: Docling parse
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

    # Unknown extension, try reading as text
    logger.warning(f"[Extract] Unknown extension {ext}; reading as text: {file_path}")
    return _read_text_file(file_path)
