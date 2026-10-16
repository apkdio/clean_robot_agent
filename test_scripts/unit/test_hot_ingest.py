"""热更新（hot_ingest）测试：增量同步的删除/重入语义。

重点回归两条：摄取失败时旧 chunk 不许删（否则留下「该文件 0 chunk」的空洞）；索引里 0 chunk 的文件要被当 changed 重入。
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

import tools.vector_store as vs
from tools import hot_ingest as hi


class _FakeCollection:
    """只实现本用例用到的三个动作：按 file_name 取 id / 删 id / 删 file_name。"""

    def __init__(self, ids_by_file=None, fail_get=False):
        self.ids = dict(ids_by_file or {})
        self.fail_get = fail_get
        self.deleted_ids = []
        self.deleted_files = []

    def get(self, where=None, include=None):
        if self.fail_get:
            raise RuntimeError("chroma boom")
        return {"ids": list(self.ids.get((where or {}).get("file_name"), []))}

    def delete(self, where=None, ids=None):
        if ids:
            self.deleted_ids += list(ids)
        if where:
            self.deleted_files.append(where.get("file_name"))


class _FakeStore:
    def __init__(self, col):
        self._collection = col


def _patch(**attrs):
    """临时替换 tools.vector_store 的模块属性（apply_changes 内部是函数级 import）。"""
    orig = {k: getattr(vs, k) for k in attrs}
    for k, v in attrs.items():
        setattr(vs, k, v)
    return lambda: [setattr(vs, k, v) for k, v in orig.items()]


def test_changed_failure_keeps_old_chunks():
    col = _FakeCollection({"型号.txt": ["a1", "a2"]})
    restore = _patch(get_vector_store=lambda: _FakeStore(col),
                     ingest_file=lambda p, source_tag="": {"status": "error", "message": "依赖不可用"})
    try:
        r = hi.apply_changes("data/knowledge", {}, {"型号.txt": "sha"}, [])
        _assert_eq(col.deleted_ids, [], "摄取失败时不许删旧 chunk（否则留下 0 chunk 空洞）")
        _assert_eq(r["changed"], 0, "失败不计入 changed")
        _assert_in("型号.txt", r["failed"], "失败要显式进 failed，供调用方不推进快照")
    finally:
        restore()


def test_changed_success_replaces_old_chunks():
    col = _FakeCollection({"型号.txt": ["a1", "a2"]})
    restore = _patch(get_vector_store=lambda: _FakeStore(col),
                     ingest_file=lambda p, source_tag="": {"status": "ok", "chunks": 3})
    try:
        r = hi.apply_changes("data/knowledge", {}, {"型号.txt": "sha"}, [])
        _assert_eq(sorted(col.deleted_ids), ["a1", "a2"], "插入成功后按旧 id 删，正好删掉旧版本")
        _assert_eq(col.deleted_files, [], "不能再按 file_name 删（会把刚插的一起删掉）")
        _assert_eq(r["changed"], 1, "成功计入 changed")
        _assert_eq(r["failed"], [], "无失败")
    finally:
        restore()


def test_removed_deletes_by_file_name():
    col = _FakeCollection()
    restore = _patch(get_vector_store=lambda: _FakeStore(col),
                     ingest_file=lambda p, source_tag="": {"status": "ok", "chunks": 1})
    try:
        hi.apply_changes("data/knowledge", {}, {}, ["老文件.txt"])
        _assert_eq(col.deleted_files, ["老文件.txt"], "文件真的没了才按 file_name 整体删")
    finally:
        restore()


def test_files_missing_from_index():
    scan = {"a.txt": "s1", "b.txt": "s2", "c.txt": "s3"}
    counts = {"a.txt": 5, "b.txt": 0, "c.txt": 0}
    _assert_eq(sorted(hi.files_missing_from_index(scan, counts)), ["b.txt", "c.txt"],
               "索引里 0 chunk 的文件要能被挑出来（哈希比对看不见这种状态）")
    _assert_eq(hi.files_missing_from_index(scan, {"a.txt": 5, "b.txt": 1, "c.txt": 2}), [],
               "都有 chunk 时不误报")


def test_run_once_repairs_missing_and_holds_snapshot_on_failure():
    col = _FakeCollection({"a.txt": ["x"], "b.txt": []})
    restore = _patch(get_vector_store=lambda: _FakeStore(col),
                     ingest_file=lambda p, source_tag="": ({"status": "ok", "chunks": 2}
                                                           if p.endswith("a.txt") else
                                                           {"status": "error", "message": "boom"}))
    orig = {k: getattr(hi, k) for k in ("scan_files", "load_snapshot", "save_snapshot",
                                        "append_kb_metadata", "rebuild_sparse_index", "chunks_by_file")}
    saved = {}
    hi.scan_files = lambda d: {"a.txt": "s1", "b.txt": "s2"}          # 磁盘无变化
    hi.load_snapshot = lambda: {"a.txt": "s1", "b.txt": "s2"}          # 快照也一致 → 哈希视角"无变更"
    hi.chunks_by_file = lambda names: {"a.txt": 1, "b.txt": 0}         # 但 b.txt 索引里是 0
    hi.save_snapshot = lambda snap: saved.update(snap)
    hi.append_kb_metadata = lambda *a, **k: None
    hi.rebuild_sparse_index = lambda: None
    try:
        import tools.redis_store as rs
        orig_lock = (rs.acquire_lock, rs.release_lock)
        rs.acquire_lock, rs.release_lock = (lambda name: True), (lambda name: None)
        try:
            changed_flag = hi.run_once("data/knowledge")
        finally:
            rs.acquire_lock, rs.release_lock = orig_lock
        _assert(changed_flag, "哈希没变但索引缺块 → 仍应执行一次同步（自愈）")
        _assert_eq(saved, {"a.txt": "s1"}, "摄取失败的 b.txt 不推进快照，下轮重试")
    finally:
        restore()
        for k, v in orig.items():
            setattr(hi, k, v)


def test_change_detail_fields():
    """变更明细要能回答「谁什么时候动过」：mtime + 大小/行数 + git 状态。"""
    path = os.path.join("data", "knowledge", "不染一尘具体型号.txt")
    detail = hi.change_detail(path)
    _assert("mtime=" in detail, "含修改时间（定位「什么时候动过」）")
    _assert("B /" in detail and "行" in detail, "含大小与行数（乱码/半截保存会明显异常）")
    _assert("git=" in detail, "含 git 状态（未提交=有人在动）")
    _assert_in("未提交" if "未提交" in detail else "已提交", detail, "git 状态是闭集里的取值")
    _assert("读取文件信息失败" in hi.change_detail("data/knowledge/不存在.txt"), "文件没了不抛异常")


def test_run_once_logs_change_detail():
    """摄取之前就要把变更明细打出来（失败也留痕）。"""
    calls = []
    orig_info = hi.logger.info
    hi.logger.info = lambda msg, *a: calls.append(msg % a if a else msg)
    orig = {k: getattr(hi, k) for k in ("scan_files", "load_snapshot", "save_snapshot", "append_kb_metadata",
                                        "rebuild_sparse_index", "chunks_by_file", "apply_changes")}
    hi.scan_files = lambda d: {"不染一尘具体型号.txt": "s2"}       # 用真实文件，好验证明细里的 mtime
    hi.load_snapshot = lambda: {"不染一尘具体型号.txt": "s1"}      # 哈希变了 → changed
    hi.chunks_by_file = lambda names: {"不染一尘具体型号.txt": 1}
    hi.save_snapshot = lambda snap: None
    hi.append_kb_metadata = lambda *a, **k: None
    hi.rebuild_sparse_index = lambda: None
    hi.apply_changes = lambda *a, **k: {"added": 0, "changed": 0, "removed": 0,
                                        "failed": ["不染一尘具体型号.txt"]}
    try:
        import tools.redis_store as rs
        orig_lock = (rs.acquire_lock, rs.release_lock)
        rs.acquire_lock, rs.release_lock = (lambda name: True), (lambda name: None)
        try:
            hi.run_once("data/knowledge")
        finally:
            rs.acquire_lock, rs.release_lock = orig_lock
        _assert(any("变更明细" in c and "changed 不染一尘具体型号.txt" in c for c in calls),
                "changed 的文件要有变更明细行（失败前就落日志）")
        _assert(any("变更明细" in c and "mtime=" in c for c in calls), "明细含 mtime")
    finally:
        hi.logger.info = orig_info
        for k, v in orig.items():
            setattr(hi, k, v)


class _AddStore(_FakeStore):
    """额外实现 add_documents（ingest_file 用它落库）。"""

    def __init__(self, col, boom=False):
        super().__init__(col)
        self.boom = boom
        self.added = []

    def add_documents(self, chunks):
        if self.boom:
            raise RuntimeError("embed boom")
        self.added += list(chunks)


def _tmp_kb_file(name="型号.txt"):
    """ingest_file 会先 isfile，用真实小文件满足它；内容能被编号条目分块器切出来。"""
    path = os.path.join("temp", name)
    os.makedirs("temp", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("1. **净白 S1** 参考价 1799\n")
    return path


def _ingest_patch(store):
    from langchain_core.documents import Document
    return _patch(get_vector_store=lambda: store,
                  get_file_sha256_hex=lambda p: "sha-x",
                  extract_file=lambda p: [Document(page_content="1. **净白 S1** 参考价 1799\n", metadata={})])


def test_ingest_file_replaces_old_chunks():
    """同一文件重复入库 = 替换：先抓旧 id，插成功再删（不叠加重复条目）。"""
    col = _FakeCollection({"型号.txt": ["old1", "old2"]})
    store = _AddStore(col)
    restore = _ingest_patch(store)
    path = _tmp_kb_file()
    try:
        r = vs.ingest_file(path)
        _assert_eq(r["status"], "ok", "入库成功")
        _assert(store.added, "新 chunk 已写入")
        _assert_eq(col.deleted_ids, ["old1", "old2"], "插成功后才删旧 id（替换语义）")
    finally:
        restore()
        os.remove(path)


def test_ingest_file_failure_keeps_old_chunks():
    """落库失败时旧 chunk 原样留着（与热更新同一条纪律：不留 0 chunk 空洞）。"""
    col = _FakeCollection({"型号.txt": ["old1"]})
    restore = _ingest_patch(_AddStore(col, boom=True))
    path = _tmp_kb_file()
    try:
        _assert_raises(RuntimeError, lambda: vs.ingest_file(path), "落库失败要抛出，不静默吞")
        _assert_eq(col.deleted_ids, [], "失败不删旧 chunk")
    finally:
        restore()
        os.remove(path)


TESTS = [
    test_changed_failure_keeps_old_chunks,
    test_changed_success_replaces_old_chunks,
    test_ingest_file_replaces_old_chunks,
    test_ingest_file_failure_keeps_old_chunks,
    test_removed_deletes_by_file_name,
    test_files_missing_from_index,
    test_run_once_repairs_missing_and_holds_snapshot_on_failure,
    test_change_detail_fields,
    test_run_once_logs_change_detail,
]


def run():
    reset()
    return run_tests("热更新测试（增量同步的删除/重入语义）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
