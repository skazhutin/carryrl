"""Training provenance identifies source paths and contents with bounded I/O."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from carryrl.training import run


@pytest.fixture
def repository(tmp_path, monkeypatch):
    files = {
        "src/carryrl/training/run.py": "# training implementation\n",
        "src/carryrl/agent/model.py": "# policy implementation\n",
        "scripts/train.py": "# launch configuration one\n",
        "scripts/nested/evaluate.py": "# evaluation implementation\n",
        "uv.lock": "version = 1\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(run, "__file__", str(tmp_path / "src/carryrl/training/run.py"))

    def git_query(command, **kwargs):
        assert kwargs["cwd"] == tmp_path
        stdout = "abc123\n" if command[1] == "rev-parse" else " M scripts/train.py\n"
        return SimpleNamespace(stdout=stdout, returncode=0)

    monkeypatch.setattr(run.subprocess, "run", git_query)
    return tmp_path


def test_script_content_changes_identity_even_with_identical_dirty_status(repository):
    before = run.provenance()
    (repository / "scripts/train.py").write_text("# launch configuration two\n")
    after = run.provenance()
    assert before["git_commit"] == after["git_commit"] == "abc123"
    assert before["git_dirty"] == after["git_dirty"] == [" M scripts/train.py"]
    assert before["source_and_lock_sha256"] != after["source_and_lock_sha256"]
    assert before["source_identity_version"] == after["source_identity_version"] == (
        "carryrl.source.v2"
    )
    assert set(before) == {
        "git_commit", "git_dirty", "source_and_lock_sha256", "source_identity_version",
        "dependencies", "torch_mps_available", "mps_fallback_environment",
    }


@pytest.mark.parametrize("name", (
    "src/carryrl/agent/model.py", "scripts/nested/evaluate.py", "uv.lock",
))
def test_each_source_group_and_lock_content_are_hashed(repository, name):
    before = run.provenance()["source_and_lock_sha256"]
    path = repository / name
    path.write_text(path.read_text() + "# changed\n")
    assert run.provenance()["source_and_lock_sha256"] != before


def test_source_identity_is_independent_of_file_enumeration_order(repository, monkeypatch):
    before = run.provenance()["source_and_lock_sha256"]
    original_rglob = Path.rglob

    def reversed_rglob(path, pattern):
        return iter(reversed(list(original_rglob(path, pattern))))

    monkeypatch.setattr(Path, "rglob", reversed_rglob)
    assert run.provenance()["source_and_lock_sha256"] == before


def test_file_paths_are_part_of_source_identity(repository):
    before = run.provenance()["source_and_lock_sha256"]
    path = repository / "scripts/nested/evaluate.py"
    path.rename(path.with_name("renamed.py"))
    assert run.provenance()["source_and_lock_sha256"] != before


def test_identity_uses_relative_paths_and_ignores_non_python_artifacts(repository, tmp_path):
    before = run.provenance()["source_and_lock_sha256"]
    (repository / "scripts/cache.json").write_text('{"ignored": true}')
    (repository / "unrelated.py").write_text("# outside source roots\n")
    assert run.provenance()["source_and_lock_sha256"] == before
    relocated = tmp_path / "relocated"
    for source in (repository / "src").rglob("*.py"):
        target = relocated / source.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for source in (repository / "scripts").rglob("*.py"):
        target = relocated / source.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    (relocated / "uv.lock").write_bytes((repository / "uv.lock").read_bytes())
    assert run._source_and_lock_sha256(relocated) == before


def test_missing_lock_and_source_addition_change_identity(repository):
    before = run.provenance()["source_and_lock_sha256"]
    (repository / "uv.lock").unlink()
    without_lock = run.provenance()["source_and_lock_sha256"]
    assert without_lock != before
    (repository / "scripts/new.py").write_text("")
    assert run.provenance()["source_and_lock_sha256"] != without_lock


@pytest.mark.parametrize("payload", (b"", b"small file", bytes(range(256)) * 10241))
def test_file_sha256_matches_hashlib_with_only_bounded_reads(tmp_path, monkeypatch, payload):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(payload)
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __enter__(self):
            self.stream = original_open(path, "rb")
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            reads.append(size)
            return self.stream.read(size)

    def bounded_open(target, *args, **kwargs):
        assert target == path and args == ("rb",) and not kwargs
        return BoundedReader()

    def forbid_whole_file_read(*args, **kwargs):
        pytest.fail("file_sha256 must not call Path.read_bytes")

    monkeypatch.setattr(Path, "open", bounded_open)
    monkeypatch.setattr(Path, "read_bytes", forbid_whole_file_read)
    assert run.file_sha256(path) == hashlib.sha256(payload).hexdigest()
    assert len(reads) >= len(payload) // (1024 * 1024) + 1


def test_provenance_does_not_read_source_files_whole(repository, monkeypatch):
    def forbid_whole_file_read(*args, **kwargs):
        pytest.fail("provenance must use streaming hashes")

    monkeypatch.setattr(Path, "read_bytes", forbid_whole_file_read)
    assert len(run.provenance()["source_and_lock_sha256"]) == 64
