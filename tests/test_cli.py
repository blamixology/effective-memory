import json
import os

import pytest

from effective_memory import cli


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "cli_test.db")


def run(argv, db_path):
    return cli.main(["--db", db_path, *argv])


def test_add_and_list(db_path, capsys):
    run(["add", "the user likes coffee", "--tags", "preference"], db_path)
    out = capsys.readouterr().out
    assert "added memory #1" in out

    run(["list"], db_path)
    out = capsys.readouterr().out
    assert "the user likes coffee" in out
    assert "1/1" in out


def test_update_and_delete(db_path, capsys):
    run(["add", "original text"], db_path)
    capsys.readouterr()

    run(["update", "1", "--content", "edited text"], db_path)
    assert "updated memory #1" in capsys.readouterr().out

    run(["recall", "edited"], db_path)
    assert "edited text" in capsys.readouterr().out

    run(["delete", "1"], db_path)
    assert "deleted memory #1" in capsys.readouterr().out

    run(["list"], db_path)
    assert "no memories match" in capsys.readouterr().out


def test_recall_context_review(db_path, capsys):
    run(["add", "the user prefers dark mode"], db_path)
    capsys.readouterr()

    run(["recall", "what theme does the user like"], db_path)
    assert "dark mode" in capsys.readouterr().out

    run(["context", "user preferences", "--budget", "50"], db_path)
    assert "dark mode" in capsys.readouterr().out

    run(["review", "--threshold", "0.99"], db_path)
    out = capsys.readouterr().out
    assert "dark mode" in out or "nothing due for review" in out


def test_tags(db_path, capsys):
    run(["add", "a", "--tags", "work,urgent"], db_path)
    run(["add", "b", "--tags", "work"], db_path)
    capsys.readouterr()

    run(["tags"], db_path)
    out = capsys.readouterr().out
    assert "work\t2" in out
    assert "urgent\t1" in out


def test_export_import_and_reindex(db_path, tmp_path, capsys):
    run(["add", "the user likes coffee", "--tags", "preference"], db_path)
    capsys.readouterr()

    dump_path = str(tmp_path / "dump.json")
    run(["export", "--out", dump_path], db_path)
    assert "exported 1 memories" in capsys.readouterr().out

    with open(dump_path) as f:
        dump = json.load(f)
    assert dump["memories"][0]["content"] == "the user likes coffee"

    fresh_db = str(tmp_path / "fresh.db")
    run(["import", dump_path], fresh_db)
    assert "imported 1 memories, 0 links" in capsys.readouterr().out

    run(["reindex"], fresh_db)
    assert "reindexed 1 memories" in capsys.readouterr().out


def test_compact_default_summarizer(db_path, capsys):
    run(["add", "the user likes coffee in the morning"], db_path)
    run(["add", "the user prefers coffee at breakfast"], db_path)
    capsys.readouterr()

    run(["compact", "--threshold", "0.99"], db_path)
    out = capsys.readouterr().out
    assert "compacted into" in out or "nothing to compact" in out


def test_stats(db_path, capsys):
    run(["add", "a"], db_path)
    run(["add", "b"], db_path)
    capsys.readouterr()

    run(["stats"], db_path)
    out = capsys.readouterr().out
    assert "total: 2" in out


def test_serve_generates_key_only_when_unset(monkeypatch, db_path):
    pytest.importorskip("uvicorn")
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    monkeypatch.delenv("EFFECTIVE_MEMORY_API_KEY", raising=False)
    cli.main(["--db", db_path, "serve"])
    assert os.environ["EFFECTIVE_MEMORY_API_KEY"] != ""

    monkeypatch.setenv("EFFECTIVE_MEMORY_API_KEY", "")
    cli.main(["--db", db_path, "serve"])
    assert os.environ["EFFECTIVE_MEMORY_API_KEY"] == ""  # explicit empty must stay disabled

    monkeypatch.delenv("EFFECTIVE_MEMORY_API_KEY", raising=False)
    cli.main(["--db", db_path, "serve", "--api-key", "mykey"])
    assert os.environ["EFFECTIVE_MEMORY_API_KEY"] == "mykey"

    monkeypatch.setenv("EFFECTIVE_MEMORY_API_KEY", "should-be-overridden")
    cli.main(["--db", db_path, "serve", "--no-auth"])
    assert os.environ["EFFECTIVE_MEMORY_API_KEY"] == ""
