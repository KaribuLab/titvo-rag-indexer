"""Unit tests for CLI path+sha256 merge planning."""

from rag_indexer.application.cli_manifest_merge import plan_cli_merge


def _files(*pairs: tuple[str, str]):
    return [{"path": path, "sha256": sha, "size": 1} for path, sha in pairs]


def test_identical_inventory_is_noop():
    inventory = _files(("a.py", "aaa"), ("b.py", "bbb"))
    plan = plan_cli_merge(inventory, inventory)
    assert plan.inventory_identical
    assert plan.skip == frozenset({"a.py", "b.py"})
    assert plan.embed == frozenset()
    assert plan.drop == frozenset()


def test_changed_file_reembeds_only_that_path():
    prev = _files(("a.py", "aaa"), ("b.py", "bbb"))
    new = _files(("a.py", "aaa"), ("b.py", "BBB"))
    plan = plan_cli_merge(prev, new)
    assert not plan.inventory_identical
    assert plan.skip == frozenset({"a.py"})
    assert plan.embed == frozenset({"b.py"})
    assert plan.drop == frozenset()


def test_absent_path_is_dropped():
    prev = _files(("a.py", "aaa"), ("gone.py", "ggg"))
    new = _files(("a.py", "aaa"))
    plan = plan_cli_merge(prev, new)
    assert plan.skip == frozenset({"a.py"})
    assert plan.embed == frozenset()
    assert plan.drop == frozenset({"gone.py"})


def test_new_path_is_embedded():
    prev = _files(("a.py", "aaa"))
    new = _files(("a.py", "aaa"), ("c.py", "ccc"))
    plan = plan_cli_merge(prev, new)
    assert plan.embed == frozenset({"c.py"})
    assert plan.skip == frozenset({"a.py"})
    assert plan.drop == frozenset()


def test_first_run_embeds_all():
    new = _files(("a.py", "aaa"), ("b.py", "bbb"))
    plan = plan_cli_merge(None, new)
    assert plan.embed == frozenset({"a.py", "b.py"})
    assert plan.skip == frozenset()
    assert plan.drop == frozenset()
