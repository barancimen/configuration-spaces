"""The interactive page: generation, and its in-browser sampler run under Node."""
import shutil
import subprocess

import pytest

from cspace.viewer import page
from helpers import EXAMPLES

HERE = EXAMPLES.parent / "tests"


def test_page_builds_and_samples(tmp_path):
    html = page([(EXAMPLES / "fourbar.yaml", {"a": "2", "b": "5", "c": "4", "d": "6"}),
                 (EXAMPLES / "door_stop.yaml", {}), (EXAMPLES / "rssr.yaml", {})], standalone=True)
    assert html.startswith("<!doctype html>") and "<title>Configuration Explorer</title>" in html
    out = tmp_path / "page.html"
    out.write_text(html)
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    run = subprocess.run(["node", str(HERE / "viewer_check.js"), str(out)], capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    lines = run.stdout.strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert "global 15/15" in line and "local 15/15" in line, line
    assert "freedoms 1" in lines[0] and "freedoms 2" in lines[2]


def test_fragment_has_no_document_skeleton():
    html = page([(EXAMPLES / "gears.yaml", {})], standalone=False)
    assert html.startswith("<title>") and "<html" not in html
