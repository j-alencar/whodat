import time

from whodat.tui import browser


def test_the_browser_runs_detached_from_the_terminal(tmp_path, monkeypatch):
    log = tmp_path / "log"
    script = tmp_path / "fake-browser"
    script.write_text(
        f'#!/bin/sh\n[ -t 0 ] && t=tty || t=detached\necho "$t $1" > {log}\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv("BROWSER", str(script))

    browser()("https://youtu.be/v?t=1")

    for _ in range(50):
        if log.exists() and log.read_text():
            break
        time.sleep(0.1)
    assert log.read_text().split() == ["detached", "https://youtu.be/v?t=1"]
