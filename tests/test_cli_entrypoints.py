from taikonation import cli


def test_generate_entrypoint(monkeypatch):
    called = {"ok": False}

    def _fake_main():
        called["ok"] = True

    monkeypatch.setattr("taikonation.cli.generate_main", _fake_main)
    cli.generate()
    assert called["ok"] is True


def test_serve_entrypoint(monkeypatch):
    calls = []

    def _fake_run(app, host, port, reload):
        calls.append((app, host, port, reload))

    monkeypatch.setattr("uvicorn.run", _fake_run)
    cli.serve()
    assert calls == [("web.server_fastapi:socket_app", "127.0.0.1", 5000, False)]
