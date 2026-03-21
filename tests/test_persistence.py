from pathlib import Path

from web.persistence import StudioStore


def test_store_chart_and_evaluation(tmp_path: Path):
    store = StudioStore(tmp_path / "studio.sqlite3")

    chart = store.create_chart(
        {
            "title": "Song",
            "artist": "Artist",
            "difficulty": "oni",
            "bpm": 180,
            "genre": "electronic",
            "rating": 0,
            "plays": 0,
            "created_at": "2026-03-21T00:00:00",
        }
    )

    assert chart["id"] > 0
    unrated = store.get_unrated_chart()
    assert unrated is not None
    assert unrated["id"] == chart["id"]

    ok = store.submit_evaluation(
        chart_id=chart["id"],
        fun=4,
        musicality=5,
        playability=3,
        coherence=4,
        comments="good flow",
        created_at="2026-03-21T00:01:00",
    )
    assert ok is True

    charts = store.list_charts()
    assert charts[0]["rating"] == 4.0
    evaluations = store.list_evaluations()
    assert evaluations[0]["comments"] == "good flow"


def test_store_task_lifecycle(tmp_path: Path):
    store = StudioStore(tmp_path / "studio.sqlite3")
    task = store.create_task(
        task_type="generation",
        payload={"title": "Song"},
        created_at="2026-03-21T00:00:00",
    )
    assert task["status"] == "queued"
    assert task["progress"] == 0
    assert task["payload"]["title"] == "Song"

    updated = store.update_task(
        task["id"],
        status="running",
        progress=40,
        message="working",
        updated_at="2026-03-21T00:01:00",
    )
    assert updated is True
    task_after = store.get_task(task["id"])
    assert task_after is not None
    assert task_after["status"] == "running"
    assert task_after["progress"] == 40

    store.update_task(
        task["id"],
        status="completed",
        progress=100,
        message="done",
        result={"chart_id": 7},
        updated_at="2026-03-21T00:02:00",
    )
    task_done = store.get_task(task["id"])
    assert task_done is not None
    assert task_done["result"]["chart_id"] == 7
