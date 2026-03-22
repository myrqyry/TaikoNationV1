from pathlib import Path

from taikonation.data.tokenization import TaikoTokenizer
from taikonation.generation.generator import save_osu_chart


def test_save_osu_chart_writes_timing_points(tmp_path: Path):
    tokenizer = TaikoTokenizer()
    token_ids = [
        tokenizer.vocab["don"],
        tokenizer.vocab["ka"],
        tokenizer.vocab["big_don"],
    ]
    output = tmp_path / "chart.osu"

    save_osu_chart(
        token_ids,
        tokenizer,
        str(output),
        audio_filename="song.wav",
        title="Song",
        artist="Artist",
        bpm=150.0,
        offset_ms=800,
    )

    content = output.read_text()
    assert "[TimingPoints]" in content
    assert "800,400.000000" in content  # 60000/150
    assert "[HitObjects]" in content

    # Ensure timestamps start at offset and advance by half-beat (200ms at 150 BPM).
    hit_lines = [line for line in content.splitlines() if line.startswith("256,192,")]
    assert len(hit_lines) == 3
    times = [int(line.split(",")[2]) for line in hit_lines]
    assert times == [800, 1000, 1200]
