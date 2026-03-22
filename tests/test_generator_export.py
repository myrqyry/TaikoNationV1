from pathlib import Path

from taikonation.data.tokenization import TaikoTokenizer
from taikonation.generation.generator import save_osu_chart, normalize_export_tokens, validate_exported_osu


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
    assert hit_lines[0].endswith("1:0:0:70:")
    assert hit_lines[1].endswith("1:2:0:70:")


def test_save_osu_chart_supports_rolls_and_finisher(tmp_path: Path):
    tokenizer = TaikoTokenizer()
    token_ids = [
        tokenizer.vocab["roll_start"],
        tokenizer.vocab["don"],
        tokenizer.vocab["roll_end"],
        tokenizer.vocab["finisher"],
    ]
    output = tmp_path / "chart_roll.osu"

    save_osu_chart(
        token_ids,
        tokenizer,
        str(output),
        audio_filename="song.wav",
        bpm=120.0,
        offset_ms=1000,
    )

    content = output.read_text()
    lines = content.splitlines()
    slider_lines = [line for line in lines if ",2,0,B|256:192,1," in line]
    spinner_lines = [line for line in lines if ",8,0," in line]
    assert slider_lines, "Expected at least one drumroll (slider) hitobject"
    assert spinner_lines, "Expected at least one spinner/denden-style hitobject"


def test_save_osu_chart_clamps_hitsound_volume(tmp_path: Path):
    tokenizer = TaikoTokenizer()
    output = tmp_path / "chart_volume.osu"
    save_osu_chart(
        [tokenizer.vocab["don"]],
        tokenizer,
        str(output),
        audio_filename="song.wav",
        hitsound_volume=999,
    )
    line = [l for l in output.read_text().splitlines() if l.startswith("256,192,")][0]
    assert line.endswith("1:0:0:100:")


def test_normalize_export_tokens_balances_rolls():
    tokens = ["don", "roll_end", "roll_start", "ka"]
    normalized = normalize_export_tokens(tokens)
    assert normalized == ["don", "roll_start", "ka", "roll_end"]


def test_validate_exported_osu_detects_non_monotonic_times():
    content = """osu file format v14
[General]
Mode: 1
[Metadata]
Title:Test
[Difficulty]
SliderMultiplier:1.4
[TimingPoints]
0,500,4,2,1,70,1,0
[HitObjects]
256,192,1000,1,0,1:0:0:70:
256,192,900,1,0,1:0:0:70:
"""
    issues = validate_exported_osu(content)
    assert any("not monotonic" in issue for issue in issues)
