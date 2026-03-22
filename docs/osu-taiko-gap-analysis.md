# osu!taiko Research Notes: Potentially Ignored Mapping Elements

_Last reviewed: 2026-03-22 (UTC), using official osu! wiki pages._

## Why this note exists

Our generator currently writes a very minimal `.osu` output (`Mode: 1`, static difficulty settings, and fixed 200 ms spacing between notes), which is good for smoke tests but omits a number of taiko-relevant authoring elements that affect authenticity and rankability.

## Evidence from current code

In `taikonation/generation/generator.py`:

- The exported file writes only `[General]`, `[Metadata]`, `[Difficulty]`, and `[HitObjects]`, with no `[TimingPoints]` or `[Events]` sections.
- Hit object times are generated from a fixed interval (`time_interval = 200`) rather than song timing/BPM maps.
- Every generated object is emitted as a hit circle (`type = 1`), so drumroll/sliders and denden/spinners are not represented.
- Hitsounds are limited to clap/finish bit toggles inferred from token names.

## Elements we are likely ignoring

## 1) Timing-point authoring (red/green lines)

The osu! timing model supports uninherited and inherited timing points for BPM/time signatures, slider velocity changes, hitsound volume, sampleset switching, and kiai toggles.

**Potential gap impact**

- No per-section timing changes.
- No SV/control-line readability tuning for dense passages.
- No kiai toggles in map data.

## 2) Full taiko object vocabulary

osu!taiko gameplay explicitly includes normal/big don/kat notes, drumrolls, and denden (spinner/shaker style objects).

**Potential gap impact**

- Generated charts may underrepresent hold/release expression and section contrast.
- Difficulty expression relies mostly on note density and color patterns only.

## 3) Hit sample and sampleset control

The `.osu` format supports per-object `hitSample` (`normalSet:additionSet:index:volume:filename`) and the taiko mode uses hitsounds to encode note color/size semantics.

**Potential gap impact**

- Limited control over taiko sound identity and mapper-style emulation.
- Missing section-level hitsound dynamics (volume/sampleset changes).

## 4) Metadata and ranking-facing polish fields

While basic metadata exists, a ranked-style map typically includes richer fields and sectioning context (`[TimingPoints]`, optional storyboard/events, preview timing choices, etc.).

**Potential gap impact**

- Output is usable for experimentation but not close to ranked-quality packaging.

## 5) Conversion-sensitive rhythm details

The taiko ecosystem has known conversion behaviors involving short sliders, low BPM drumroll rhythm, and slider tick rate behavior.

**Potential gap impact**

- Generated structures may not transfer predictably under conversion-like assumptions.

## Recommended implementation order

1. ✅ **Add `[TimingPoints]` emission** and align note times to beat-derived timestamps (not fixed 200 ms).
2. 🟡 **Add drumroll/denden token support** and proper object writing in exporter (now partially implemented; needs gameplay validation/tuning).
3. 🟡 **Add `hitSample` modeling** (volume/sampleset/index controls) and configurable taiko sample profiles (basic volume/set routing now present; profile-level control still pending).
4. **Add section annotations (kiai + SV lanes)** from model outputs or post-processing heuristics.
5. 🟡 **Add validation against osu! file-format constraints** before export (token-stream normalization started; full parser-level validation pending).

## Suggested acceptance checks

- Parse generated file and assert required sections exist.
- Verify hit object timestamps follow timing points.
- Verify don/kat/big/drumroll/denden object presence distributions.
- Verify hitSound/hitSample fields map to intended token semantics.
- Regression-test output in osu! client for visual/readability sanity.

## External references used

- osu! file format (`.osu`) wiki page: https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29
- osu! beatmap editor timing tab (timing-point behavior): https://osu.ppy.sh/wiki/en/Client/Beatmap_editor/Timing
- osu! hitsound guide (including taiko-specific note semantics): https://osu.ppy.sh/wiki/en/Beatmapping/Hitsound
- osu!taiko game mode overview (notes, drumrolls, denden): https://osu.ppy.sh/wiki/en/Game_mode/osu%21taiko
