# geoanim — animated maps for geography Shorts

A renderer for vertical (1080×1920) geography explainer videos in the
"satellite map + glowing countries + cartoon eyes" style. Plain Python
(numpy, OpenCV, Pillow) that pipes frames straight into ffmpeg — no After
Effects needed.

## What it can do

| Element | What it does |
|---|---|
| `Camera` | smooth moves and zooms across the map (keyframes: time, lon, lat, frame height in degrees) |
| `Highlight` | fills a country or sea with color, glow and outline; `style="glow"` gives a glowing outline; `move_to=` slides a shape to another spot **at true size** (compare areas) |
| `Eyes` | cartoon eyes on a country: `happy`, `angry`, `worried`, `smug`, `shock`, blink, wink, look towards a point |
| `Label` | rounded pill label pinned to a country or point |
| `Route` | dashed line drawn progressively along points |
| `XMark` | red ✕ that pops in |
| `Sprite` | any PNG (pointing hand, flag, icon) pinned to a point, with a bob animation |
| `Captions` | word-timed captions (Whisper timings, authored text) |
| `Vignette` | darkened frame edges |

## Data (public domain — fine for monetised videos)

```
data/
  earth4k.jpg              NASA Blue Marble, 4096×2048
                           https://raw.githubusercontent.com/turban/webgl-earth/master/images/2_no_clouds_4k.jpg
  sr50/NE1_50M_SR_W/...tif Natural Earth shaded relief (relief detail)
                           https://naciscdn.org/naturalearth/50m/raster/NE1_50M_SR_W.zip
  countries10.geojson      https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson
  marine50.geojson         seas, incl. the Caspian:
                           https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_geography_marine_polys.geojson
Montserrat-ExtraBold.ttf   font for labels and captions (OFL)
```

Requirements: `python3`, `numpy`, `opencv-python`, `pillow`, `faster-whisper`
(captions), `ffmpeg`.

### Sharper close-ups

The base map is 8K, which gets soft when you zoom into small countries. For
close-ups, add a detail patch from NASA GIBS (Blue Marble, public domain); it
is colour-matched to the base map and fades in as the camera zooms:

```python
box = (12.0, 50.0, 40.0, 60.5)          # lon0, lat0, lon1, lat1
tex.add_patch(ga.fetch_gibs(*box, px_per_deg=150, out_path="data/gibs_baltic.jpg"), *box)
```

### Colours

Map layers (`Highlight`, `Route`, `XMark`) take **BGR** tuples (OpenCV);
`Label` pills take **RGB(A)** (Pillow).

### Examples

* `demo_kazakhstan.py` — Kazakhstan's coastline on the Caspian (23 s)
* `demo_kaliningrad.py` — Kaliningrad, the Russian exclave (29 s); uses the
  shared `ga.build_voice_track()` / `ga.caption_chunks()` helpers and a GIBS
  detail patch — the best template to copy.

## Making a new video

1. Write the script as separate lines (one line ≈ one scene).
2. Voice each line → `voice/line1.wav … lineN.wav`.
3. Copy `demo_kazakhstan.py`, replace `LINES` and the scene list in `main()`.
   All times are tied to the line spans `s1, s2, …` (start, end of each line),
   so the animation follows the voice.
4. Check frames quickly:
   `python3 demo_kazakhstan.py --preview 2,5,9,14`
5. Render:
   `python3 demo_kazakhstan.py --out short.mp4`

A 23-second Short renders in about a minute on a regular CPU.
