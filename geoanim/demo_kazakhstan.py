#!/usr/bin/env python3
"""Demo Short: "Kazakhstan has a coastline, yet it's landlocked".

Usage (paths relative to a working dir that holds data/ and voice/):
  python3 demo_kazakhstan.py --data data --voice voice --hand hand.png --out kazakhstan_short.mp4

voice/ holds line1.wav .. line5.wav (one narration line each).
"""
import argparse
import os
import subprocess
import wave

import geoanim as ga

LINES = [
    "Kazakhstan has a coastline, yet it's the largest landlocked country on Earth.",
    "Its beaches are on the Caspian Sea, and the Caspian isn't really a sea.",
    "It has no connection to the ocean at all, which makes it the largest lake in the world.",
    "And it's massive. The Caspian is even bigger than the whole of Germany.",
    "So Kazakhstan has beaches, and it's still landlocked. Geography is weird.",
]
GAP = 0.35   # pause between lines
LEAD = 0.4   # silence before the first line
TAIL = 1.0   # hold after the last line


def wav_dur(p):
    with wave.open(p) as w:
        return w.getnframes() / w.getframerate()


def build_audio(voice_dir, work):
    """Normalise each line, lay them out with gaps, return (audio path, [(start, end)])."""
    spans, parts, t = [], [], LEAD
    for i in range(1, len(LINES) + 1):
        src = os.path.join(voice_dir, f"line{i}.wav")
        norm = os.path.join(work, f"n{i}.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "48000",
                        "-af", "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                               "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                               "loudnorm=I=-16:TP=-1.5:LRA=11", norm], check=True)
        d = wav_dur(norm)
        spans.append((t, t + d))
        parts.append((t, norm))
        t += d + GAP
    total = spans[-1][1] + TAIL
    inputs, filt = [], []
    for k, (st, p) in enumerate(parts):
        inputs += ["-i", p]
        filt.append(f"[{k}:a]adelay={int(st * 1000)}|{int(st * 1000)}[d{k}]")
    mix = "".join(f"[d{k}]" for k in range(len(parts)))
    filt.append(f"{mix}amix=inputs={len(parts)}:normalize=0,apad,atrim=0:{total:.3f}[a]")
    out = os.path.join(work, "voice_track.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(filt),
                    "-map", "[a]", "-ar", "48000", out], check=True)
    return out, spans, total


def caption_chunks(audio, spans, max_words=3):
    """Word timestamps from faster-whisper; displayed words come from LINES (authored text)."""
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio, language="en", word_timestamps=True, vad_filter=False)
    words = [w for s in segs for w in s.words]
    chunks = []
    for (s, e), line in zip(spans, LINES):
        authored = line.split()
        ws = [w for w in words if s - 0.25 <= (w.start + w.end) / 2 <= e + 0.25]
        if len(ws) == len(authored):
            times = [(w.start, w.end) for w in ws]
        else:  # fall back to even spacing inside the line's span
            step = (e - s) / len(authored)
            times = [(s + i * step, s + (i + 1) * step) for i in range(len(authored))]
        i = 0
        while i < len(authored):
            j = min(i + max_words, len(authored))
            for k in range(i, j):  # break after punctuation
                if authored[k][-1] in ",.?!" and k + 1 < j:
                    j = k + 1
                    break
            chunks.append((times[i][0], times[j - 1][1] + 0.05, " ".join(authored[i:j])))
            i = j
    # hold each chunk until the next one starts (no flicker inside a line)
    out = []
    for n, (a, b, txt) in enumerate(chunks):
        nxt = chunks[n + 1][0] if n + 1 < len(chunks) else b + 0.4
        out.append((a, nxt if nxt - b < 0.5 else b + 0.15, txt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--voice", default="voice")
    ap.add_argument("--hand", default="hand.png")
    ap.add_argument("--out", default="kazakhstan_short.mp4")
    ap.add_argument("--work", default="work_kz")
    ap.add_argument("--preview", type=lambda s: [float(x) for x in s.split(",")], default=None,
                    help="comma-separated times: save still frames instead of rendering the video")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    audio, S, total = build_audio(a.voice, a.work)
    print("line spans:", [(round(s, 2), round(e, 2)) for s, e in S], "total", round(total, 2))
    caps = caption_chunks(audio, S)

    tex = ga.Texture(ga.build_texture(os.path.join(a.data, "earth4k.jpg"),
                                      os.path.join(a.data, "sr50/NE1_50M_SR_W/NE1_50M_SR_W.tif"),
                                      out_path=os.path.join(a.data, "texture8k.jpg")))
    geo = ga.Geo(os.path.join(a.data, "countries10.geojson"), os.path.join(a.data, "marine50.geojson"))
    kz = geo.get("Kazakhstan")
    caspian = geo.get("Caspian Sea")
    germany = geo.get("Germany")

    s1, s2, s3, s4, s5 = S
    cam = ga.Camera([
        (0.0, 62, 38, 95),                    # wide Eurasia
        (s1[0] + 1.6, 66.5, 47.5, 34),        # push in on Kazakhstan
        (s2[0] + 0.1, 66.5, 47.5, 34),
        (s2[0] + 1.3, 51.5, 42.5, 19),        # over to the Caspian
        (s3[0] + 0.1, 51.5, 42.5, 19),
        (s3[0] + 1.3, 45.0, 42.5, 32),        # pull back: Caspian vs Black Sea
        (s4[0] + 0.1, 45.0, 42.5, 32),
        (s4[0] + 1.2, 54.0, 42.5, 25),        # Caspian vs Germany
        (s5[0] + 0.1, 54.0, 42.5, 25),
        (s5[0] + 1.3, 63.0, 46.5, 36),        # back to Kazakhstan
        (total, 63.5, 46.5, 34),
    ])

    orange, red, blue = (30, 120, 245), (50, 50, 235), (220, 120, 40)  # BGR
    hl_kz = ga.Highlight(kz, s1[0] + 0.9, total, color=orange, dim_after=s2[0] + 0.6)
    hl_casp = ga.Highlight(caspian, s2[0] + 0.7, s5[0] + 0.6, color=red, style="glow")
    hl_ger = ga.Highlight(germany, s4[0] + 1.3, s5[0] + 0.5, color=blue,
                          move_to=(59.5, 42.0), move_t=(s4[0] + 1.3, s4[0] + 2.4))
    hl_kz2 = ga.Highlight(kz, s5[0] + 0.8, total, color=orange, glow=False, fill_alpha=0.6, outline=None)
    els = [
        ga.Vignette(0.38),
        hl_kz, hl_casp, hl_ger, hl_kz2,
        ga.Label("KAZAKHSTAN", kz, s1[0] + 1.3, s2[0] + 0.8, offset=(0, -170)),
        ga.Eyes(hl_kz, s1[0] + 1.8, s2[1], size=170, mood="happy", look=(52.0, 42.0)),
        ga.Sprite(a.hand, (52.5, 43.5), s2[0] + 1.4, s3[0] + 0.3, height=330, anchor=(0.74, 0.0)),
        ga.Label("Caspian Sea?", caspian, s2[0] + 2.0, s3[0] + 0.4, offset=(0, -300)),
        ga.Route([(49.5, 41.5), (46.0, 43.5), (41.5, 44.6), (37.5, 44.0)], s3[0] + 1.2, s4[0] + 0.3,
                 draw_time=1.4),
        ga.XMark((43.6, 44.3), s3[0] + 2.4, s4[0] + 0.3, size=120),
        ga.Label("WORLD'S LARGEST LAKE", (45.0, 38.6), s3[1] - 1.6, s4[0] + 0.3, size=44, offset=(0, 0),
                 bg=(230, 50, 50, 240), fill=(255, 255, 255)),
        ga.Label("371,000 km²", caspian, s4[0] + 2.6, s5[0] + 0.5, size=42, offset=(0, 190)),
        ga.Label("357,000 km²", hl_ger, s4[0] + 2.8, s5[0] + 0.5, size=42, offset=(0, 150)),
        ga.Label("GERMANY", hl_ger, s4[0] + 2.4, s5[0] + 0.5, size=40, offset=(0, -190),
                 bg=(40, 120, 220, 235), fill=(255, 255, 255)),
        ga.Eyes(hl_ger, s4[0] + 2.0, s5[0] + 0.5, size=110, mood="shock", look=(50.5, 42.0)),
        ga.Eyes(hl_kz, s5[0] + 1.0, total, size=170, mood="smug", wink_at=s5[1] - 0.9),
        ga.Label("STILL LANDLOCKED", kz, s5[0] + 2.2, total, size=46, offset=(0, 190),
                 bg=(245, 120, 30, 240), fill=(255, 255, 255)),  # label colours are RGBA
        ga.Captions(caps, y=0.78, size=66),
    ]
    if a.preview:
        ga.render_stills(a.preview, a.work, tex, cam, els)
        return
    ga.render(a.out, tex, cam, els, total, audio_path=audio)
    print("DONE", a.out)


if __name__ == "__main__":
    main()
