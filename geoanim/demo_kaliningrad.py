#!/usr/bin/env python3
"""Short: "This piece of Russia is cut off from Russia" (Kaliningrad).

  python3 demo_kaliningrad.py --data data --voice voice_kgd --hand hand.png --out kaliningrad_short.mp4
  python3 demo_kaliningrad.py ... --preview 2,6,10      # stills only
"""
import argparse
import os

import geoanim as ga

LINES = [
    "This piece of Russia is completely cut off from the rest of Russia.",
    "Kaliningrad sits between Poland and Lithuania, about a thousand kilometres from Moscow.",
    "Until 1945 it was the German city of Königsberg, but after World War Two the Soviet Union took it.",
    "Back then it didn't matter, because Lithuania was part of the Soviet Union too.",
    "But when the Soviet Union collapsed in 1991, Lithuania walked away, and Kaliningrad was left stranded.",
    "Today it's surrounded by NATO, and it's still home to Russia's Baltic Fleet.",
]

RED, BLUE, GOLD, GREEN = (50, 50, 225), (205, 120, 40), (40, 175, 235), (70, 170, 60)  # BGR
KGD_CITY, MOSCOW, BALTIYSK = (20.51, 54.71), (37.62, 55.75), (19.92, 54.65)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--voice", default="voice_kgd")
    ap.add_argument("--hand", default="hand.png")
    ap.add_argument("--out", default="kaliningrad_short.mp4")
    ap.add_argument("--work", default="work_kgd")
    ap.add_argument("--preview", type=lambda s: [float(x) for x in s.split(",")], default=None)
    a = ap.parse_args()

    audio, S, total = ga.build_voice_track(len(LINES), a.voice, a.work)
    print("line spans:", [(round(s, 2), round(e, 2)) for s, e in S], "total", round(total, 2), flush=True)
    caps = ga.caption_chunks(audio, S, LINES)

    tex = ga.Texture(ga.build_texture(os.path.join(a.data, "earth4k.jpg"),
                                      os.path.join(a.data, "sr50/NE1_50M_SR_W/NE1_50M_SR_W.tif"),
                                      out_path=os.path.join(a.data, "texture8k.jpg")))
    box = (12.0, 50.0, 40.0, 60.5)  # sharper NASA imagery for the Baltic region
    tex.add_patch(ga.fetch_gibs(*box, px_per_deg=150, out_path=os.path.join(a.data, "gibs_baltic.jpg")), *box)

    countries = ga.Geo(os.path.join(a.data, "countries10.geojson"))
    regions = ga.Geo(os.path.join(a.data, "admin1_10m.geojson"))
    kgd = regions.get("Kaliningrad")
    poland, lith = countries.get("Poland"), countries.get("Lithuania")
    latvia, eston, belarus = countries.get("Latvia"), countries.get("Estonia"), countries.get("Belarus")

    s1, s2, s3, s4, s5, s6 = S
    mid3 = s3[0] + (s3[1] - s3[0]) * 0.55
    walk = s5[0] + (s5[1] - s5[0]) * 0.5
    cam = ga.Camera([
        (0.0, 24.0, 53.5, 30),
        (s1[0] + 1.8, 21.2, 54.75, 9),        # zoom on the exclave
        (s2[0] + 0.1, 21.2, 54.75, 9),
        (s2[0] + 1.1, 29.5, 54.9, 22),        # pull back: Poland, Lithuania, Moscow
        (s3[0] + 0.1, 29.5, 54.9, 22),
        (s3[0] + 1.0, 21.2, 54.75, 8.5),      # back in: Königsberg
        (s4[0] + 0.1, 21.2, 54.75, 8.5),
        (s4[0] + 1.0, 25.5, 56.2, 17),        # Baltic Soviet republics
        (s6[0] + 0.1, 25.5, 56.2, 17),
        (s6[0] + 1.0, 22.3, 54.4, 12),        # surrounded
        (total, 22.3, 54.4, 11.5),
    ])

    els = [ga.Vignette(0.38)]
    # 1-2: the exclave, its neighbours, the distance to Moscow
    k1 = ga.Highlight(kgd, s1[0] + 1.0, s3[0] + 0.3, color=RED)
    els += [
        k1,
        ga.Label("RUSSIA?!", kgd, s1[0] + 1.6, s2[0] + 0.6, offset=(0, -260)),
        ga.Eyes(k1, s1[0] + 2.0, s3[0] + 0.3, size=140, mood="worried", look=MOSCOW),
        ga.Highlight(poland, s2[0] + 1.0, s3[0] + 0.3, color=(230, 230, 230), fill_alpha=0.35, glow=False),
        ga.Highlight(lith, s2[0] + 1.3, s3[0] + 0.3, color=(230, 230, 230), fill_alpha=0.35, glow=False),
        ga.Label("POLAND", poland, s2[0] + 1.2, s3[0] + 0.3, size=40, offset=(0, 0)),
        ga.Label("LITHUANIA", lith, s2[0] + 1.5, s3[0] + 0.3, size=40, offset=(0, 0)),
        ga.Route([KGD_CITY, (25.0, 55.6), (31.0, 56.0), MOSCOW], s2[1] - 2.3, s3[0] + 0.3, draw_time=1.4),
        ga.Label("MOSCOW", MOSCOW, s2[1] - 1.0, s3[0] + 0.3, size=40, offset=(0, -70),
                 bg=(50, 50, 225, 240), fill=(255, 255, 255)),
        ga.Label("~1,000 km", (30.0, 56.6), s2[1] - 1.2, s3[0] + 0.3, size=44, offset=(0, -40)),
    ]
    # 3: Königsberg -> Soviet
    kg = ga.Highlight(kgd, s3[0] + 0.3, mid3 + 0.2, color=GOLD)
    k3 = ga.Highlight(kgd, mid3, s4[0] + 0.8, color=RED)
    els += [
        kg, k3,
        ga.Label("KÖNIGSBERG", kgd, s3[0] + 0.9, mid3 + 0.2, size=50, offset=(0, -280),
                 bg=(40, 175, 235, 245), fill=(30, 30, 30)),
        ga.Label("GERMANY", kgd, s3[0] + 1.3, mid3 + 0.2, size=40, offset=(0, 260)),
        ga.Label("USSR", kgd, mid3 + 0.3, s4[0] + 0.8, size=56, offset=(0, -280),
                 bg=(50, 50, 225, 245), fill=(255, 255, 255)),
        ga.Eyes(k3, mid3 + 0.4, s4[0] + 0.8, size=150, mood="smug"),
    ]
    # 4: the Soviet Baltic, all one country
    k4 = ga.Highlight(kgd, s4[0] + 0.6, s6[0] + 0.4, color=RED)
    soviet = [ga.Highlight(sh, s4[0] + 0.8 + 0.25 * i, walk if sh is lith else s5[0] + 1.0, color=RED)
              for i, sh in enumerate([lith, latvia, eston, belarus])]
    els += [k4] + soviet + [
        ga.Label("SOVIET UNION", (26.5, 57.6), s4[0] + 1.6, s5[0] + 1.0, size=52,
                 bg=(50, 50, 225, 245), fill=(255, 255, 255)),
        ga.Sprite(a.hand, (23.8, 55.2), s4[1] - 2.2, s5[0] + 0.3, height=300, anchor=(0.74, 0.0)),
    ]
    # 5: 1991, Lithuania leaves, the exclave is stranded
    els += [
        ga.Label("1991", (26.5, 57.6), s5[0] + 1.1, s6[0] + 0.3, size=64,
                 bg=(255, 255, 255, 240), fill=(30, 30, 30)),
        ga.Highlight(lith, walk, s6[0] + 0.3, color=GREEN),
        ga.Label("LITHUANIA", lith, walk + 0.2, s6[0] + 0.3, size=40, offset=(0, 0)),
        ga.Route([KGD_CITY, (23.5, 55.0), (26.5, 54.6), (28.0, 53.8)], walk + 0.4, s6[0] + 0.3, draw_time=1.0),
        ga.XMark((24.2, 54.95), walk + 1.2, s6[0] + 0.3, size=120),
        ga.Eyes(k4, s5[0] + 0.3, s6[0] + 0.4, size=120, mood="shock", look=(28.0, 54.0)),
    ]
    # 6: surrounded by NATO, Baltic Fleet
    k6 = ga.Highlight(kgd, s6[0] + 0.3, total, color=RED)
    els += [
        ga.Highlight(poland, s6[0] + 0.8, total, color=BLUE, style="glow"),
        ga.Highlight(lith, s6[0] + 1.0, total, color=BLUE, style="glow"),
        k6,
        ga.Label("NATO", poland, s6[0] + 1.1, total, size=52, offset=(0, 0),
                 bg=(205, 120, 40, 245), fill=(255, 255, 255)),
        ga.Label("NATO", lith, s6[0] + 1.3, total, size=52, offset=(0, 0),
                 bg=(205, 120, 40, 245), fill=(255, 255, 255)),
        ga.Eyes(k6, s6[0] + 0.5, total, size=130, mood="angry", look=(23.0, 53.0)),
        ga.Label("BALTIC FLEET", BALTIYSK, s6[1] - 1.8, total, size=44, offset=(-30, -150),
                 bg=(50, 50, 225, 245), fill=(255, 255, 255)),
        ga.Captions(caps, y=0.78, size=66),
    ]

    if a.preview:
        ga.render_stills(a.preview, a.work, tex, cam, els)
        return
    ga.render(a.out, tex, cam, els, total, audio_path=audio)
    print("DONE", a.out)


if __name__ == "__main__":
    main()
