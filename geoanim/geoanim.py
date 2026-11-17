#!/usr/bin/env python3
"""geoanim — explainer-style animated map renderer for vertical Shorts.

Satellite-style world texture + smooth camera moves + glowing country
highlights, cartoon eyes, labels, dashed routes, X marks, a pointing hand
and word-timed captions.  Pure numpy / OpenCV / Pillow; renders straight
into ffmpeg.

Data (all public domain):
  * Natural Earth vector countries / marine polygons
  * NASA Blue Marble texture (equirectangular)
  * Natural Earth NE1_50M_SR_W shaded relief (optional detail layer)
"""
import json
import math
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30


# --------------------------------------------------------------------- easing
def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_in_out(x):
    x = clamp01(x)
    return 4 * x * x * x if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x, s=1.9):
    x = clamp01(x) - 1
    return x * x * ((s + 1) * x + s) + 1


def fade(t, start, end, fin=0.25, fout=0.25):
    """Opacity envelope of an element living in [start, end]."""
    if t < start or t > end:
        return 0.0
    a = clamp01((t - start) / fin) if fin > 0 else 1.0
    b = clamp01((end - t) / fout) if fout > 0 else 1.0
    return min(a, b)


# ---------------------------------------------------------------- geo data
def _rings(geom):
    if geom is None:
        return []
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [[np.asarray(r, dtype=np.float64)[:, :2] for r in poly] for poly in polys]


def ring_area_centroid(r):
    x, y = r[:, 0], r[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    a = c.sum() / 2
    if abs(a) < 1e-12:
        return 0.0, x.mean(), y.mean()
    return a, ((x + x1) * c).sum() / (6 * a), ((y + y1) * c).sum() / (6 * a)


class Shape:
    """A named set of polygons in lon/lat."""

    def __init__(self, name, polys):
        self.name = name
        self.polys = polys  # list of [outer, hole, hole...]
        main = max(polys, key=lambda p: abs(ring_area_centroid(p[0])[0]))
        _, cx, cy = ring_area_centroid(main[0])
        self.centroid = (cx, cy)
        allpts = np.concatenate([p[0] for p in polys])
        self.bbox = (allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max())

    def moved(self, lon, lat):
        """Copy translated to (lon, lat), keeping its true size (cos-lat corrected)."""
        cx, cy = self.centroid
        k = math.cos(math.radians(cy)) / math.cos(math.radians(lat))
        out = []
        for poly in self.polys:
            out.append([np.column_stack([lon + (r[:, 0] - cx) * k, lat + (r[:, 1] - cy)]) for r in poly])
        s = Shape(self.name, out)
        s.centroid = (lon, lat)
        return s


class Geo:
    KEYS = ("NAME", "NAME_EN", "NAME_LONG", "ADMIN", "ISO_A3", "ADM0_A3", "name", "name_en")

    def __init__(self, *paths):
        self.items = {}
        for p in paths:
            with open(p) as f:
                data = json.load(f)
            for feat in data["features"]:
                props = feat.get("properties") or {}
                rings = _rings(feat.get("geometry"))
                if not rings:
                    continue
                for k in self.KEYS:
                    v = props.get(k)
                    if v and str(v).lower() not in self.items:
                        self.items[str(v).lower()] = (props, rings)

    def get(self, name):
        props, rings = self.items[name.lower()]
        return Shape(name, rings)


# ------------------------------------------------------------------ texture
def build_texture(earth_path, relief_path=None, width=8192, out_path=None):
    """Upscaled Blue Marble with shaded-relief detail and a light grade."""
    if out_path and os.path.exists(out_path):
        return cv2.imread(out_path)
    e = cv2.imread(earth_path)
    e = cv2.resize(e, (width, width // 2), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    if relief_path:
        r = cv2.imread(relief_path, cv2.IMREAD_GRAYSCALE)
        r = cv2.resize(r, (width, width // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
        detail = r - cv2.GaussianBlur(r, (0, 0), 6)
        e *= 1 + 0.010 * detail[..., None]
    # grade: a little more saturation and contrast, teal-ish water
    hsv = cv2.cvtColor(np.clip(e, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1.12
    hsv[..., 2] = np.clip((hsv[..., 2] - 128) * 1.08 + 136, 0, 255)
    e = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if out_path:
        cv2.imwrite(out_path, e, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return e


def fetch_gibs(lon0, lat0, lon1, lat1, px_per_deg=150, out_path=None,
               layer="BlueMarble_NextGeneration", tile_deg=8.0):
    """High-res public-domain NASA GIBS imagery for a lon/lat box (stitched from WMS tiles)."""
    if out_path and os.path.exists(out_path):
        return cv2.imread(out_path)
    import urllib.request
    cols = max(1, math.ceil((lon1 - lon0) / tile_deg))
    rows = max(1, math.ceil((lat1 - lat0) / tile_deg))
    dw, dh = (lon1 - lon0) / cols, (lat1 - lat0) / rows
    tw, th = int(round(dw * px_per_deg)), int(round(dh * px_per_deg))
    out = np.zeros((th * rows, tw * cols, 3), np.uint8)
    for r in range(rows):
        for c in range(cols):
            a0, a1 = lat1 - (r + 1) * dh, lat1 - r * dh
            o0, o1 = lon0 + c * dw, lon0 + (c + 1) * dw
            url = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap"
                   f"&VERSION=1.3.0&LAYERS={layer}&STYLES=&CRS=EPSG:4326&BBOX={a0},{o0},{a1},{o1}"
                   f"&WIDTH={tw}&HEIGHT={th}&FORMAT=image/jpeg")
            data = urllib.request.urlopen(url, timeout=120).read()
            tile = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cv2.resize(tile, (tw, th))
    if out_path:
        cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


class Texture:
    def __init__(self, img):
        self.levels = [img]
        while self.levels[-1].shape[1] > 1024:
            self.levels.append(cv2.pyrDown(self.levels[-1]))
        self.patches = []

    def add_patch(self, img, lon0, lat0, lon1, lat1, feather=0.06):
        """Overlay sharper imagery for a region; colours are matched to the base map."""
        img = img.astype(np.float32)
        base = self.levels[0]
        th, tw = base.shape[:2]
        x0, x1 = int((lon0 + 180) / 360 * tw), int((lon1 + 180) / 360 * tw)
        y0, y1 = int((90 - lat1) / 180 * th), int((90 - lat0) / 180 * th)
        ref = cv2.resize(base[y0:y1, x0:x1], (img.shape[1], img.shape[0])).astype(np.float32)
        small = cv2.resize(img, (ref.shape[1], ref.shape[0]))
        for ch in range(3):  # per-channel mean/std transfer
            m_s, s_s = small[..., ch].mean(), small[..., ch].std() + 1e-6
            m_r, s_r = ref[..., ch].mean(), ref[..., ch].std() + 1e-6
            img[..., ch] = (img[..., ch] - m_s) * (s_r / s_s) + m_r
        # keep the base map's low frequencies so seams vanish, take detail from the patch
        low_ref = cv2.GaussianBlur(ref, (0, 0), 12)
        low_img = cv2.GaussianBlur(img, (0, 0), 12)
        img = np.clip(img - low_img + low_ref, 0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        fy, fx = max(1, int(h * feather)), max(1, int(w * feather))
        ay = np.minimum(np.arange(h) / fy, (h - 1 - np.arange(h)) / fy).clip(0, 1)
        ax = np.minimum(np.arange(w) / fx, (w - 1 - np.arange(w)) / fx).clip(0, 1)
        alpha = (np.outer(ay, ax) * 255).astype(np.uint8)
        levels = [(img, alpha)]
        while levels[-1][0].shape[1] > 512:
            levels.append((cv2.pyrDown(levels[-1][0]), cv2.pyrDown(levels[-1][1])))
        self.patches.append(((lon0, lat0, lon1, lat1), levels))

    def _warp(self, tex, box, lon0, lat0, dpp, k, border):
        plon0, plat0, plon1, plat1 = box
        th, tw = tex.shape[:2]
        ax = dpp / k * tw / (plon1 - plon0)
        ay = dpp * th / (plat1 - plat0)
        tx0 = (lon0 - plon0) / (plon1 - plon0) * tw
        ty0 = (plat1 - lat0) / (plat1 - plat0) * th
        M = np.float32([[ax, 0, tx0 - ax * W / 2], [0, ay, ty0 - ay * H / 2]])
        return cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=border), ax

    def render(self, lon0, lat0, span):
        """Frame showing `span` degrees of latitude vertically, centred on lon0/lat0."""
        dpp = span / H
        k = math.cos(math.radians(lat0))
        tw = self.levels[0].shape[1]
        a = dpp / k * tw / 360.0  # texture px per frame px (horizontal)
        lvl = 0
        while lvl + 1 < len(self.levels) and a / (2 ** (lvl + 1)) >= 0.75:
            lvl += 1
        frame, _ = self._warp(self.levels[lvl], (-180, -90, 180, 90), lon0, lat0, dpp, k, cv2.BORDER_WRAP)
        if a < 0.6:  # zoomed in past the base resolution: use detail patches
            half_w, half_h = W / 2 * dpp / k, H / 2 * dpp
            for (box, levels) in self.patches:
                if (box[0] > lon0 + half_w or box[2] < lon0 - half_w or
                        box[1] > lat0 + half_h or box[3] < lat0 - half_h):
                    continue
                pa = dpp / k * levels[0][0].shape[1] / (box[2] - box[0])
                pl = 0
                while pl + 1 < len(levels) and pa / (2 ** (pl + 1)) >= 0.75:
                    pl += 1
                img, alpha = levels[pl]
                pimg, _ = self._warp(img, box, lon0, lat0, dpp, k, cv2.BORDER_CONSTANT)
                palpha, _ = self._warp(alpha, box, lon0, lat0, dpp, k, cv2.BORDER_CONSTANT)
                fade_in = clamp01((0.6 - a) / 0.3)  # blend in gradually while zooming
                al = palpha.astype(np.float32)[..., None] / 255.0 * fade_in
                frame = np.clip(frame * (1 - al) + pimg * al, 0, 255).astype(np.uint8)
        return frame


# ------------------------------------------------------------------- camera
class View:
    def __init__(self, lon0, lat0, span):
        self.lon0, self.lat0, self.span = lon0, lat0, span
        self.dpp = span / H
        self.k = math.cos(math.radians(lat0))

    def xy(self, lon, lat):
        return (W / 2 + (lon - self.lon0) * self.k / self.dpp, H / 2 - (lat - self.lat0) / self.dpp)

    def pts(self, ring):
        x = W / 2 + (ring[:, 0] - self.lon0) * self.k / self.dpp
        y = H / 2 - (ring[:, 1] - self.lat0) / self.dpp
        return np.column_stack([x, y])

    def px_per_deg(self):
        return 1.0 / self.dpp


class Camera:
    """Keyframes: (time, lon, lat, span). Eased moves, log-space zoom."""

    def __init__(self, keys):
        self.keys = sorted(keys)

    def at(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return View(*ks[0][1:])
        for (t0, lo0, la0, s0), (t1, lo1, la1, s1) in zip(ks, ks[1:]):
            if t0 <= t <= t1:
                u = ease_in_out((t - t0) / max(t1 - t0, 1e-6))
                return View(lo0 + (lo1 - lo0) * u, la0 + (la1 - la0) * u,
                            math.exp(math.log(s0) + (math.log(s1) - math.log(s0)) * u))
        return View(*ks[-1][1:])


# ---------------------------------------------------------------- drawing
def blend(frame, color, alpha):
    """frame: uint8 HxWx3, color: (b,g,r), alpha: float HxW in 0..1."""
    a = alpha[..., None]
    out = frame.astype(np.float32) * (1 - a) + np.float32(color) * a
    frame[:] = np.clip(out, 0, 255).astype(np.uint8)


def _poly_int(pts):
    return np.round(pts * 16).astype(np.int32)  # shift=4 sub-pixel


def shape_mask(view, shape, scale=1.0, about=None):
    mask = np.zeros((H, W), np.uint8)
    outers, holes = [], []
    for poly in shape.polys:
        for i, r in enumerate(poly):
            p = view.pts(r)
            if scale != 1.0 and about is not None:
                p = (p - about) * scale + about
            (outers if i == 0 else holes).append(_poly_int(p))
    if outers:
        cv2.fillPoly(mask, outers, 255, lineType=cv2.LINE_AA, shift=4)
    if holes:
        cv2.fillPoly(mask, holes, 0, lineType=cv2.LINE_AA, shift=4)
    return mask, outers


def soft_glow(mask, sigma=14):
    small = cv2.resize(mask, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


_FONT_CACHE = {}


def font(size, path=None):
    path = path or os.environ.get("GEOANIM_FONT", "Montserrat-ExtraBold.ttf")
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def paste_rgba(frame, rgba, x, y, opacity=1.0):
    """Alpha-composite an RGBA numpy patch with its top-left at (x, y)."""
    h, w = rgba.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + w, W), min(y0 + h, H)
    if x1 >= x2 or y1 >= y2:
        return
    patch = rgba[y1 - y0:y2 - y0, x1 - x0:x2 - x0].astype(np.float32)
    a = patch[..., 3:4] / 255.0 * opacity
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    rgb = patch[..., :3][..., ::-1]  # RGBA -> BGR
    frame[y1:y2, x1:x2] = np.clip(roi * (1 - a) + rgb * a, 0, 255).astype(np.uint8)


def text_patch(text, size, fill=(255, 255, 255), stroke=(0, 0, 0), stroke_w=0,
               bg=None, pad=(26, 14), radius=18, shadow=True):
    f = font(size)
    l, t, r, b = f.getbbox(text, stroke_width=stroke_w)
    tw, th = r - l, b - t
    pw, ph = tw + 2 * pad[0], th + 2 * pad[1]
    sh = 8 if shadow else 0
    img = Image.new("RGBA", (pw + sh, ph + sh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg is not None:
        if shadow:
            d.rounded_rectangle([sh, sh, pw + sh - 1, ph + sh - 1], radius, fill=(0, 0, 0, 90))
        d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius, fill=bg)
    elif shadow:
        d.text((pad[0] - l + 3, pad[1] - t + 4), text, font=f, fill=(0, 0, 0, 150),
               stroke_width=stroke_w, stroke_fill=(0, 0, 0, 150))
    d.text((pad[0] - l, pad[1] - t), text, font=f, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)
    return np.array(img)


# ---------------------------------------------------------------- elements
class Element:
    def __init__(self, start, end):
        self.start, self.end = start, end

    def active(self, t):
        return self.start <= t <= self.end

    def draw(self, frame, view, t):
        raise NotImplementedError


class Highlight(Element):
    """Country / sea fill with glow and outline.  style: 'fill' | 'glow'."""

    def __init__(self, shape, start, end, color=(40, 90, 230), style="fill", fill_alpha=0.78,
                 outline=(30, 30, 30), glow=True, move_to=None, move_t=None, dim_after=None):
        super().__init__(start, end)
        self.shape, self.color, self.style = shape, color, style
        self.fill_alpha, self.outline, self.glow = fill_alpha, outline, glow
        self.move_to, self.move_t, self.dim_after = move_to, move_t, dim_after

    def current_shape(self, t):
        if not self.move_to:
            return self.shape
        t0, t1 = self.move_t
        u = ease_in_out((t - t0) / max(t1 - t0, 1e-6))
        cx, cy = self.shape.centroid
        lon = cx + (self.move_to[0] - cx) * u
        lat = cy + (self.move_to[1] - cy) * u
        return self.shape.moved(lon, lat)

    def draw(self, frame, view, t):
        op = fade(t, self.start, self.end, 0.3, 0.3)
        if self.dim_after is not None and t > self.dim_after:
            op *= 1 - 0.55 * smooth((t - self.dim_after) / 0.4)
        if op <= 0:
            return
        shp = self.current_shape(t)
        pop = ease_out_back((t - self.start) / 0.45, 1.4)
        scale = 0.9 + 0.1 * pop
        c = np.array(view.xy(*shp.centroid), np.float32)
        mask, outers = shape_mask(view, shp, scale, c)
        m = mask.astype(np.float32) / 255.0
        if self.glow:
            g = soft_glow(mask, 26 if self.style == "glow" else 16)
            gcol = tuple(min(255, int(v * 1.15 + 40)) for v in self.color)
            blend(frame, gcol, np.clip(g * (1.4 if self.style == "glow" else 0.7), 0, 1) * op * (1 - m * 0.5))
        if self.style == "fill":
            blend(frame, self.color, m * self.fill_alpha * op)
            if self.outline is not None:
                lay = np.zeros((H, W), np.uint8)
                cv2.polylines(lay, outers, True, 255, 3, cv2.LINE_AA, shift=4)
                blend(frame, self.outline, lay.astype(np.float32) / 255.0 * 0.85 * op)
        else:
            lay = np.zeros((H, W), np.uint8)
            cv2.polylines(lay, outers, True, 255, 6, cv2.LINE_AA, shift=4)
            blend(frame, self.color, m * 0.18 * op)
            blend(frame, (255, 255, 255), lay.astype(np.float32) / 255.0 * op)
            lay2 = cv2.GaussianBlur(lay, (0, 0), 5).astype(np.float32) / 255.0
            blend(frame, self.color, np.clip(lay2 * 2.2, 0, 1) * 0.9 * op)


class Eyes(Element):
    """Cartoon eyes on a shape (or any lon/lat). mood: happy|angry|worried|smug|shock."""

    def __init__(self, target, start, end, size=150, mood="happy", look=None, blink_every=2.6,
                 offset=(0.0, 0.0), wink_at=None):
        super().__init__(start, end)
        self.target, self.size, self.mood, self.look = target, size, mood, look
        self.blink_every, self.offset, self.wink_at = blink_every, offset, wink_at

    def _pos(self, t):
        tg = self.target
        if isinstance(tg, Highlight):
            lon, lat = tg.current_shape(t).centroid
        elif isinstance(tg, Shape):
            lon, lat = tg.centroid
        else:
            lon, lat = tg
        return lon + self.offset[0], lat + self.offset[1]

    def draw(self, frame, view, t):
        op = fade(t, self.start, self.end, 0.15, 0.2)
        if op <= 0:
            return
        s = self.size * (0.2 + 0.8 * ease_out_back((t - self.start) / 0.35, 2.2))
        cx, cy = view.xy(*self._pos(t))
        lt = t - self.start
        ph = (lt % self.blink_every) / self.blink_every
        open_ = 1.0 if lt < 0.6 or ph < 0.95 else abs(ph - 0.975) / 0.025
        look = (0.0, 0.0)
        if self.look is not None:  # lon/lat to look at
            lx, ly = view.xy(*self.look)
            dx, dy = lx - cx, ly - cy
            n = math.hypot(dx, dy) or 1
            look = (dx / n, dy / n)
        ew, eh = s * 0.23, s * 0.30
        for side in (-1, 1):
            ex, ey = cx + side * s * 0.25, cy
            o = open_
            if self.wink_at is not None and side == 1 and self.wink_at <= t <= self.wink_at + 0.5:
                o = 0.08
            h = max(2.0, eh * o)
            cv2.ellipse(frame, (int(ex), int(ey)), (int(ew + 5), int(h + 5)), 0, 0, 360, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.ellipse(frame, (int(ex), int(ey)), (int(ew), int(h)), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
            if o > 0.3:
                px = ex + look[0] * ew * 0.45
                py = ey + look[1] * h * 0.45 + (h * 0.15 if self.mood == "smug" else 0)
                pr = ew * (0.30 if self.mood == "shock" else 0.48)
                cv2.circle(frame, (int(px), int(py)), max(2, int(pr)), (15, 15, 15), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(px - pr * 0.35), int(py - pr * 0.35)), max(1, int(pr * 0.28)),
                           (255, 255, 255), -1, cv2.LINE_AA)
            if self.mood == "smug":  # heavy lids
                cv2.ellipse(frame, (int(ex), int(ey - h * 0.35)), (int(ew + 5), int(h * 0.7)), 0, 180, 360,
                            (20, 20, 20), -1, cv2.LINE_AA)
            bt = max(3, int(s * 0.045))
            by = ey - eh - s * 0.08
            if self.mood == "angry":
                p1, p2 = (ex - side * ew * 1.1, by - s * 0.06), (ex + side * ew * 0.6, by + s * 0.05)
            elif self.mood in ("worried", "shock"):
                lift = s * (0.10 if self.mood == "shock" else 0.0)
                p1, p2 = (ex - side * ew * 1.0, by + s * 0.03 - lift), (ex + side * ew * 0.8, by - s * 0.07 - lift)
            else:
                p1, p2 = (ex - ew, by), (ex + ew, by - s * 0.02 * side)
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (20, 20, 20), bt, cv2.LINE_AA)
        if op < 1:
            pass  # eyes are opaque cartoon overlays; fade handled by pop scale


class Label(Element):
    """Rounded pill label. NB: `bg` / `fill` are RGB(A), unlike map layers which use BGR."""

    def __init__(self, text, at, start, end, size=50, bg=(255, 255, 255, 235), fill=(25, 25, 25),
                 offset=(0, -110)):
        super().__init__(start, end)
        self.at, self.offset = at, offset
        self.patch = text_patch(text, size, fill=fill, bg=bg)

    def draw(self, frame, view, t):
        op = fade(t, self.start, self.end, 0.2, 0.25)
        if op <= 0:
            return
        at = self.at
        if isinstance(at, Highlight):
            at = at.current_shape(t).centroid
        elif isinstance(at, Shape):
            at = at.centroid
        x, y = view.xy(*at)
        x += self.offset[0]
        y += self.offset[1]
        s = 0.6 + 0.4 * ease_out_back((t - self.start) / 0.35, 2.0)
        p = self.patch
        if abs(s - 1) > 0.01:
            p = cv2.resize(p, (max(1, int(p.shape[1] * s)), max(1, int(p.shape[0] * s))), interpolation=cv2.INTER_LINEAR)
        paste_rgba(frame, p, x - p.shape[1] / 2, y - p.shape[0] / 2, op)


class Route(Element):
    """Dashed line drawn progressively along lon/lat points."""

    def __init__(self, pts, start, end, draw_time=1.2, color=(255, 255, 255), width=8, dash=28):
        super().__init__(start, end)
        self.pts = np.asarray(pts, np.float64)
        self.draw_time, self.color, self.width, self.dash = draw_time, color, width, dash

    def draw(self, frame, view, t):
        op = fade(t, self.start, self.end, 0.05, 0.3)
        if op <= 0:
            return
        p = view.pts(self.pts)
        seg = np.hypot(*np.diff(p, axis=0).T)
        total = seg.sum()
        upto = total * smooth((t - self.start) / self.draw_time)
        lay = np.zeros((H, W), np.uint8)
        d = 0.0
        for (a, b), L in zip(zip(p[:-1], p[1:]), seg):
            if L <= 0:
                continue
            s0 = d
            while s0 < d + L and s0 < upto:
                s1 = min(s0 + self.dash, d + L, upto)
                if int((s0 // self.dash)) % 2 == 0:
                    pa = a + (b - a) * (s0 - d) / L
                    pb = a + (b - a) * (s1 - d) / L
                    cv2.line(lay, tuple(_poly_int(pa)), tuple(_poly_int(pb)), 255, self.width, cv2.LINE_AA, shift=4)
                s0 = s1 if s1 > s0 else s0 + self.dash
            d += L
        shadow = cv2.GaussianBlur(lay, (0, 0), 4).astype(np.float32) / 255.0
        blend(frame, (0, 0, 0), shadow * 0.5 * op)
        blend(frame, self.color, lay.astype(np.float32) / 255.0 * op)


class XMark(Element):
    def __init__(self, at, start, end, size=110, color=(40, 40, 230)):
        super().__init__(start, end)
        self.at, self.size, self.color = at, size, color

    def draw(self, frame, view, t):
        op = fade(t, self.start, self.end, 0.05, 0.25)
        if op <= 0:
            return
        s = self.size * ease_out_back((t - self.start) / 0.3, 2.5) / 2
        x, y = view.xy(*self.at)
        lay = np.zeros((H, W), np.uint8)
        th = max(6, int(self.size * 0.17))
        for d in (1, -1):
            cv2.line(lay, (int(x - s), int(y - s * d)), (int(x + s), int(y + s * d)), 255, th, cv2.LINE_AA)
        out = cv2.dilate(lay, np.ones((9, 9), np.uint8))
        blend(frame, (255, 255, 255), out.astype(np.float32) / 255.0 * op)
        blend(frame, self.color, lay.astype(np.float32) / 255.0 * op)


class Sprite(Element):
    """RGBA image (e.g. pointing hand) whose `anchor` (fx, fy fraction) sits on a lon/lat."""

    def __init__(self, path, at, start, end, height=320, anchor=(0.5, 0.0), bob=14, rotate=0.0):
        super().__init__(start, end)
        img = Image.open(path).convert("RGBA")
        if rotate:
            img = img.rotate(rotate, expand=True, resample=Image.BICUBIC)
        k = height / img.height
        img = img.resize((max(1, int(img.width * k)), height), Image.LANCZOS)
        self.img = np.array(img)
        self.at, self.anchor, self.bob = at, anchor, bob

    def draw(self, frame, view, t):
        op = fade(t, self.start, self.end, 0.2, 0.25)
        if op <= 0:
            return
        at = self.at.centroid if isinstance(self.at, Shape) else self.at
        x, y = view.xy(*at)
        rise = (1 - ease_out_back((t - self.start) / 0.45, 1.6)) * 220
        y += rise + math.sin((t - self.start) * 5.0) * self.bob
        h, w = self.img.shape[:2]
        paste_rgba(frame, self.img, x - self.anchor[0] * w, y - self.anchor[1] * h, op)


class Captions(Element):
    """Word-timed captions: list of (start, end, text) chunks."""

    def __init__(self, chunks, y=0.70, size=58):
        super().__init__(0, 1e9)
        self.chunks, self.y, self.size = chunks, y, size
        self.cache = {}

    def draw(self, frame, view, t):
        for s, e, text in self.chunks:
            if s <= t <= e:
                if text not in self.cache:
                    self.cache[text] = text_patch(text, self.size, stroke=(0, 0, 0), stroke_w=5, shadow=True)
                p = self.cache[text]
                paste_rgba(frame, p, W / 2 - p.shape[1] / 2, H * self.y - p.shape[0] / 2)
                return


class Vignette(Element):
    def __init__(self, strength=0.35):
        super().__init__(0, 1e9)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / math.sqrt(2)
        self.a = np.clip((r - 0.45) / 0.55, 0, 1) ** 1.6 * strength

    def draw(self, frame, view, t):
        blend(frame, (0, 0, 0), self.a)


# ------------------------------------------------------------- voice track
def _wav_dur(p):
    import wave
    with wave.open(p) as w:
        return w.getnframes() / w.getframerate()


def build_voice_track(n_lines, voice_dir, work, lead=0.4, gap=0.35, tail=1.0):
    """line1.wav..lineN.wav -> one normalised track. Returns (path, [(start, end)], total)."""
    os.makedirs(work, exist_ok=True)
    spans, parts, t = [], [], lead
    for i in range(1, n_lines + 1):
        src = os.path.join(voice_dir, f"line{i}.wav")
        norm = os.path.join(work, f"n{i}.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "48000",
                        "-af", "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                               "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                               "loudnorm=I=-16:TP=-1.5:LRA=11", norm], check=True)
        d = _wav_dur(norm)
        spans.append((t, t + d))
        parts.append((t, norm))
        t += d + gap
    total = spans[-1][1] + tail
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


def caption_chunks(audio, spans, lines, language="en", max_words=3):
    """Whisper word timings, authored words from `lines`. Returns [(start, end, text)]."""
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=False)
    words = [w for s in segs for w in s.words]
    chunks = []
    for (s, e), line in zip(spans, lines):
        authored = line.split()
        ws = [w for w in words if s - 0.25 <= (w.start + w.end) / 2 <= e + 0.25]
        if len(ws) == len(authored):
            times = [(w.start, w.end) for w in ws]
        else:  # even spacing inside the line when word counts disagree
            step = (e - s) / len(authored)
            times = [(s + i * step, s + (i + 1) * step) for i in range(len(authored))]
        i = 0
        while i < len(authored):
            j = min(i + max_words, len(authored))
            for k in range(i, j):
                if authored[k][-1] in ",.?!" and k + 1 < j:
                    j = k + 1
                    break
            chunks.append((times[i][0], times[j - 1][1] + 0.05, " ".join(authored[i:j])))
            i = j
    out = []
    for n, (a, b, txt) in enumerate(chunks):
        nxt = chunks[n + 1][0] if n + 1 < len(chunks) else b + 0.4
        out.append((a, nxt if nxt - b < 0.5 else b + 0.15, txt))
    return out


# ----------------------------------------------------------------- render
def frame_at(texture, camera, elements, t):
    view = camera.at(t)
    frame = texture.render(view.lon0, view.lat0, view.span)
    for el in elements:
        if el.active(t):
            el.draw(frame, view, t)
    return frame


def render_stills(times, out_dir, texture, camera, elements):
    """Quick look: save frames at the given times as JPEGs."""
    os.makedirs(out_dir, exist_ok=True)
    for t in times:
        p = os.path.join(out_dir, f"still_{t:06.2f}.jpg")
        cv2.imwrite(p, frame_at(texture, camera, elements, t), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("still", p, flush=True)


def render(out_path, texture, camera, elements, duration, audio_path=None, fps=FPS, crf=18):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio_path:
        cmd += ["-i", audio_path, "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-t", f"{duration:.3f}", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(round(duration * fps))
    for i in range(n):
        t = i / fps
        view = camera.at(t)
        frame = texture.render(view.lon0, view.lat0, view.span)
        for el in elements:
            if el.active(t):
                el.draw(frame, view, t)
        proc.stdin.write(frame.tobytes())
        if i % (fps * 5) == 0:
            print(f"  frame {i}/{n}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
