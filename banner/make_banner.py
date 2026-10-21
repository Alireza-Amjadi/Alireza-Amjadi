"""
Animated profile banner for github.com/Alireza-Amjadi.

A sagittal-plane leg walks through one gait cycle with an active ankle-foot
orthosis (AFO). Hip, knee and ankle angles are periodic shape-preserving interpolations to
typical normative gait curves (Winter, Biomechanics and Motor Control of Human
Movement). The plot on the right traces the ankle angle and a normalized AFO
assist torque: plantarflexion assist at push-off and a small dorsiflexion
assist in swing for foot clearance. The torque profile is illustrative.

Run:  python make_banner.py        -> banner-light.svg, banner-dark.svg
Needs: numpy, scipy
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

N = 72            # frames per gait cycle
DUR = 2.4         # seconds per cycle (slowed down for readability)

# ------------------------------------------------------------------ gait data
def periodic(pts):
    """Shape-preserving periodic interpolation (no overshoot between key points)."""
    x, y = np.array(pts, float).T
    x, y = x[:-1] / 100, y[:-1]
    xx = np.r_[x - 1, x, x + 1, 2]
    yy = np.r_[y, y, y, y[0]]
    f = PchipInterpolator(xx, yy)
    return lambda s: f(np.asarray(s) % 1)

HIP = periodic([(0, 30), (10, 27), (30, 10), (50, -10), (60, -6), (70, 14), (85, 32), (100, 30)])
KNEE = periodic([(0, 3), (15, 18), (40, 4), (60, 38), (73, 62), (87, 28), (100, 3)])
ANKLE = periodic([(0, 0), (7, -6), (30, 5), (48, 11), (62, -18), (75, -4), (90, 2), (100, 0)])


def assist(s):
    """Normalized AFO torque (+ plantarflexion assist, - dorsiflexion assist)."""
    s = np.asarray(s) % 1
    pf = np.where((s > 0.36) & (s < 0.62), np.sin(np.pi * (s - 0.36) / 0.26) ** 2, 0)
    df = np.where((s > 0.66) & (s < 0.90), -0.22 * np.sin(np.pi * (s - 0.66) / 0.24) ** 2, 0)
    return pf + df

# ------------------------------------------------------------------ kinematics (pixels, y down)
LT, LS = 118, 112          # thigh, shank
HEEL, TOE, AH = 20, 54, 16 # foot: heel behind ankle, toe ahead, ankle height


def rot(v, a):  # rotate vector v by angle a (rad), screen coords, positive = counter-clockwise on screen
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


def leg(s):
    h, k, a = np.radians([HIP(s), KNEE(s), ANKLE(s)])
    th = h                     # thigh angle from vertical, forward positive
    sh = h - k                 # shank angle
    hip = np.zeros(2)
    knee = hip + rot([0, LT], th)
    ank = knee + rot([0, LS], sh)
    fa = sh + a                # foot orientation (dorsiflexion lifts the toe)
    heel = ank + rot([-HEEL, AH], fa)
    toe = ank + rot([TOE, AH], fa)
    return dict(hip=hip, knee=knee, ank=ank, heel=heel, toe=toe, sh=sh, fa=fa)


S = np.linspace(0, 1, N, endpoint=False)
L1 = [leg(s) for s in S]
L2 = [leg((s + 0.5) % 1) for s in S]
# vertical offset: the lowest point of either foot touches the ground
low = np.array([max(d[p][1] for d in (a, b) for p in ("heel", "toe")) for a, b in zip(L1, L2)])
low = np.convolve(np.r_[low[-2:], low, low[:2]], np.ones(5) / 5, "valid")   # light smoothing
GROUND = 0.0

# ------------------------------------------------------------------ svg helpers
PAL = {
    "light": dict(bg="#f7f9fb", ink="#17212b", ink2="#55616e", faint="#c9d1d8", grid="#e7ecf0",
                  acc="#d9622b", acc2="#1f5fa8", skin="#ffffff", leg2="#dfe5ea"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9aa7b4", faint="#2d3a48", grid="#161e28",
                 acc="#ff8a4c", acc2="#79b8ff", skin="#161b22", leg2="#1f2933"),
}
F = lambda v: f"{v:.1f}"


def anim(attr, vals):
    return (f'<animate attributeName="{attr}" dur="{DUR}s" repeatCount="indefinite" calcMode="linear" '
            f'values="{";".join(vals + vals[:1])}"/>')


def limb(p, q, w, frames_p, frames_q, fill, stroke, sw=1.4, extra="", w2=None):
    """Tapered segment between two moving points, as an animated polygon."""
    w2 = w if w2 is None else w2
    def poly(a, b):
        d = b - a
        n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
        pts = [a + n * w / 2, b + n * w2 / 2, b - n * w2 / 2, a - n * w / 2]
        return " ".join(f"{F(x)},{F(y)}" for x, y in pts)
    vals = [poly(a, b) for a, b in zip(frames_p, frames_q)]
    return (f'<polygon points="{vals[0]}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" {extra}>'
            f'{anim("points", vals)}</polygon>')


def joint(frames, r, fill, stroke, sw=1.4):
    xs = [F(p[0]) for p in frames]
    ys = [F(p[1]) for p in frames]
    return (f'<circle cx="{xs[0]}" cy="{ys[0]}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}">'
            f'{anim("cx", xs)}{anim("cy", ys)}</circle>')


def foot(frames_d, fill, stroke, sw=1.4):
    def poly(d):
        a, fa = d["ank"], d["fa"]
        pts = [a + rot(v, fa) for v in ([-9, -9], [-HEEL + 1, -1], [-HEEL, AH - 5], [-HEEL + 5, AH], [TOE - 4, AH],
                                         [TOE, AH - 3], [TOE - 10, AH - 9], [14, -1], [7, -9])]
        return " ".join(f"{F(x)},{F(y)}" for x, y in pts)
    vals = [poly(d) for d in frames_d]
    return (f'<polygon points="{vals[0]}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round">'
            f'{anim("points", vals)}</polygon>')


def leg_svg(frames, ox, oy, c, ghost=False):
    T = lambda key: [np.array([ox, oy - lo]) + d[key] for d, lo in zip(frames, low)]
    sh = [dict(ank=np.array([ox, oy - lo]) + d["ank"], fa=d["fa"]) for d, lo in zip(frames, low)]
    hip, knee, ank = T("hip"), T("knee"), T("ank")
    if ghost:
        col, st = c["leg2"], c["faint"]
        return (limb(None, None, 32, hip, knee, col, st, w2=22) + limb(None, None, 22, knee, ank, col, st, w2=15) +
                foot(sh, col, st))
    out = [limb(None, None, 32, hip, knee, c["skin"], c["ink"], w2=22),
           limb(None, None, 22, knee, ank, c["skin"], c["ink"], w2=15),
           foot(sh, c["skin"], c["ink"])]
    # AFO: shank cuff, foot plate, joint at the ankle and a linear actuator behind the shank
    cuff_a = [k + (a - k) * 0.30 for k, a in zip(knee, ank)]
    cuff_b = [k + (a - k) * 0.55 for k, a in zip(knee, ank)]
    out.append(limb(None, None, 27, cuff_a, cuff_b, c["acc2"], c["acc2"], 1, 'fill-opacity=".22"', w2=23))
    plate_a = [d["ank"] + rot([-HEEL + 2, AH + 3], d["fa"]) for d in sh]
    plate_b = [d["ank"] + rot([TOE - 14, AH + 3], d["fa"]) for d in sh]
    out.append(limb(None, None, 4, plate_a, plate_b, c["acc2"], c["acc2"], 1))
    strut_a = [k + (a - k) * 0.42 for k, a in zip(knee, ank)]
    out.append(limb(None, None, 4, strut_a, ank, c["acc2"], c["acc2"], 1))
    # actuator: from the back of the cuff to the heel lever
    act_top = [p + rot([-21, 0], d["sh"]) for p, d in zip(cuff_a, frames)]
    act_bot = [d["ank"] + rot([-HEEL - 2, AH - 10], d["fa"]) for d in sh]
    out.append(limb(None, None, 7, act_top, act_bot, c["acc"], c["acc"], 1))
    out.append(joint(act_top, 3, c["bg"], c["acc"], 1.6))
    out.append(joint(act_bot, 3, c["bg"], c["acc"], 1.6))
    out.append(joint(ank, 5.5, c["bg"], c["acc2"], 2))
    out.append(joint(knee, 6, c["skin"], c["ink"]))
    return "".join(out)


def plot(c, x0, y0, w, h, f, ylo, yhi, label, color, unit):
    xs = np.linspace(0, 1, 201)
    X = lambda s: x0 + s * w
    Y = lambda v: y0 + h - (v - ylo) / (yhi - ylo) * h
    o = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="{c["faint"]}" stroke-width="1"/>']
    o.append(f'<rect x="{F(X(0))}" y="{y0}" width="{F(0.6 * w)}" height="{h}" fill="{c["grid"]}"/>')
    if ylo < 0 < yhi:
        o.append(f'<line x1="{x0}" y1="{F(Y(0))}" x2="{x0 + w}" y2="{F(Y(0))}" stroke="{c["faint"]}" stroke-width="1" stroke-dasharray="3 3"/>')
    d = "M" + " L".join(f"{F(X(s))} {F(Y(v))}" for s, v in zip(xs, f(xs)))
    o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    cx = [F(X(s)) for s in S]
    cy = [F(Y(v)) for v in f(S)]
    o.append(f'<line x1="{cx[0]}" y1="{y0}" x2="{cx[0]}" y2="{y0 + h}" stroke="{c["ink2"]}" stroke-width="1">'
             f'{anim("x1", cx)}{anim("x2", cx)}</line>')
    o.append(f'<circle cx="{cx[0]}" cy="{cy[0]}" r="4" fill="{color}">{anim("cx", cx)}{anim("cy", cy)}</circle>')
    M = 'font-family="IBM Plex Mono, SFMono-Regular, Menlo, Consolas, monospace"'
    o.append(f'<text x="{x0}" y="{y0 - 8}" {M} font-size="11.5" fill="{c["ink"]}">{label}</text>')
    o.append(f'<text x="{x0 + w}" y="{y0 - 8}" {M} font-size="10.5" fill="{c["ink2"]}" text-anchor="end">{unit}</text>')
    return "".join(o)


def svg(theme):
    c = PAL[theme]
    W, H = 1280, 360
    M = 'font-family="IBM Plex Mono, SFMono-Regular, Menlo, Consolas, monospace"'
    T = 'font-family="Barlow Condensed, Arial Narrow, Roboto Condensed, Arial, sans-serif"'
    S_ = 'font-family="IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif"'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         'aria-label="Alireza Amjadi, Mechanical Engineering, Sharif University of Technology. Animated leg walking with an active ankle-foot orthosis, with the ankle angle and assist torque traced over the gait cycle.">',
         f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>']
    for gx in range(0, W, 24):
        o.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{H}" stroke="{c["grid"]}" stroke-width="1"/>')
    for gy in range(0, H, 24):
        o.append(f'<line x1="0" y1="{gy}" x2="{W}" y2="{gy}" stroke="{c["grid"]}" stroke-width="1"/>')
    o.append(f'<rect x="8" y="8" width="{W - 16}" height="{H - 16}" fill="none" stroke="{c["ink"]}" stroke-width="1.5"/>')

    # text block
    o += [f'<text x="48" y="104" {T} font-size="58" font-weight="700" fill="{c["ink"]}" letter-spacing="1.5">ALIREZA AMJADI</text>',
          f'<text x="50" y="140" {M} font-size="13" fill="{c["ink2"]}" letter-spacing="1">MECHANICAL ENGINEERING · SHARIF UNIVERSITY OF TECHNOLOGY</text>',
          f'<text x="50" y="186" {S_} font-size="20" fill="{c["ink"]}">Wearable robotics and biomechatronics</text>',
          f'<text x="50" y="214" {S_} font-size="16" fill="{c["ink2"]}">Active ankle–foot orthoses · exoskeleton modelling in OpenSim</text>',
          f'<text x="50" y="238" {S_} font-size="16" fill="{c["ink2"]}">Machine design · dynamics and control · simulation</text>',
          f'<line x1="50" y1="266" x2="470" y2="266" stroke="{c["faint"]}" stroke-width="1"/>',
          f'<text x="50" y="292" {M} font-size="11.5" fill="{c["acc"]}">Gait cycle with an active AFO: ankle angle and assist torque</text>']

    # walking figure
    ox, oy = 640, 318
    o.append(f'<line x1="{ox - 150}" y1="{oy}" x2="{ox + 150}" y2="{oy}" stroke="{c["ink"]}" stroke-width="1.6"/>')
    o.append(f'<line x1="{ox - 150}" y1="{oy + 6}" x2="{ox + 150}" y2="{oy + 6}" stroke="{c["ink2"]}" stroke-width="1" stroke-dasharray="10 8">'
             f'<animate attributeName="stroke-dashoffset" dur="{DUR / 2}s" repeatCount="indefinite" values="0;18"/></line>')
    o.append(leg_svg(L2, ox, oy, c, ghost=True))
    o.append(leg_svg(L1, ox, oy, c))
    hy = [F(oy - lo - 4) for lo in low]
    o.append(f'<ellipse cx="{ox}" cy="{hy[0]}" rx="27" ry="17" fill="{c["skin"]}" stroke="{c["ink"]}" stroke-width="1.4">{anim("cy", hy)}</ellipse>')
    o.append(joint([np.array([ox, oy - lo]) for lo in low], 6.5, c["skin"], c["ink"]))

    # plots
    px, pw, ph = 860, 360, 92
    o.append(plot(c, px, 62, pw, ph, ANKLE, -22, 16, "Ankle angle", c["acc2"], "deg, + dorsiflexion"))
    o.append(plot(c, px, 196, pw, ph, assist, -0.35, 1.1, "AFO assist torque", c["acc"], "normalized, + plantarflexion"))
    o.append(f'<text x="{px}" y="{196 + ph + 18}" {M} font-size="10.5" fill="{c["ink2"]}">0 %</text>')
    o.append(f'<text x="{px + 0.6 * pw}" y="{196 + ph + 18}" {M} font-size="10.5" fill="{c["ink2"]}" text-anchor="middle">60 %</text>')
    o.append(f'<text x="{px + pw}" y="{196 + ph + 18}" {M} font-size="10.5" fill="{c["ink2"]}" text-anchor="end">100 % gait cycle</text>')
    o.append(f'<text x="{px + 0.3 * pw}" y="{62 + ph - 6}" {M} font-size="10.5" fill="{c["ink2"]}" text-anchor="middle">stance</text>')
    o.append(f'<text x="{px + 0.8 * pw}" y="{62 + ph - 6}" {M} font-size="10.5" fill="{c["ink2"]}" text-anchor="middle">swing</text>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        with open(f"banner-{theme}.svg", "w") as fh:
            fh.write(svg(theme))
    print("wrote banner-light.svg and banner-dark.svg")


def hero(c):
    """Figure and plots for the portfolio site (600 x 330), without the text block."""
    o = []
    ox, oy = 165, 300
    o.append(f'<line x1="{ox - 140}" y1="{oy}" x2="{ox + 140}" y2="{oy}" stroke="{c["ink"]}" stroke-width="1.6"/>')
    o.append(f'<line x1="{ox - 140}" y1="{oy + 6}" x2="{ox + 140}" y2="{oy + 6}" stroke="{c["ink2"]}" stroke-width="1" stroke-dasharray="10 8">'
             f'<animate attributeName="stroke-dashoffset" dur="{DUR / 2}s" repeatCount="indefinite" values="0;18"/></line>')
    o.append(leg_svg(L2, ox, oy, c, ghost=True))
    o.append(leg_svg(L1, ox, oy, c))
    hy = [F(oy - lo - 4) for lo in low]
    o.append(f'<ellipse cx="{ox}" cy="{hy[0]}" rx="27" ry="17" fill="{c["skin"]}" stroke="{c["ink"]}" stroke-width="1.4">{anim("cy", hy)}</ellipse>')
    o.append(joint([np.array([ox, oy - lo]) for lo in low], 6.5, c["skin"], c["ink"]))
    M = 'font-family="IBM Plex Mono, SFMono-Regular, Menlo, Consolas, monospace"'
    px, pw, ph = 335, 245, 82
    o.append(plot(c, px, 50, pw, ph, ANKLE, -22, 16, "Ankle angle", c["acc2"], "deg"))
    o.append(plot(c, px, 178, pw, ph, assist, -0.35, 1.1, "AFO assist torque", c["acc"], "normalized"))
    o.append(f'<text x="{px + 0.3 * pw}" y="{50 + ph - 6}" {M} font-size="10" fill="{c["ink2"]}" text-anchor="middle">stance</text>')
    o.append(f'<text x="{px + 0.8 * pw}" y="{50 + ph - 6}" {M} font-size="10" fill="{c["ink2"]}" text-anchor="middle">swing</text>')
    o.append(f'<text x="{px}" y="{178 + ph + 16}" {M} font-size="10" fill="{c["ink2"]}">0 %</text>')
    o.append(f'<text x="{px + pw}" y="{178 + ph + 16}" {M} font-size="10" fill="{c["ink2"]}" text-anchor="end">100 % gait cycle</text>')
    return "".join(o)
