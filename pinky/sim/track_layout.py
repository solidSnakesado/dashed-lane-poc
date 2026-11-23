"""트랙 도로의 배치와 경로를 무작위로 만든다(그리기는 gen_track.py).

구조: 보드 둘레를 시계 방향으로 도는 외곽 도로(1개 차로) + 안쪽 도로(세로, 가로) + 그 사이의 구역(블록).
안쪽 도로는 가운데 교차로에서 만난다. 갈래(N, E, S, W)마다 교차로로 들어가는 차로를 따로 뽑는다:
  바깥쪽부터 [우회전 전용] 본 차로 [좌회전 전용]. 마주 보는 갈래의 본 차로는 서로의 나가는 차로와 일직선이다.
  전용 차로가 있는 쪽으로 도로가 한 차로씩 밀리므로 어긋난 교차로가 저절로 생긴다.
좌표: mm, x 오른쪽, y 위쪽. 경로 = 들어오는 차로 → 교차로 → 나가는 차로, 종류는 straight·left·right.
우회전은 안쪽 블록의 둥근 모서리와 같은 중심의 원호, 좌회전은 두 차로 중심선에 접하는 원호다.
"""
import numpy as np

ARMS = "NESW"                                              # 시계 방향. i에서 들어와 직진 = i+2, 좌회전 = i+1, 우회전 = i+3
DIN = {"N": (0, -1), "E": (-1, 0), "S": (0, 1), "W": (1, 0)}   # 교차로로 들어가는 방향
UP, DOWN = {"N": ("NW", "tr"), "E": ("NE", "br"), "S": ("SE", "bl"), "W": ("SW", "tl")}, {"N": ("NE", "tl"), "E": ("SE", "tr"), "S": ("SW", "br"), "W": ("NW", "bl")}
INNER = {"N": ("NW", "br"), "E": ("NE", "bl"), "S": ("SE", "tl"), "W": ("SW", "tr")}     # 그 갈래에서 우회전할 때 안쪽에 있는 블록 모서리


def fillet(pts, radii, step=4.0):
    """직각으로 꺾이는 점마다 반지름 radii의 원호로 둥글린 길. (점들, 원호마다 (시작 번호, 끝 번호))"""
    pts, out, arcs = [np.array(p, float) for p in pts], [], []
    out.append(pts[0])
    for k in range(1, len(pts) - 1):
        a, c, b, R = pts[k - 1], pts[k], pts[k + 1], radii[k - 1]
        d1, d2 = (c - a) / np.linalg.norm(c - a), (b - c) / np.linalg.norm(b - c)
        t1, t2 = c - d1 * R, c + d2 * R
        o = t1 + d2 * R
        a0, a1 = np.arctan2(*(t1 - o)[::-1]), np.arctan2(*(t2 - o)[::-1])
        th = a0 + np.linspace(0, (a1 - a0 + np.pi) % (2 * np.pi) - np.pi, max(8, int(1.6 * R / step)))
        arcs.append((len(out), len(out) + len(th) - 1))
        out += [o + R * np.array([np.cos(t), np.sin(t)]) for t in th]
    p = np.array(out + [pts[-1]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    ss = np.arange(0, s[-1], step)
    q = np.c_[np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])]
    return q, [(int(np.searchsorted(ss, s[i])), int(min(np.searchsorted(ss, s[j]), len(q) - 1))) for i, j in arcs]


class Layout:
    def __init__(self, rng):
        U = rng.uniform
        self.lw, self.m = lw, m = U(160, 260), U(10, 40)                 # 차로 폭, 보드 끝에서 외곽 도로까지
        self.a = a = m + lw                                              # 보드 끝에서 블록까지
        mode = rng.choice(["cross", "tee", "single"], p=[0.60, 0.25, 0.15])
        has = set(ARMS) if mode != "single" else set("NS" if rng.random() < 0.5 else "EW")
        has -= {rng.choice(list(ARMS))} if mode == "tee" else set()
        self.has = has = [X for X in ARMS if X in has]             # 순서를 고정(같은 seed면 같은 도로가 나오게)
        ok = {X: {"S": ARMS[(i + 2) % 4] in has, "L": ARMS[(i + 1) % 4] in has, "R": ARMS[(i + 3) % 4] in has} for i, X in enumerate(ARMS)}
        lrow = {r: bool(all(ok[X]["L"] for X in r if X in has) and any(X in has for X in r) and rng.random() < 0.75) for r in ("NS", "EW")}   # 좌회전 전용 줄
        self.arm = {}
        for X in has:
            lp, rp = lrow["NS" if X in "NS" else "EW"], ok[X]["R"] and rng.random() < 0.7
            rp = rp and (ok[X]["S"] or not lp)                           # 직진이 없는 갈래에서 본 차로가 비지 않게
            main = {k for k in "SLR" if ok[X][k]} - ({"L"} if lp or rng.random() < 0.5 else set()) - ({"R"} if rp else set())
            main = main or ({"L"} if ok[X]["L"] and not lp else {"S"} if ok[X]["S"] else main)       # 본 차로에 할 수 있는 동작이 하나는 남게
            self.arm[X] = dict(lanes=[{"R"}] * rp + [main] + [{"L"}] * lp, rp=int(rp))
        nR, Lv, Lh = {X: self.arm[X]["rp"] if X in has else 0 for X in ARMS}, int(lrow["NS"]), int(lrow["EW"])
        bw, be, bn, bs = U(450, 1000, 4)                                 # 블록의 기본 크기(서, 동, 북, 남)
        V, H = bool(set(has) & set("NS")), bool(set(has) & set("EW"))
        # 세로 도로의 줄(서 → 동): [N 우회전] 남행 [좌회전] 북행 [S 우회전]
        xs0 = a + bw + lw * nR["N"]
        xnb = xs0 + lw * (1 + Lv) * V
        self.W = Wb = xnb + lw * V + lw * nR["S"] + be + a
        # 가로 도로의 줄(남 → 북): [W 우회전] 동행 [좌회전] 서행 [E 우회전]
        ye0 = a + bs + lw * nR["W"]
        ywb = ye0 + lw * (1 + Lh) * H
        self.H = Hb = ywb + lw * H + lw * nR["E"] + bn + a
        sx, sy = xs0 + lw * V * (1 + Lv / 2), ye0 + lw * H * (1 + Lh / 2)   # 갈래가 없을 때 양쪽 블록이 맞닿는 선
        self.seg = seg = {"N": (xs0 - lw * nR["N"], xnb + lw * V) if "N" in has else (sx, sx), "S": (xs0, xnb + lw * V + lw * nR["S"]) if "S" in has else (sx, sx),
                          "W": (ye0 - lw * nR["W"], ywb + lw * H) if "W" in has else (sy, sy), "E": (ye0, ywb + lw * H + lw * nR["E"]) if "E" in has else (sy, sy)}
        box = {"NW": (a, seg["N"][0], seg["W"][1], Hb - a), "NE": (seg["N"][1], Wb - a, seg["E"][1], Hb - a),
               "SW": (a, seg["S"][0], a, seg["W"][0]), "SE": (seg["S"][1], Wb - a, a, seg["E"][0])}
        gone = {"NW": {"tr": "N", "br": "NW", "bl": "W"}, "NE": {"tl": "N", "bl": "NE", "br": "E"}, "SW": {"tl": "W", "tr": "SW", "br": "S"}, "SE": {"tr": "E", "tl": "SE", "bl": "S"}}
        self.blocks = {}
        for k, (x0, x1, y0, y1) in box.items():                          # 모서리 반지름. 옆 갈래가 없어 다른 블록과 맞닿는 모서리는 0
            r = {c: min(U(60, 300), 0.45 * min(x1 - x0, y1 - y0)) for c in ("tl", "tr", "bl", "br")}
            for c, need in gone[k].items():
                r[c] = 0.0 if not all(X in has for X in need) else r[c]
            self.blocks[k] = dict(box=(x0, x1, y0, y1), r=r)
        rad = lambda kc: self.blocks[kc[0]]["r"][kc[1]]
        # 갈래마다: 들어가는 차로들의 위치(바깥쪽부터), 나가는 차로의 위치, 교차로 쪽 끝
        cen = {"N": ([xs0 - lw / 2] * nR["N"] + [xs0 + lw / 2] + [xs0 + 1.5 * lw] * Lv, xnb + lw / 2), "S": ([xnb + 1.5 * lw] * nR["S"] + [xnb + lw / 2] + [xs0 + 1.5 * lw] * Lv, xs0 + lw / 2),
               "W": ([ye0 - lw / 2] * nR["W"] + [ye0 + lw / 2] + [ye0 + 1.5 * lw] * Lh, ywb + lw / 2), "E": ([ywb + 1.5 * lw] * nR["E"] + [ywb + lw / 2] + [ye0 + 1.5 * lw] * Lh, ye0 + lw / 2)}
        pick = lambda p, q, i: seg[p][i] if p in has else seg[q][i]
        end = {"N": (pick("W", "E", 1), pick("E", "W", 1)), "S": (pick("E", "W", 0), pick("W", "E", 0)),      # (들어가는 차로가 끝나는 곳, 나가는 차로가 시작하는 곳)
               "W": (pick("S", "N", 0), pick("N", "S", 0)), "E": (pick("N", "S", 1), pick("S", "N", 1))}
        # 외곽 도로: 중심선은 네모(x0r..x1r, y0r..y1r)를 왼쪽 위에서 시계 방향으로 도는 거리 s로 적는다
        x0r, y0r = m + lw / 2, m + lw / 2
        x1r, y1r = Wb - x0r, Hb - y0r
        Lx, Ly = x1r - x0r, y1r - y0r
        self.P = P = 2 * (Lx + Ly)
        self.rc = [(0.0, (x0r, y1r), rad(("NW", "tl")) + lw / 2), (Lx, (x1r, y1r), rad(("NE", "tr")) + lw / 2),
                   (Lx + Ly, (x1r, y0r), rad(("SE", "br")) + lw / 2), (2 * Lx + Ly, (x0r, y0r), rad(("SW", "bl")) + lw / 2)]
        def pt(s):
            s = s % P
            return (x0r + s, y1r) if s < Lx else (x1r, y1r - (s - Lx)) if s < Lx + Ly else (x1r - (s - Lx - Ly), y0r) if s < 2 * Lx + Ly else (x0r, y0r + (s - 2 * Lx - Ly))
        self.pt = pt
        s_of = {"N": lambda c: c - x0r, "E": lambda c: Lx + y1r - c, "S": lambda c: Lx + Ly + x1r - c, "W": lambda c: 2 * Lx + Ly + c - y0r}
        xy = lambda X, c, v: (c, v) if X in "NS" else (v, c)             # 갈래 X에서 줄 위치 c, 도로 방향 좌표 v
        for X in has:
            A, (ins, out) = self.arm[X], cen[X]
            A.update(c_in=ins, c_out=out, s_in=[s_of[X](c) for c in ins], s_out=s_of[X](out), r_in=rad(UP[X]) + lw / 2, r_out=rad(DOWN[X]) + lw / 2,
                     p_end=[xy(X, c, end[X][0]) for c in ins], p_start=xy(X, out, end[X][1]), end=end[X])
            A["en"], A["ex"] = A["s_in"][0] - A["r_in"], A["s_out"] + A["r_out"]
        self.rad = rad
        self._routes()

    def ring(self, s0, s1):
        """외곽 중심선에서 s0부터 s1 사이의 꺾이는 점들. ([s0의 점, 모서리들...], [모서리 반지름들...])"""
        s1 += self.P * (s1 <= s0)
        cs = sorted((s + k * self.P, p, r) for s, p, r in self.rc for k in (0, 1, 2) if s0 < s + k * self.P < s1)
        return [self.pt(s0)] + [p for _, p, _ in cs], [r for _, _, r in cs]

    def _routes(self):
        """경로들: (이름, 종류, 들어갈 차로 번호, 꺾이는 점들, 반지름들, 교차로 구간, 들어오는 차로, 나가는 차로). 차로 = ("R",) 외곽 또는 (갈래, "in"·"out", 번호)."""
        order = sorted(self.has, key=lambda X: self.arm[X]["en"])
        self.routes, lw = [], self.lw
        add = lambda *r: self.routes.append(r)
        for i, X in enumerate(order):
            A, prev, nxt = self.arm[X], self.arm[order[i - 1]], self.arm[order[(i + 1) % len(order)]]
            for k, s in enumerate(A["s_in"]):                            # 외곽 → 들어가는 차로(우회전)
                pts, rr = self.ring(prev["ex"], s)
                add(f"R>{X}{k}", "right", k + 1, pts + [self.pt(s), A["p_end"][k]], rr + [A["r_in"]], len(rr), ("R",), (X, "in", k))
            pts, rr = self.ring(A["s_out"], nxt["en"])                   # 나가는 차로 → 외곽(우회전)
            add(f"{X}>R", "right", 1, [A["p_start"], self.pt(A["s_out"])] + pts[1:] + [self.pt(nxt["en"])], [A["r_out"]] + rr, 0, (X, "out", 0), ("R",))
            pts, rr = self.ring(prev["ex"], nxt["en"])                   # 외곽이 이 들머리를 그대로 지남(직진)
            add(f"R>R {X}", "straight", 1, pts + [self.pt(nxt["en"])], rr, (self.pt(A["en"]), self.pt(A["ex"])), ("R",), ("R",))
        near = lambda X, c, r, s: tuple(np.array(self.pt(s)) + r * np.array(DIN[X]))          # 차로에서 외곽 쪽 끝(외곽으로 도는 원호가 닿는 곳)
        for i, X in enumerate(ARMS):
            if X not in self.has:
                continue
            A = self.arm[X]
            for k, mv in enumerate(A["lanes"]):
                b_in = near(X, A["c_in"][k], A["r_in"], A["s_in"][k])
                for m_, kind, j in (("S", "straight", 2), ("L", "left", 1), ("R", "right", 3)):
                    T = ARMS[(i + j) % 4]
                    if m_ not in mv or T not in self.has:
                        continue
                    B = self.arm[T]
                    b_out, pe, ps = near(T, B["c_out"], B["r_out"], B["s_out"]), A["p_end"][k], B["p_start"]
                    if kind == "straight":
                        add(f"{X}{k}>{T}", kind, 1, [b_in, b_out], [], (pe, ps), (X, "in", k), (T, "out", 0))
                        continue
                    C = (pe[0], ps[1]) if X in "NS" else (ps[0], pe[1])
                    leg = min(np.hypot(C[0] - b_in[0], C[1] - b_in[1]), np.hypot(C[0] - b_out[0], C[1] - b_out[1])) - 1.0
                    if kind == "right":
                        add(f"{X}{k}>{T}", kind, 1, [b_in, C, b_out], [min(self.rad(INNER[X]) + lw / 2, leg)], 0, (X, "in", k), (T, "out", 0))
                    else:
                        R = 0.95 * min(np.hypot(C[0] - pe[0], C[1] - pe[1]), np.hypot(C[0] - ps[0], C[1] - ps[1]))
                        add(f"{X}{k}>{T}", kind, 1, [b_in, C, b_out], [min(max(R, lw), leg)], (pe, ps), (X, "in", k), (T, "out", 0))
