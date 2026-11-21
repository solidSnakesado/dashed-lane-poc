"""선 하나를 곡선으로 나타낸다 (5단계: 곡선 맞춤).

기본은 점열 전체에 식 하나(다항식)를 맞춘다: 완만한 실제 도로용.
LOCAL_FIT을 켜면 식 하나로 뭉뚱그리지 않고 본 점들을 따라간다: 급한 곡선용(핑키). 선을 따라 일정 간격의 지점마다
그 주변 점들만으로 작은 2차식을 맞추므로(가까운 점일수록 큰 가중치), 선이 얼마나 휘든 따라가고 점선 사이의 빈 구간도 잇는다.
"""
import numpy as np

AHEAD = 20.0        # 선 끝에서 앞으로 더 이어 그리는 최대 거리(m)
FAR_W = 30.0        # 이 거리(m)보다 먼 점은 곡선 맞춤에서 가중치를 낮춤 (멀수록 위치 오차가 큼)
LOCAL_FIT, FIT_SPAN = 0.0, 4.0   # 급한 곡선용 맞춤을 쓸지(0 = 식 하나로 맞춤), 그때 주변으로 보는 거리(m, 점선 주기의 절반)
KAPPA_MAX = 0.1     # 선 끝을 이어 그릴 때 허용하는 가장 급한 휨(1/m). 0.1 = 반경 10 m


def local_fit(pts, w, span, step=0.5):
    """pts = 점 (N,2, 대체로 선을 따라간 순서), w = 점 가중치, span = 주변으로 보는 거리(m).

    점의 순서가 군데군데 뒤섞여 있어도 되도록, 선이 뻗은 방향으로 잰 위치를 기준으로 맞춘다.
    (선이 뻗은 방향에서 90도 넘게 꺾여 되돌아오는 선은 다루지 못한다.)
    돌려주는 값: 지점까지 선을 따라간 거리 (G,), 지점 위치 (G,2), 지점의 진행 방향 (G,2 단위 벡터), 점마다의 거리 (N,)."""
    c = np.average(pts, axis=0, weights=w)
    axis = np.linalg.eigh(np.cov((pts - c).T, aweights=w))[1][:, -1]          # 선이 뻗은 방향
    n = max(len(pts) // 4, 1)
    axis = axis if (pts[-n:].mean(axis=0) - pts[:n].mean(axis=0)) @ axis >= 0 else -axis   # 원래 순서와 같은 쪽을 앞으로
    u = (pts - c) @ axis
    u = u - u.min()
    g = np.linspace(0.0, u.max(), max(int(u.max() / step) + 1, 2))
    du = u[None, :] - g[:, None]                                     # (G, N) 지점에서 각 점까지의 거리
    k = w[None, :] * np.exp(-0.5 * (du / span) ** 2)
    k = k / (k.sum(axis=1, keepdims=True) + 1e-12)
    basis = np.stack([np.ones_like(du), du, du ** 2], axis=2)        # (G, N, 3): 1, 거리, 거리²
    m = np.einsum("gn,gni,gnj->gij", k, basis, basis)
    # 주변 점이 적거나 한쪽에 몰린 지점은 휘는 항·기울기 항을 눌러 엉뚱하게 휘지 않게 함
    m = m + np.diag([0.0, 1e-3 * span ** 2, 1e-2 * span ** 4])
    p = np.linalg.solve(m, np.einsum("gn,gni,nc->gic", k, basis, pts))[:, 0]   # 지점 위치 (G, 2)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]   # 맞춘 선을 따라간 거리
    d = np.gradient(p, arc + np.arange(len(arc)) * 1e-9, axis=0)
    return arc, p, d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12), np.interp(u, g, arc)


class Curve:
    """점열에 맞춘 매개변수 곡선 x(s), z(s). s = 선을 따라간 거리(m)."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, float)
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1))]
        # 짧은 선일수록 낮은 차수를 써서 엉뚱하게 휘는 것을 막음
        deg = 1 if self.s[-1] < 6 else 2 if self.s[-1] < 20 else 3
        deg = min(deg, len(self.pts) - 1)
        w = 1.0 / (1.0 + (np.linalg.norm(self.pts, axis=1) / FAR_W) ** 2)
        self.cx, self.cz = (np.polyfit(self.s, self.pts[:, i], deg, w=w) for i in (0, 1))
        # 급한 곡선용(LOCAL_FIT): 식 하나 대신 지점마다 주변 점으로 맞춘 값을 씀 (lane_curve.py)
        self.loc = None
        if LOCAL_FIT and self.s[-1] > 0:
            *self.loc, self.s = local_fit(self.pts, w, FIT_SPAN)
        err = np.linalg.norm(self.at(self.s) - self.pts, axis=1)
        self.rms = float(np.sqrt(np.average(err ** 2, weights=w)))   # 맞춤 오차(m)

    @property
    def length(self):
        return float(self.s[-1] if getattr(self, "loc", None) is None else self.loc[0][-1])

    def at(self, s):
        """거리 s에서의 바닥 좌표 (N,2)."""
        s = np.atleast_1d(np.asarray(s, float))
        if getattr(self, "loc", None) is not None:   # 지점 사이는 잇고, 양 끝 밖은 끝 방향으로 곧게 늘림
            g, p, d = self.loc
            c, out = np.clip(s, 0.0, g[-1]), s - np.clip(s, 0.0, g[-1])
            return np.c_[[np.interp(c, g, p[:, i]) + out * np.interp(c, g, d[:, i]) for i in (0, 1)]].T
        return np.c_[np.polyval(self.cx, s), np.polyval(self.cz, s)]

    def sample(self, step=0.5):
        """곡선을 일정 간격으로 찍은 점열."""
        return self.at(np.linspace(0, self.length, max(int(self.length / step) + 1, 2)))


    def heading(self, s):
        """거리 s에서의 진행 방향 각(rad). 0 = 정면(+z), + = 오른쪽."""
        if getattr(self, "loc", None) is not None:
            return float(np.arctan2(*(np.interp(np.clip(s, 0.0, self.loc[0][-1]), self.loc[0], self.loc[2][:, i]) for i in (0, 1))))
        return float(np.arctan2(np.polyval(np.polyder(self.cx), s), np.polyval(np.polyder(self.cz), s)))

    def arc(self, dist, step=0.5, tip=None):
        """선 끝에서 dist만큼 더 나아간 원호 점열. tip = 평활해 둔 (방향, 휘는 정도)를 주면 그 값을 쓴다."""
        back = min(10.0, self.length)                # 끝에서 이 구간의 방향 변화로 휘는 정도를 구함
        th = self.heading(self.length)
        dth = (th - self.heading(self.length - back) + np.pi) % (2 * np.pi) - np.pi
        kappa = float(np.clip(dth / back, -KAPPA_MAX, KAPPA_MAX)) if self.length >= 12 else 0.0
        if tip is not None:
            th, kappa = float(tip[0]), float(tip[1])
        if abs(kappa) > 1e-6:                        # 나아가는 동안 방향이 20도 넘게 돌지 않게 제한
            dist = min(dist, np.radians(20) / abs(kappa))
        ang = th + kappa * np.arange(step, dist + step, step)
        return self.at(self.length)[0] + np.c_[np.cumsum(np.sin(ang)), np.cumsum(np.cos(ang))] * step

    def ahead(self):
        """화면에 앞으로 더 이어 그릴 점열. 본 길이보다 길게는 늘리지 않는다(짧게 본 선을 멀리 늘리면 틀리기 쉬움)."""
        tip = getattr(self, "tip", None)             # 추적이 평활해 둔 (끝 방향, 휘는 정도, 그릴 거리 z)
        if tip is not None:
            dist = tip[2] - self.at(self.length)[0, 1]
            return self.arc(dist, tip=tip) if dist >= 1 else np.zeros((0, 2))
        dist = min(AHEAD, self.length)
        return self.arc(dist) if dist >= 3 else np.zeros((0, 2))
