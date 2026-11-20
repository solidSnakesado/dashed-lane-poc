"""차량이 움직여서 생기는 '선들의 한꺼번에 이동'을 선 자체에서 추정한다 (추적의 예측 단계).

바퀴 센서 같은 이동량 입력 없이, 이전 프레임의 선과 새 프레임의 선을 견주어 구한다.
- 옆 이동과 회전: 선에 수직인 방향의 어긋남으로 구한다 (선은 길이 방향 움직임을 알려 주지 못함).
- 앞뒤 이동: 점선 조각의 끝점이 프레임 사이에 선을 따라 얼마나 밀렸는지로 구한다.
또, 최종 곡선이 프레임마다 떨리지 않도록 고정된 거리 지점마다 시간 방향으로 평활한다(smooth).
"""
import numpy as np

from lane_group import AHEAD, Curve

MOTION_GATE = 3.0     # 옆 이동·회전 추정에 쓰는 점 쌍의 최대 거리(m)
END_GATE = 2.5        # 같은 조각 끝점으로 볼 최대 거리(m)
MIN_ENDS = 4          # 앞뒤 이동을 추정하려면 필요한 끝점 쌍의 수
EXTEND = 8.0          # 대응 찾을 때 선의 양 끝을 늘려 보는 거리(m)
SMOOTH_SEC = 0.25     # 최종 곡선을 평활하는 시간 상수(초). 클수록 덜 떨리지만 변화에 늦게 따라감
TIP_SEC = 0.6         # 선 끝의 방향·휘는 정도·그릴 거리를 평활하는 시간 상수(초). 클수록 끝이 덜 흔들림
REACH_MAX = 50.0      # 앞으로 이어 그리는 최대 거리(m)
JUMP = 1.0            # 평활 전후 차이가 이보다 크면(m) 선이 바뀐 것으로 보고 새 값으로 바로 바꿈


def nearest(pts, ref):
    """pts의 각 점에서 ref 점열까지 가장 가까운 점의 번호와 거리."""
    d = np.linalg.norm(pts[:, None, :] - ref[None, :, :], axis=2)
    return d.argmin(axis=1), d.min(axis=1)


def extended(curve):
    """곡선을 촘촘히 찍고 양 끝을 늘린 점열과, 그 안에서 원래 곡선이 차지하는 범위 (점열, 시작 번호, 끝 번호).

    가까운 쪽은 직선으로, 먼 쪽은 휘는 정도를 유지한 원호로 늘린다. 대응 찾기와 점 순서 매기기에 쓴다.
    """
    p = curve.sample(0.25)
    d0 = (p[0] - p[1]) / max(np.linalg.norm(p[0] - p[1]), 1e-9)
    pre = (p[0] + np.arange(0.25, EXTEND, 0.25)[:, None] * d0)[::-1]
    return np.vstack([pre, p, curve.arc(EXTEND, 0.25)]), len(pre), len(pre) + len(p)


def lateral(tracks, lanes, pairs):
    """대응된 선들이 한꺼번에 얼마나 돌고 옆으로 옮겨졌는지 추정한다. (회전 행렬, 이동)"""
    rows, rhs = [], []
    for ti, li in pairs:
        ref = tracks[ti].curve.sample(0.25)
        new = lanes[li].sample(1.0)
        idx, dist = nearest(new, ref)
        tan = np.gradient(ref, axis=0)[idx]
        nrm = np.c_[-tan[:, 1], tan[:, 0]] / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        ok = dist <= MOTION_GATE
        old = ref[idx][ok]
        rows.append(np.c_[np.sum(nrm[ok] * np.c_[-old[:, 1], old[:, 0]], axis=1), nrm[ok]])
        rhs.append(np.sum(nrm[ok] * (new[ok] - old), axis=1))
    if not rows or sum(len(r) for r in rows) < 8:
        return np.eye(2), np.zeros(2)
    A, b = np.vstack(rows), np.concatenate(rhs)
    w = np.sqrt(len(b))                              # 값이 튀지 않도록 약하게 0쪽으로 당김
    A = np.vstack([A, np.diag([3.0 * w, 0.05 * w, 0.5 * w])])   # 각도(rad), 옆 이동, 앞뒤 이동
    th, tx, tz = np.linalg.lstsq(A, np.r_[b, 0, 0, 0], rcond=None)[0]
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]]), np.array([tx, tz])


def piece_ends(lanes):
    """선들의 조각 시작점, 끝점, 조각 방향을 모은다. (시작점들, 끝점들, 방향들)"""
    pieces = [p for ln in lanes for p in ln.pieces]
    if not pieces:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    return (np.array([p.start for p in pieces]), np.array([p.end for p in pieces]),
            np.array([p.direction(True) for p in pieces]))


def along(old_ends, lanes):
    """조각 끝점이 선을 따라 밀린 양으로 앞뒤 이동(벡터)을 추정한다. 근거가 부족하면 None.

    old_ends = 이전 프레임의 (시작점들, 끝점들). 시작점은 시작점끼리, 끝점은 끝점끼리 견준다.
    """
    starts, ends, dirs = piece_ends(lanes)
    moves, tans = [], []
    for new, old in ((starts, old_ends[0]), (ends, old_ends[1])):
        if len(new) == 0 or len(old) == 0:
            continue
        idx, dist = nearest(new, old)
        ok = dist <= END_GATE
        moves += list(np.sum((new[ok] - old[idx[ok]]) * dirs[ok], axis=1))
        tans += list(dirs[ok])
    if len(moves) < MIN_ENDS:
        return None
    tan = np.mean(tans, axis=0)
    return float(np.median(moves)) * tan / max(np.linalg.norm(tan), 1e-9)


class Path(Curve):
    """점열을 그대로 잇는 곡선(다항식을 맞추지 않음). Curve와 쓰는 법이 같다."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, float)
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1))]
        self.rms = 0.0

    def at(self, s):
        s = np.atleast_1d(np.asarray(s, float))
        return np.c_[np.interp(s, self.s, self.pts[:, 0]), np.interp(s, self.s, self.pts[:, 1])]

    def heading(self, s):
        a, b = self.at([max(s - 1.0, 0.0), min(s + 1.0, self.length)])
        return float(np.arctan2(b[0] - a[0], b[1] - a[1]))


def stations(prev, curve):
    """곡선을 1m 간격의 고정된 거리(z) 지점으로 나타낸다. (지점 z, 이번 옆 위치, 직전 평활값 또는 nan)

    선이 옆으로 누워 거리 지점으로 나타낼 수 없으면 None.
    """
    raw = curve.sample(0.5)
    z = raw[:, 1]
    st = np.arange(np.ceil(z[0]), z[-1] + 1e-6, 1.0)
    if np.any(np.diff(z) <= 0) or len(st) < 2:
        return None
    old = np.full(len(st), np.nan)
    if prev is not None:
        o = np.argsort(prev[:, 1])
        old = np.interp(st, prev[o, 1], prev[o, 0], left=np.nan, right=np.nan)
    return st, np.interp(st, z, raw[:, 0]), old


def common(items):
    """모든 선이 이번 프레임에 함께 옆으로 움직인 양(m). 카메라 흔들림처럼 실제로 생긴 움직임이다.

    items = [(직전 평활 점열, 이번 곡선)]. 30m 안쪽 지점들의 '이번 값 - 직전 값'의 중앙값을 쓴다.
    """
    diff = []
    for prev, curve in items:
        s = stations(prev, curve)
        if s is not None:
            ok = ~np.isnan(s[2]) & (s[0] <= 30)
            diff += list((s[1] - s[2])[ok])
    return float(np.median(diff)) if len(diff) >= 10 else 0.0


def smooth(prev, curve, fps, shift=0.0):
    """이번 프레임의 곡선을 직전까지의 평활 결과(prev, 차량 움직임만큼 이미 옮겨 둔 점열)와 섞는다.

    모든 선이 함께 움직인 양(shift)은 그대로 따라가고, 선마다 따로 흔들리는 부분만 지수 평활한다.
    돌려주는 값: (평활한 점열, 그 곡선). 거리 지점으로 나타낼 수 없는 선은 (None, 원래 곡선).
    """
    s = stations(prev, curve)
    if s is None:
        return None, curve
    st, x, old = s
    old = old + shift
    ok = ~np.isnan(old) & (np.abs(x - np.nan_to_num(old, nan=1e9)) <= JUMP)
    alpha = 1.0 - np.exp(-1.0 / (fps * SMOOTH_SEC))
    x[ok] = old[ok] + alpha * (x[ok] - old[ok])
    sm = np.c_[x, st]
    return sm, Path(sm)


def tip_of(curve):
    """선 끝의 상태 [방향(rad), 휘는 정도(1/m), 앞으로 그릴 거리 z(m)].

    끝부분만 보면 프레임마다 크게 바뀌므로, 선 전체에 2차식 x(z)를 맞춰(가까운 점에 큰 가중치) 구한다.
    """
    p = curve.sample(0.5)
    z = p[:, 1]
    reach = min(z[-1] + min(AHEAD, curve.length), REACH_MAX)
    if curve.length < 5 or np.any(np.diff(z) <= 0):
        return np.array([curve.heading(curve.length), 0.0, reach])
    deg = 2 if curve.length >= 12 else 1
    c = np.polyfit(z, p[:, 0], deg, w=1.0 / (1.0 + (z / 30.0) ** 2))
    slope = np.polyval(np.polyder(c), z[-1])
    kappa = 2 * c[0] / (1 + slope ** 2) ** 1.5 if deg == 2 else 0.0
    return np.array([np.arctan(slope), np.clip(kappa, -0.05, 0.05), reach])


def blend_tip(old, curve, fps):
    """이번 프레임의 끝 상태를 직전까지의 값(old)과 섞어 천천히 따라가게 한다."""
    new = tip_of(curve)
    return new if old is None else old + (1.0 - np.exp(-1.0 / (fps * TIP_SEC))) * (new - old)
