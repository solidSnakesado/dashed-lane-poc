"""내 차로의 양쪽 선(lane1 = 왼쪽, lane2 = 오른쪽)을 고르고 중심선을 만든다 (6단계: 차선 번호)."""
import numpy as np

from lane_motion import Path, nearest

ROLE_LEN = 3.0        # 내 차로 선으로 인정하는 최소 길이(m)
ROLE_TURN = 30.0      # 내 차로 선은 차량 진행 방향과 이 각(도) 안으로 나란해야 함 (가로지르는 유도선 제외)
ROLE_MARGIN = 1.0     # 내 차로 선을 다른 선으로 바꾸려면 이만큼(m) 더 가까워야 함
Z_REF = 4.0           # 차선 번호를 정하는 기준 거리(차량 앞, m)
MAX_SIDE = 3.5        # 내 차로 경계로 인정하는 최대 옆 거리(m)


def x_at(curve, z_ref):
    """곡선이 차량 앞 z_ref(m) 거리를 지날 때의 옆 위치(m). 그 거리에 닿지 않으면 가까운 끝을 늘려서 구한다."""
    p = curve.sample(0.5)
    z = p[:, 1] - z_ref
    cross = np.where(z[:-1] * z[1:] <= 0)[0]
    if len(cross):
        a, b = p[cross[0]], p[cross[0] + 1]
        return float(a[0] + (b[0] - a[0]) * (z_ref - a[1]) / (b[1] - a[1] + 1e-9))
    e, d, lim = (p[0], p[1] - p[0], 20.0) if z[0] > 0 else (p[-1], p[-1] - p[-2], 5.0)
    if abs(e[1] - z_ref) > lim or abs(d[1]) < 0.3 * np.linalg.norm(d):
        return None
    return float(e[0] + d[0] * (z_ref - e[1]) / d[1])


def pick(tracks, prev, min_hits):
    """추적 중인 선들에서 내 차로의 (왼쪽 선, 오른쪽 선)을 고른다. 없으면 None. prev = 직전에 고른 (왼쪽, 오른쪽).

    직전에 고른 선이 조건에 맞으면 유지하고, 다른 선이 뚜렷이 더 가까울 때만 바꾼다(번호가 오락가락하지 않게).
    """
    cand = {0: [], 1: []}
    for t in tracks:
        x, x0, x1 = (x_at(t.curve, Z_REF + dz) for dz in (0.0, -3.0, 3.0))
        if x is None or t.hits < min_hits or t.curve.length < ROLE_LEN:
            continue
        if x0 is not None and x1 is not None and abs(np.degrees(np.arctan2(x1 - x0, 6.0))) > ROLE_TURN:
            continue
        if -MAX_SIDE < x < MAX_SIDE:
            cand[0 if x < 0 else 1].append((abs(x), t))
    out = []
    for side in (0, 1):
        best = min(cand[side], key=lambda c: c[0], default=None)
        keep = next((c for c in cand[side] if c[1] is prev[side]), None)
        # 직전 선이 잠깐 안 보이는 중이면(다른 보이는 선도 없을 때) 그대로 두고, 보이는 중이면 다른 선이 뚜렷이 더 가까울 때만 바꿈
        seen = [c for c in cand[side] if c[1].det is not None]
        if keep is not None and keep[1].det is None and seen:     # 직전 선은 안 보이는데 지금 보이는 선이 있으면 그 선으로
            best = min(seen, key=lambda c: c[0])
        elif keep is not None and (keep[1].det is None or best[0] > keep[0] - ROLE_MARGIN):
            best = keep
        out.append(best[1] if best else None)
    return out[0], out[1]


def center_line(left, right, width):
    """내 차로 양쪽 선으로 중심선을 만든다. (곡선 또는 None, 추정 여부, 갱신한 차로 폭)

    양쪽 선이 있으면 그 가운데를 잇고 차로 폭을 천천히 갱신한다.
    한쪽뿐이면 그 선에서 차로 폭의 절반만큼 옆으로 옮긴 선을 쓴다(추정).
    """
    if left is None and right is None:
        return None, True, width
    if left is not None and right is not None:
        a, b = left.curve.sample(1.0), right.curve.sample(0.25)
        idx, dist = nearest(a, b)
        ok = (dist > 2.0) & (dist < 5.0)
        if ok.sum() >= 2:
            width += 0.2 * (float(np.median(dist[ok])) - width)
            return Path((a[ok] + b[idx[ok]]) / 2), False, width
    t, sign = (left, 1.0) if left is not None else (right, -1.0)
    p = t.curve.sample(1.0)
    d = np.gradient(p, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return Path(p + sign * width / 2 * np.c_[d[:, 1], -d[:, 0]]), True, width   # 오른쪽 방향 = (dz, -dx)
