"""점 구름을 '선을 따라가는 순서'의 점열로 바꾼다. lane_direct.py가 쓴다.

앞 거리 순서로 정렬하지 않는 이유: 교차로에서 돌 때는 갈 길이 옆으로 누워서 앞 거리로는 순서를 정할 수 없다.
"""
import numpy as np


def trace(q, w, r, reach, side=0.0):
    """점 구름 q (N,2) [오른쪽 m, 앞 m]와 가중치 w → 로봇에서 가장 가까운 곳부터 선을 따라가는 순서의 점열, 지나간 점 표시.
    r = 주변으로 보는 거리. 한 걸음(r/2)씩, 진행 방향 앞쪽 r 안에 있는 점들의 무게중심 쪽으로 나아간다.
    출발 방향: 앞쪽. 선이 옆으로 누워 있으면 side(왼쪽 -1, 오른쪽 +1) 쪽, side가 없으면 점이 많은 쪽."""
    seen, step = np.zeros(len(q), bool), r / 2
    near = np.linalg.norm(q - q[np.argmin(np.linalg.norm(q, axis=1))], axis=1) <= r
    c = np.average(q[near], axis=0, weights=w[near])
    rel = q[near] - c
    d = np.linalg.eigh((rel * w[near, None]).T @ rel)[1][:, -1]
    far = ((q - c) @ d) @ w
    d = d * np.sign(d[1] if abs(d[1]) >= 0.3 else d[0] * side if side else far or 1.0)
    path = [c]
    for _ in range(int(2 * reach / step)):
        rel = q - c
        dist = np.linalg.norm(rel, axis=1)
        seen |= dist <= r
        cand = (dist <= r) & (rel @ d > 0)
        if cand.sum() < 2:
            break
        dn = np.average(q[cand], axis=0, weights=w[cand]) - c
        if np.linalg.norm(dn) < 1e-6:
            break
        dn /= np.linalg.norm(dn)
        c2 = c + step * dn
        nb = np.linalg.norm(q - c2, axis=1) <= step
        if nb.any():                                       # 옆으로만 구름 가운데에 맞춤
            off = np.average(q[nb], axis=0, weights=w[nb]) - c2
            c2 = c2 + off - (off @ dn) * dn
        if len(path) > 2 and np.linalg.norm(np.array(path[:-1]) - c2, axis=1).min() < step / 2:
            break                                          # 제자리로 돌아옴
        path.append(c2)
        c, d = c2, dn
    p = np.array(path)
    if len(p) > 2:
        p[1:-1] = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4       # 가볍게 다듬음
    return p, seen
