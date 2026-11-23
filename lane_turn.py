"""회전(좌회전·우회전)할 때의 차로 중심선을 만든다. lane_direct.py가 쓴다.

두 가지 재료:
  between: 모델이 그린 회전 왼쪽 선·오른쪽 선 → 그 가운데. 한쪽 선만 보이면 차로 폭의 절반만큼 옮긴다.
  edge:    모델이 그린 회전 면 → 면의 바깥쪽 경계에서 차로 폭의 절반만큼 안쪽.
바깥쪽 = 좌회전은 오른쪽, 우회전은 왼쪽. 안쪽 경계는 화면 밖으로 잘리기 쉬워 바깥쪽을 기준으로 삼는다.
도로 크기에 걸린 숫자는 쓰지 않는다. 거리 기준은 카메라 높이나 차로 폭의 배수다.
"""
import cv2
import numpy as np

REACH = 12.0     # 카메라 높이의 이 배수까지만 선을 내보냄 (58 mm → 약 0.7 m). 그보다 먼 바닥은 위치 오차가 큼
SPAN = 2.0       # 선을 다듬을 때 주변으로 보는 거리 = 카메라 높이의 이 배수
EDGE = 2         # 화면 끝에서 이 픽셀 안에 닿은 경계는 잘린 것으로 봄
MIN_PX = 30      # 마스크 픽셀이 이보다 적으면 선으로 쓰지 않음
MAX_PTS = 600    # 선 하나에 쓰는 마스크 픽셀 수의 상한 (계산량 제한)
PAIR = (0.5, 1.6)   # 왼쪽 선과 오른쪽 선의 간격이 차로 폭의 이 범위 안일 때만 짝으로 봄


def shift(line, d):
    """선(가는 순서의 점열)을 진행 방향 오른쪽으로 d만큼 옮긴다."""
    t = np.gradient(line, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return line + d * np.c_[t[:, 1], -t[:, 0]]


def between(left, right, lane_w):
    """회전 왼쪽 선·오른쪽 선(가는 순서의 점열, 없으면 None) → 중심선 (N,2). 만들 수 없으면 None.
    둘 다 있으면 긴 쪽을 따라가며 맞은편에서 가장 가까운 점과의 가운데를 잡고, 맞은편이 없는 구간과 한쪽만 있을 때는 차로 폭의 절반만큼 옮긴다."""
    if left is None and right is None:
        return None
    if left is None or right is None:
        return None if not lane_w else shift(left, lane_w / 2) if right is None else shift(right, -lane_w / 2)
    ref, oth, sg = (left, right, 1.0) if len(left) >= len(right) else (right, left, -1.0)
    d = np.linalg.norm(ref[:, None] - oth[None], axis=2)
    j, gap = d.argmin(axis=1), d.min(axis=1)
    inside = (j > 0) & (j < len(oth) - 1)                  # 맞은편 선의 끝점에 붙은 짝은 버린다(그 너머는 맞은편이 안 보이는 구간)
    ok = inside & (gap >= PAIR[0] * lane_w) & (gap <= PAIR[1] * lane_w) if lane_w else inside
    mid = (ref + oth[j]) / 2
    if lane_w:                                             # 짝이 없는 구간: 보이는 선에서 절반만큼
        mid[~ok] = shift(ref, sg * lane_w / 2)[~ok]
    elif not ok.any():
        return None
    return mid if len(mid) >= 2 else None


def edge(cam, cam_h, prev, polys, size, lane_w, outer):
    """회전 면 → 차로 중심선 (N,2). 면을 따라 한 걸음씩 가면서, 가는 방향에 가로로 자른 단면의 바깥쪽 끝에서 차로 폭의 절반만큼 안쪽을 잡는다.
    outer = 바깥쪽(오른쪽 +1: 좌회전, 왼쪽 -1: 우회전). prev = 직전 프레임의 중심선(출발점과 출발 방향을 여기서 얻는다).
    면의 가운데를 쓰지 않는 이유: 휘어 나가는 면은 화면 끝에서 잘려, 보이는 부분의 가운데가 바깥쪽으로 치우친다.
    바깥쪽 끝이 화면 밖으로 잘린 단면에서는 안쪽 끝에서 절반만큼 바깥을 잡고, 양쪽이 다 잘렸으면 가던 방향을 유지한다."""
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polys], 1)
    v, u = np.nonzero(mask)
    if len(u) < MIN_PX:
        return None
    pick = np.linspace(0, len(u) - 1, min(len(u), 5 * MAX_PTS)).astype(int)
    g, ok = cam.to_ground(np.c_[u[pick], v[pick]].astype(float))
    ok &= (g[:, 1] > 0) & (np.linalg.norm(g, axis=1) <= REACH * cam_h)
    cut = {-1: mask[v[pick], : EDGE + 1].any(axis=1)[ok], 1: mask[v[pick], -EDGE - 1:].any(axis=1)[ok]}   # 그 줄이 화면 왼쪽·오른쪽 끝에 닿았는지
    g = g[ok]
    if len(g) < MIN_PX:
        return None
    step, d, path = SPAN * cam_h / 2, np.array([0.0, 1.0]), []
    c = np.array([np.median(g[g[:, 1] <= g[:, 1].min() + step, 0]), g[:, 1].min() - step / 2])
    if prev is not None and len(prev) >= 2:                # 출발점과 출발 방향: 직전 중심선이 면에 처음 닿는 곳
        hit = np.nonzero(np.linalg.norm(prev[:, None] - g[None, ::7], axis=2).min(axis=1) <= step / 2)[0]
        if len(hit):
            k = min(int(hit[0]), len(prev) - 2)
            d = (prev[k + 1] - prev[k]) / (np.linalg.norm(prev[k + 1] - prev[k]) + 1e-9)
            c = prev[k] - step / 2 * d
    for _ in range(int(2 * REACH * cam_h / step)):
        c2, n = c + step * d, np.array([d[1], -d[0]])
        rel = g - c2
        sel = (np.abs(rel @ d) <= step / 2) & (np.abs(rel @ n) <= 1.5 * lane_w)
        if sel.sum() < 5:
            break
        s = rel[sel] @ n * outer                           # 바깥쪽이 +가 되게
        i, j = int(np.argmax(s)), int(np.argmin(s))
        if not cut[outer][sel][i]:
            c2 = c2 + outer * (np.percentile(s, 95) - lane_w / 2) * n
        elif not cut[-outer][sel][j]:
            c2 = c2 + outer * (np.percentile(s, 5) + lane_w / 2) * n
        d = (c2 - c) / (np.linalg.norm(c2 - c) + 1e-9)
        path.append(c2)
        c = c2
    return np.array(path) if len(path) >= 2 else None
