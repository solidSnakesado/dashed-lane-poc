"""2차 체크: 지도와 위치로 계산한 '지금 보여야 할 차선'과 카메라가 찾은 칠해진 선을 비교해, 위치가 얼마나 틀렸는지 낸다.

주행은 도로 그래프(Route Server)와 RPP가 하고, 이 처리는 그게 맞게 가고 있는지만 본다.
  지도 쪽: 지금 달리는 차로의 왼쪽 선·오른쪽 선(세계 좌표). 칠해진 선이 없는 구간(교차로 안)은 체크하지 않는다.
  비전 쪽: 모델이 그린 내 차로 왼쪽 선·오른쪽 선(카메라 기준 바닥 좌표, lane_direct.py가 만든 것).
  비교: 지도 쪽 선을 조금 돌리고(바퀴 축 둘레로 θ) 옮겨서(t) 비전 쪽 점들에 가장 잘 겹치게 맞춘다(점과 선 사이 거리, 몇 번 반복).
    같은 앞 거리끼리 비교하지 않는 이유: 급한 곡선에서는 선이 옆으로 누워 앞 거리가 같은 점끼리 짝이 맞지 않는다.
    추정 위치가 실제보다 왼쪽으로 e, 반시계로 δ 틀렸다면 맞춘 값은 θ = δ, t의 옆 성분 = -e 이다. 앞뒤 성분은 곧은 길에서 정해지지 않아 작게 누른다.
도로 크기에 걸린 숫자는 쓰지 않는다. 거리 기준은 카메라 높이의 배수, 판정은 차로 폭의 배수다.
"""
import numpy as np

NEAR = (1.5, 8.0)    # 비교에 쓰는 앞 거리 = 카메라 높이의 이 배수 사이 (58 mm → 약 9 ~ 46 cm). 멀수록 위치 오차가 큼
MIN_N = 6            # 이보다 적은 지점으로는 추정하지 않음
WARN = 0.15          # 옆으로 차로 폭의 이 배수보다 틀리면 경고


def to_cam(P, pose, cam_fwd):
    """세계 점들 (N,2) [m] → 카메라 기준 [오른쪽 m, 앞 m]. pose = (x, y, yaw) 바퀴 축 기준."""
    x, y, yaw = pose
    h = np.array([np.cos(yaw), np.sin(yaw)])
    d = P - (np.array([x, y]) + cam_fwd * h)
    return np.c_[d @ [h[1], -h[0]], d @ h]


def check(seen, expect, cam_h, cam_fwd, lane_w, iters=4):
    """seen = {"left": 점열, "right": 점열} (비전, 카메라 기준), expect = 같은 형식(지도와 추정 위치로 계산한 선, 촘촘한 점열).
    돌려주는 값: dict(lat = 추정 위치의 옆 오차 m(왼쪽 +), yaw = 방향 오차 rad(반시계 +, 선이 하나뿐이면 None), n = 쓴 점 수, warn) 또는 None."""
    pairs = []
    for side in ("left", "right"):
        q, m = seen.get(side), expect.get(side)
        if q is None or m is None or len(q) < 2 or len(m) < 2:
            continue
        r = np.linalg.norm(q, axis=1)
        pairs.append((q[(q[:, 1] >= NEAR[0] * cam_h) & (r <= NEAR[1] * cam_h)], m))
    if sum(len(q) for q, _ in pairs) < MIN_N:
        return None
    both = sum(len(q) >= MIN_N // 2 for q, _ in pairs) == 2   # 선이 하나뿐이면 방향은 정하지 않고 옆만 낸다(하나로는 옆·방향이 서로 바꿔치기됨)
    th, t = 0.0, np.zeros(2)
    for it in range(iters):
        A, y = [], []
        for q, m in pairs:
            c, s_ = np.cos(th), np.sin(th)
            piv = np.array([0.0, -cam_fwd])                       # 바퀴 축(카메라 기준)
            mm = (m - piv) @ np.array([[c, s_], [-s_, c]]) + piv + t   # 지금까지 맞춘 만큼 돌리고 옮긴 지도 선
            k = np.linalg.norm(q[:, None] - mm[None], axis=2).argmin(axis=1)
            tg = np.gradient(mm, axis=0)[k]
            tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
            n = np.c_[tg[:, 1], -tg[:, 0]]                         # 선의 법선
            res = np.sum(n * (q - mm[k]), axis=1)
            keep = np.abs(res) <= (0.5 if it == 0 else 0.25) * lane_w   # 짝이 틀린 점은 버림
            keep &= (k > 0) & (k < len(mm) - 1)                    # 지도 선 범위 밖(끝점과 짝)은 버림: 점들이 끝점 하나에 몰리면 돌림과 옆 옮김을 구분하지 못함
            perp = np.c_[-(mm[k, 1] - piv[1]), mm[k, 0] - piv[0]]   # 바퀴 축 둘레로 반시계로 돌릴 때 움직이는 방향
            A.append(np.c_[np.sum(n * perp, axis=1), n][keep]), y.append(res[keep])
        A, y = np.vstack(A), np.concatenate(y)
        if len(y) < MIN_N:
            return None
        reg = np.sqrt(len(y)) * np.array([[0, 0, 0, 0.0], [0, 0, 0, 0], [0, 0, 1.0, 0]])[:, :3] * 0.3   # 앞뒤 옮김은 작게 누름
        M = np.r_[A, reg[2:3]] if both else np.r_[A, reg[2:3]][:, 1:]
        d = np.linalg.lstsq(M, np.r_[y, 0.0], rcond=None)[0]
        d = d if both else np.r_[0.0, d]
        th, t = th + d[0], t + d[1:]
    e = -float(t[0])
    return dict(lat=e, yaw=float(th) if both else None, n=int(len(y)), warn=bool(abs(e) > WARN * lane_w))
