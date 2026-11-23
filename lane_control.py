"""중심선을 받아 속도·회전 명령을 만든다 (순수 추종, pure pursuit). 시뮬 주행과 실제 핑키에서 함께 쓴다."""
import numpy as np


def pursuit(center, look, speed, w_max, cam_fwd=0.0):
    """중심선 위에서 look(m)만큼 떨어진 점을 향해 도는 명령 (전진 속도 m/s, 회전 속도 rad/s: 왼쪽 +). 못 만들면 None.

    center = 점열 (N,2) [오른쪽 m, 앞 m] (카메라 기준), 가는 순서. cam_fwd = 바퀴 축에서 카메라까지 앞쪽 거리(m).
    목표점은 '앞으로 look인 곳'이 아니라 '로봇에서 가장 가까운 곳부터 선을 따라가다 처음으로 look만큼 멀어지는 곳'이다.
    (교차로에서 돌 때는 갈 길이 옆으로 누워서 앞 거리로는 목표를 정할 수 없다.)
    중심선이 look까지 닿지 않으면 가장 먼 점을 쓰되, look의 절반에도 못 미치면 명령을 만들지 않는다."""
    if center is None or len(center) < 2:
        return None
    d = np.linalg.norm(center, axis=1)
    k = int(np.argmin(d))
    far = np.nonzero(d[k:] >= look)[0]
    if len(far) and k + far[0] > 0:
        j = k + far[0]
        tgt = center[j - 1] + (look - d[j - 1]) / (d[j] - d[j - 1] + 1e-9) * (center[j] - center[j - 1])
    else:
        j = 0 if len(far) else k + int(np.argmax(d[k:]))
        if d[j] < look / 2:
            return None
        tgt = center[j]
    x, z = float(tgt[0]), float(tgt[1]) + cam_fwd      # 회전 중심은 바퀴 축
    curv = 2.0 * x / (x * x + z * z)                   # 목표점을 지나는 원호의 굽은 정도 (오른쪽 +)
    return speed, float(np.clip(-speed * curv, -w_max, w_max))
