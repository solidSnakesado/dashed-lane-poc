"""중심선을 받아 속도·회전 명령을 만든다 (순수 추종, pure pursuit). 시뮬 주행과 실제 핑키에서 함께 쓴다."""
import numpy as np


def pursuit(center, look, speed, w_max, cam_fwd=0.0):
    """중심선 위에서 앞 look(m) 거리의 점을 향해 도는 명령 (전진 속도 m/s, 회전 속도 rad/s: 왼쪽 +). 못 만들면 None.

    center = 점열 (N,2) [오른쪽 m, 앞 m] (카메라 기준), cam_fwd = 바퀴 축에서 카메라까지 앞쪽 거리(m).
    중심선이 look까지 닿지 않으면 가장 먼 점을 쓰되, look의 절반에도 못 미치면 명령을 만들지 않는다."""
    if center is None or len(center) < 2:
        return None
    p = center[np.argsort(center[:, 1])]
    if p[-1, 1] < look / 2:
        return None
    z = min(look, p[-1, 1])
    x = float(np.interp(z, p[:, 1], p[:, 0]))
    z += cam_fwd                                   # 회전 중심은 바퀴 축
    curv = 2.0 * x / (x * x + z * z)               # 목표점을 지나는 원호의 굽은 정도 (오른쪽 +)
    return speed, float(np.clip(-speed * curv, -w_max, w_max))
