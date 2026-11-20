"""교차로처럼 내 차로 선이 통째로 끊기는 구간을 건너는 처리 (직진, 예측선 방식).

  1) 진입: 추적하던 내 차로 선이 ENTER_SEC 동안 앞에서 끊겨 있으면(차량 앞 CUT_Z까지만 보이거나,
     GAP_Z보다 먼 곳에서만 보이거나, 전혀 안 보임) 교차로 구간으로 본다.
  2) 고정: 진입 직전의 선을 끝 방향 그대로 앞으로 이어 둔다(예측선). 차량 움직임만큼만 옮기고,
     유도선·횡단보도처럼 조건에 안 맞는 검출은 이 선에 붙이지 않는다.
  3) 붙이기: 예측선과 나란하고 가까운 긴 차선이 잡히면 그 선에 맞춘다.
  4) 끝: 내 차로 선이 다시 끊김 없이 보이거나, CROSS_LEN만큼 달리거나 CROSS_SEC가 지나면 구간을 끝낸다.
경로는 직진으로 가정한다. 지도를 쓸 때는 예측선 자리에 지도의 차선 좌표를 넣으면 된다.
"""
import numpy as np

ENTER_SEC = 0.3       # 내 차로 선이 이 시간(초) 동안 끊겨 있으면 교차로 구간으로 봄
CUT_Z = 12.0          # 검출된 선이 차량 앞 이 거리(m)도 못 미쳐 끝나면 '앞에서 끊김'
GAP_Z = 15.0          # 검출된 선이 차량 앞 이 거리(m)보다 먼 곳에서 시작하면 '사이가 끊김'
READY_SEC = 1.0       # 그 전에 이 시간(초) 이상 추적된 선이 있어야 함
CROSS_LEN = 60.0      # 교차로 구간으로 유지하는 최대 주행 거리(m)
CROSS_SEC = 20.0      # 교차로 구간으로 유지하는 최대 시간(초). 신호 대기처럼 서 있는 경우를 감안해 길게 둠
EXIT_LEN = 6.0        # 건너편 차선으로 인정하는 최소 길이(m). 횡단보도 줄무늬(약 4m)보다 길게
EXIT_TURN = 10.0      # 예측선과 건너편 차선의 방향 차 한계(도)
EXIT_GATE = 1.0       # 예측선과 건너편 차선의 옆 거리 한계(m). 차로 폭의 절반(약 1.6m)보다 작게
ON_GATE = 0.4         # 짧은 조각은 예측선 바로 위(이 거리 m 이내)에 있을 때만 붙임
HOLD = 10 ** 6        # 교차로 구간에서는 예측선을 시간 제한 없이 유지(구간이 끝나면 풀림)


def cut(t):
    """추적 중인 선 t가 이번 프레임에 앞에서 끊겨 보이는지."""
    return t.det is None or t.det.pts[-1, 1] < CUT_Z or t.det.pts[0, 1] > GAP_Z


class Crossing:
    """교차로 구간의 상태. Tracker가 프레임마다 step()을 불러 갱신한다."""

    def __init__(self, fps):
        self.enter, self.ready = max(int(round(ENTER_SEC * fps)), 2), int(round(READY_SEC * fps))
        self.max_frames = int(CROSS_SEC * fps)
        self.unseen = self.frames = 0               # 안 보인 프레임 수, 구간에 들어온 뒤 지난 프레임 수
        self.dist, self.active = 0.0, False         # 구간에서 달린 거리(m), 지금 교차로 구간인지

    def accept(self, lane, cost, cos):
        """교차로 구간에서 예측선에 붙여도 되는 차선인지. cost = 옆 거리(m), cos = 방향이 나란한 정도."""
        if lane.length < 1.5:                       # 아주 짧은 조각은 방향을 믿을 수 없어 거리만 봄
            return cost <= ON_GATE
        gate = EXIT_GATE if lane.length >= EXIT_LEN else ON_GATE
        return cos >= np.cos(np.radians(EXIT_TURN)) and cost <= gate

    def hold(self, is_role, base):
        """선을 안 보이는 동안 유지할 프레임 수와 교차로 구간 여부. 내 차로 선만 유지한다(base = 평소 값)."""
        return ((HOLD if self.active else base), self.active) if is_role else (0, False)

    def step(self, roles, moved):
        """내 차로 선(왼쪽, 오른쪽)의 이번 프레임 상태로 구간을 시작하거나 끝낸다. moved = 이번에 달린 거리(m)."""
        roles = [t for t in roles if t is not None]
        if not roles:
            self.unseen, self.active = 0, False
            return
        if self.active:
            self.frames, self.dist = self.frames + 1, self.dist + moved
            done = not any(cut(t) for t in roles)   # 내 차로 선이 모두 끊김 없이 보이면 다 건넌 것
            if done or self.dist > CROSS_LEN or self.frames > self.max_frames:
                self.unseen, self.active = 0, False
            return
        self.unseen = self.unseen + 1 if all(cut(t) for t in roles) else 0
        if self.unseen >= self.enter and any(t.hits >= self.ready for t in roles):
            self.active, self.frames, self.dist = True, 0, 0.0
