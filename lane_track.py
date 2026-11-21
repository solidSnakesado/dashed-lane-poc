"""선을 프레임 사이에 이어서 추적하고 차선 번호와 중심선을 정한다 (6단계: 차선 번호, 7단계: 추적).
추적 중인 선은 '최근에 본 점들의 모음'이다. 프레임마다 차량 움직임만큼 점들을 옮기고(lane_motion.py),
새로 본 점과 겹치는 곳은 칼만 필터로 갱신하고, 안 보인 곳의 점은 잠시(KEEP_SEC) 남겨 둔다.
"""
from collections import Counter

import numpy as np

from lane_cross import Crossing
from lane_group import Curve
from lane_motion import along, blend_tip, common, extended, lateral, nearest, piece_ends, smooth
from lane_roles import center_line, pick

GATE, Q_RATE = 1.2, 1.0   # 같은 선으로 볼 최대 거리(m) / 1초 동안 선 위치가 변할 수 있는 정도(m²/s)
MOTION_SEC = 0.15     # 차량 움직임 추정을 고쳐 나가는 시간 상수(초)
MOTION_MAX = np.array([np.radians(40.0), 3.0, 45.0])   # 1초에 가능한 최대 회전(rad)·옆 이동(m)·앞뒤 이동(m)
R_MEAS = 0.10         # 한 번 측정의 오차(m²)
KEEP_SEC = 1.0        # 안 보이는 점을 유지하는 시간(초). KEEP_Z보다 먼 점은 거리에 반비례해 짧게
KEEP_DIST, HOLD_DIST = 8.0, 4.0   # 움직임 값을 받을 때: 안 보이는 점·내 차로 선을 이 거리(m)를 달릴 동안 유지 (점선 주기의 1배, 0.5배)
NEAR_KEEP = 1e9       # 이번에 안 보인 점은 이 거리(m)보다 가까운 것만 기억함 (핑키: 위치를 믿을 수 있는 거리)
KEEP_Z = 15.0         # 먼 점일수록 움직임 추정 오차가 크게 쌓이므로 빨리 버림
HOLD_SEC, HOLD_LEN = 1.0, 30.0   # 내 차로 선은 통째로 안 보여도 이 시간(초) 동안, 앞으로 이 거리(m)까지 이어서 유지
END_GATE, END_TURN = 0.6, 20.0   # 선 끝 너머에 있는 조각을 붙일 때의 거리(m)·방향 차(도) 한계
MIN_NEW = 2.0         # 이보다 짧고 조각 하나뿐인 선은 새 추적 대상으로 삼지 않음(m)
ROLE_SEC, ROLE_BIAS = 0.15, 0.5   # 내 차로 선으로 인정하려면 이 시간(초) 이상 보였어야 함 / 대응 찾을 때 내 차로 선에 주는 우선권(m)
MATCH_TURN = 30.0     # 추적 중인 선과 방향이 이 각(도) 넘게 다른 조각은 붙이지 않음
Z_BACK, DEFAULT_WIDTH = -3.0, 3.3   # 차량 뒤로 이만큼(m) 넘어간 점은 버림 / 한쪽 선만 보일 때 쓰는 차로 폭 초기값(m)


class Track:
    """추적 중인 선 하나."""
    def __init__(self, tid, lane):
        self.id, self.det, self.hits = tid, lane, 1
        self.pts = lane.sample(1.0)
        self.age = np.zeros(len(self.pts), int)      # 점마다 마지막으로 본 뒤 지난 프레임 수
        self.var = R_MEAS                            # 위치 불확실도(m²)
        self.kinds = Counter([lane.kind])
        self.curve, self.sm = Curve(self.pts), None  # sm = 시간 방향으로 평활한 점열
        self.tip = None                              # 평활한 끝 상태 [방향, 휘는 정도, 그릴 거리]

    @property
    def kind(self):
        return self.kinds.most_common(1)[0][0]      # 지금까지 가장 많이 판정된 종류(점선/실선)

    def move(self, rot, shift):
        """차량 움직임으로 생긴 회전·이동을 선 전체에 적용한다 (예측 단계)."""
        self.pts = self.pts @ rot.T + shift
        self.sm = None if self.sm is None else self.sm @ rot.T + shift
        self.curve = Curve(self.pts)
        if self.tip is not None:                     # 끝 방향도 차량이 돈 만큼 돌림
            self.tip = self.tip - np.array([np.arctan2(rot[1, 0], rot[0, 0]), 0.0, 0.0])

    def update(self, lanes, q, keep, hold=0, bridge=False, step=1):
        """이번 프레임에 이 선으로 대응된 선들(없으면 빈 목록)로 갱신한다. 점이 남지 않으면 False.

        hold > 0이면 내 차로 선이다: 통째로 안 보이는 동안 hold 프레임까지 유지하고 앞으로 이어 붙인다(예측선).
        bridge면 교차로 구간이다: 점을 버리지 않고 예측선을 계속 유지한다(건너편 차선에 붙은 뒤에도)."""
        self.var += q
        self.age = self.age + step                   # step = 1프레임, 또는 (움직임 값을 받을 때) 이번에 달린 거리
        if lanes:
            new = np.vstack([ln.sample(1.0) for ln in lanes])
            ref = self.curve.sample(0.25)
            idx, dist = nearest(new, ref)
            k = self.var / (self.var + R_MEAS)       # 칼만 이득: 새 측정을 믿는 정도
            self.var *= 1 - k
            # 이전 위치와 가까운 점은 칼만 갱신, 새로 보인 구간은 측정값 그대로
            seen = np.where((dist <= GATE)[:, None], ref[idx] + k * (new - ref[idx]), new)
            stale = nearest(self.pts, new)[1] > 1.5  # 이번에 본 점과 겹치지 않는 이전 점만 남김
            self.pts = np.vstack([seen, self.pts[stale]])
            self.age = np.r_[np.zeros(len(seen), int), self.age[stale]]
            self.hits += 1
            for ln in lanes:
                self.kinds[ln.kind] += 1
        self.det = max(lanes, key=lambda ln: ln.length) if lanes else None
        coast = hold > 0 and not lanes
        far = 1.0 if coast else np.maximum(np.linalg.norm(self.pts, axis=1) / KEEP_Z, 1.0)
        ok = ((self.age * far <= (hold if coast else keep)) | bridge) & (self.pts[:, 1] > Z_BACK)
        ok &= (self.age == 0) | (self.pts[:, 1] <= NEAR_KEEP) | bridge   # 멀리서만 본 점은 부정확하므로 기억하지 않음
        if ok.sum() < 2:
            return False
        self.pts, self.age = self.pts[ok], self.age[ok]
        # 점들을 선을 따라간 순서로 정렬 (이전 곡선 위의 위치 기준, 끝 너머는 떨어진 거리만큼 더함)
        ref = extended(self.curve)[0]
        idx, dist = nearest(self.pts, ref)
        key = idx + np.where(idx == 0, -dist * 4, np.where(idx == len(ref) - 1, dist * 4, 0))
        order = np.argsort(key, kind="stable")
        self.pts, self.age = self.pts[order], self.age[order]
        self.curve = Curve(self.pts)                 # 내보내기 전에 Tracker가 떨림을 줄인 곡선으로 바꿈
        short = HOLD_LEN - self.pts[-1, 1]
        if (coast or bridge) and short > 1:          # 안 보이거나 교차로 구간: 끝 방향 그대로 앞으로 이어 붙임
            more = self.curve.arc(short, 1.0, tip=self.tip)
            self.pts, self.age = np.vstack([self.pts, more]), np.r_[self.age, np.full(len(more), self.age[-1])]
            self.curve = Curve(self.pts)
        return True


class Tracker:
    """프레임마다 선 목록을 받아 추적 상태를 갱신한다."""
    def __init__(self, fps):
        self.q, self.fps, self.tracks, self.next_id, self.width = Q_RATE / fps, fps, [], 0, DEFAULT_WIDTH
        self.keep, self.min_hits, self.hold = (max(int(round(v * fps)), 1) for v in (KEEP_SEC, ROLE_SEC, HOLD_SEC))
        self.ends = (np.zeros((0, 2)), np.zeros((0, 2)))     # 직전 프레임의 조각 시작점·끝점
        self.left, self.right, self.by_odom = None, None, False   # 직전의 내 차로 선, 움직임 값을 밖에서 받는 중인지
        self.cross = Crossing(fps)                           # 교차로 구간 상태 (lane_cross.py)
        self.motion, self.seen_along = np.zeros(3), False    # 프레임당 움직임(회전, 옆, 앞뒤), 앞뒤 근거 유무
        # 이번 프레임의 어긋남을 반영하는 비율, 한 프레임에 가능한 최대 움직임
        self.gain, self.limit = 1.0 - np.exp(-1.0 / (fps * MOTION_SEC)), MOTION_MAX / fps

    def match(self, lanes):
        """새 선마다 가장 가까운 추적을 찾는다. 한 추적에 여러 선(끊겨 보인 조각들)이 붙을 수 있다."""
        refs, pairs = [extended(t.curve) for t in self.tracks], []
        for li, ln in enumerate(lanes):
            # 차량에 가까운 쪽 절반으로 비교 (먼 쪽은 차량이 조금만 돌아도 크게 어긋남)
            p = ln.sample(1.0)
            p = p[np.argsort(np.linalg.norm(p, axis=1))[:max(len(p) // 2, 3)]]
            best = None
            for ti, (ref, lo, hi) in enumerate(refs):
                idx, dist = nearest(p, ref)
                cost = float(np.median(dist))
                d, v = np.gradient(ref, axis=0)[int(np.median(idx))], ln.pts[-1] - ln.pts[0]
                cos = abs(d @ v) / max(np.linalg.norm(d) * np.linalg.norm(v), 1e-9)
                role = self.tracks[ti] is self.left or self.tracks[ti] is self.right
                rank = cost - ROLE_BIAS * role                   # 같은 선에 추적이 겹치면 내 차로 선을 먼저 고름
                if self.cross.active and role:                   # 교차로 구간: 예측선에는 건너편 차선만 붙임
                    if self.cross.accept(ln, cost, cos) and (best is None or rank < best[0]):
                        best = (rank, ti)
                    continue
                if ln.length >= 1.5 and cos < np.cos(np.radians(MATCH_TURN)):
                    continue                                     # 방향이 다른 선(가로지르는 유도선 등)
                if np.mean((idx < lo) | (idx >= hi)) > 0.5:      # 대부분 선 끝 너머: 더 엄격하게
                    cost = cost * GATE / END_GATE if cos >= np.cos(np.radians(END_TURN)) else np.inf
                if cost <= GATE and (best is None or rank < best[0]):
                    best = (rank, ti)
            if best is not None:
                pairs.append((best[1], li))
        return pairs

    def shift_all(self, th, vec):
        """추적 중인 모든 선(과 직전 조각 끝점)을 각 th(rad)만큼 돌리고 vec만큼 옮긴다."""
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for t in self.tracks:
            t.move(rot, vec)
        self.ends = tuple(e @ rot.T + vec for e in self.ends)

    def predict(self, lanes, odom=None):
        """추적 중인 선들을 차량이 움직인 만큼 옮긴다. odom(바퀴 센서로 잰 회전·이동)이 있으면 그 값만 쓴다.
        없으면 차선으로 추정: 직전까지의 움직임을 먼저 적용하고 남은 어긋남의 일부(self.gain)만 반영한다."""
        if odom is not None:
            self.motion, self.seen_along, self.ends, self.by_odom = np.r_[odom[0], odom[1]], True, piece_ends(lanes)[:2], True
            return self.shift_all(*odom)
        self.shift_all(self.motion[0], self.motion[1:])
        pairs = self.match(lanes)
        if self.left is not None or self.right is not None:
            # 내 차로 선이 정해져 있으면 그 선만 기준으로 삼음 (교차로의 유도선 같은 다른 선에 휘둘리지 않게)
            pairs = [p for p in pairs if self.tracks[p[0]] is self.left or self.tracks[p[0]] is self.right]
        rot, shift = lateral(self.tracks, lanes, pairs)
        more = along(self.ends, lanes) if pairs else None    # 조각 끝점으로 본 앞뒤 어긋남 (근거 부족이면 None)
        self.seen_along = self.seen_along or more is not None
        err = np.r_[np.arctan2(rot[1, 0], rot[0, 0]), shift[0], 0.0]
        err[1:] += more if more is not None else 0.0
        if not pairs or self.cross.active:           # 견줄 선이 없거나 교차로 구간이면 회전·옆 이동은 0쪽으로
            err[:2] = -self.motion[:2]
        # 물리적으로 가능한 범위를 벗어난 추정은 잘라 냄 (잘못된 대응이 겹쳐 값이 폭주하는 것을 막음)
        new = np.clip(self.motion + self.gain * err, -self.limit, self.limit)
        self.shift_all(new[0] - self.motion[0], new[1:] - self.motion[1:])
        self.motion = new
        self.ends = piece_ends(lanes)[:2]

    def update(self, lanes, odom=None):
        """새 프레임의 선들로 추적을 갱신한다: 움직임 반영 → 대응 → 갱신. odom = 밖에서 받은 움직임(없으면 추정)."""
        self.predict(lanes, odom)
        got = {}
        _ = [got.setdefault(ti, []).append(lanes[li]) for ti, li in self.match(lanes)]
        used = {id(ln) for v in got.values() for ln in v}   # 아래 keep: 앞뒤 이동을 모르면 안 보인 점을 남기지 않음
        keep, hold, step = (self.keep if self.seen_along else 0), self.hold, 1
        if self.by_odom:   # 움직임 값을 받으면 '시간'이 아니라 '달린 거리'로 기억을 지움 (속도와 무관, 서 있으면 유지)
            keep, hold, step = KEEP_DIST, HOLD_DIST, float(np.linalg.norm(self.motion[1:]))
        self.tracks = [t for ti, t in enumerate(self.tracks) if t.update(
            got.get(ti, []), self.q, keep, *self.cross.hold(t is self.left or t is self.right, hold), step)]
        shift = common([(t.sm, t.curve) for t in self.tracks])   # 모든 선이 함께 움직인 양은 그대로 따라감
        for t in self.tracks:                                    # 선마다 따로 흔들리는 부분만 평활
            t.sm, t.curve = smooth(t.sm, t.curve, self.fps, shift)
            t.tip = t.curve.tip = blend_tip(t.tip, t.curve, self.fps)
        for ln in (ln for ln in lanes if id(ln) not in used and (ln.length >= MIN_NEW or len(ln.pieces) > 1)):
            self.tracks.append(Track(self.next_id, ln))
            self.next_id += 1
        return self.tracks

    def roles(self):
        """내 차로의 왼쪽 선(lane1)과 오른쪽 선(lane2)을 고른다. 없으면 None."""
        self.left, self.right = pick(self.tracks, (self.left, self.right), self.min_hits, self.width)
        self.cross.step((self.left, self.right), abs(self.motion[2]))
        return self.left, self.right

    def center(self, left, right):
        """중심선 곡선과 '추정 여부'. 양쪽 선이 있으면 그 가운데, 한쪽뿐이면 차로 폭의 절반만큼 옮긴다."""
        curve, guess, self.width = center_line(left, right, self.width)
        if curve is not None:                        # 중심선의 끝 상태 = 양쪽 선 끝 상태의 평균
            curve.tip = np.mean([t.tip for t in (left, right) if t is not None], axis=0)
        return curve, guess
