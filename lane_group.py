"""차선 조각을 선 단위로 묶고 곡선을 맞춘다 (4단계: 조각 묶기, 5단계: 곡선 맞춤).

묶기 원리(방향 추적): 조각 A의 끝에서 조각 B의 시작으로 이어지는 연결선을 그었을 때,
같은 선 위의 두 조각이라면 'A의 진행 방향 → 연결선'으로 꺾인 각과 '연결선 → B의 진행 방향'으로
꺾인 각이 같은 쪽으로 비슷한 크기여야 한다 (직선이면 둘 다 0, 원호면 둘이 같음).
옆 차선 조각은 두 각의 부호가 반대로 나오므로 걸러진다.
"""
import numpy as np

from lane_curve import Curve
from lane_geom import Piece

MAX_GAP = 15.0      # 이어 붙일 최대 빈 구간(m). 점선 조각 하나가 가려진 경우까지 허용
MAX_BEND = 40.0     # 한 번 이을 때 양 끝 각각에 허용하는 최대 꺾임 각(도)
BASE_TOL = 6.0      # 두 꺾임 각의 차이 허용값 기본(도)
DIR_NOISE = 0.2     # 조각 끝점의 옆 위치 오차 가정(m). 짧은 조각일수록 방향 오차가 커짐
SHORT_BEND = 15.0   # 방향을 못 믿는 조각이 낀 연결에 허용하는 꺾임 각(도)
SHORT_GAP = 3.0     # 둘 다 짧은 조각이면 이 거리(m) 안에서만 이음
SIDE_TOL = 1.0      # 방향을 못 믿는 조각은 연장선에서 옆으로 이 거리(m) 안에 있어야 이음
SOLID_LEN = 8.0     # 이보다 긴 조각(m)이 있으면 실선으로 판정
CHAIN_TRUST = 2.5   # 이어 붙인 길이가 이 이상(m)이면 짧은 조각들의 묶음이라도 방향을 믿음
MAX_RMS = 0.5       # 맞춘 곡선과 점들의 차이가 이보다 크면(m) 잘못 묶인 선으로 보고 버림


def _angle(u, v):
    """벡터 u에서 v로 도는 각(도, 반시계 +)."""
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)))


def _end(piece, tail):
    """조각의 한쪽 끝점과, 그 끝에서 바깥으로 나가는 방향."""
    if tail:
        return piece.end, piece.direction(True)
    return piece.start, -piece.direction(False)


def link_cost(a, ta, b, tb):
    """조각 a의 ta쪽 끝과 b의 tb쪽 끝을 잇는 비용. 이을 수 없으면 None."""
    (pa, ua), (pb, ub) = _end(a, ta), _end(b, tb)
    c = pb - pa
    dist = float(np.linalg.norm(c))
    if dist < 1e-6 or dist > MAX_GAP:
        return None
    c = c / dist
    alpha, beta = _angle(ua, c), _angle(c, -ub)     # a에서 나가며 꺾인 각, b로 들어가며 꺾인 각
    ok_a, ok_b = a.trust, b.trust
    # 방향을 못 믿는 조각이 낀 연결은 조건을 좁히고, 양쪽 다 믿는 연결보다 뒤에 확정되게 비용을 높임
    if not ok_a and not ok_b:                       # 둘 다 못 믿으면 아주 가까울 때만
        return 2.0 + dist / MAX_GAP if dist <= SHORT_GAP else None
    if not (ok_a and ok_b):                         # 한쪽만 믿을 수 있으면 그쪽 방향의 연장선 근처일 때만
        ang = alpha if ok_a else beta
        if abs(ang) > SHORT_BEND or dist * abs(np.sin(np.radians(ang))) > SIDE_TOL:
            return None
        return 1.0 + abs(ang) / SHORT_BEND + dist / MAX_GAP
    if max(abs(alpha), abs(beta)) > MAX_BEND:
        return None
    tol = BASE_TOL + np.degrees(np.arctan(DIR_NOISE / a.length) + np.arctan(DIR_NOISE / b.length))
    if abs(alpha - beta) > tol:
        return None
    return abs(alpha - beta) / tol + dist / MAX_GAP


def group(pieces):
    """조각 목록을 선(조각의 순서 있는 목록)들로 묶는다. 비용이 낮은 연결부터 확정한다."""
    n = len(pieces)
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            for ti in (True, False):
                for tj in (True, False):
                    c = link_cost(pieces[i], ti, pieces[j], tj)
                    if c is not None:
                        cands.append((c, i, ti, j, tj))
    parent = list(range(n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    links = {}                                      # (조각 번호, 끝) → 이어진 (조각 번호, 끝)
    for _, i, ti, j, tj in sorted(cands):
        # 끝 하나에는 연결 하나만, 고리가 생기는 연결은 금지
        if (i, ti) in links or (j, tj) in links or find(i) == find(j):
            continue
        links[(i, ti)], links[(j, tj)] = (j, tj), (i, ti)
        parent[find(i)] = find(j)

    chains, seen = [], set()
    for i in range(n):
        free = [t for t in (False, True) if (i, t) not in links]
        if i in seen or not free:
            continue                                # 선의 끝에 있는 조각에서만 출발
        chain, cur, out = [], i, not free[0]
        while True:
            seen.add(cur)
            chain.append(pieces[cur] if out else pieces[cur].reversed())
            nxt = links.get((cur, out))
            if nxt is None:
                break
            cur, out = nxt[0], not nxt[1]
        # 차량에 가까운 쪽이 먼저 오게 뒤집음
        if np.linalg.norm(chain[0].start) > np.linalg.norm(chain[-1].end):
            chain = [p.reversed() for p in chain[::-1]]
        chains.append(chain)
    return chains


def merged(chain):
    """묶인 조각들을 긴 조각 하나처럼 만든다. 짧은 조각들도 묶이면 끝 방향을 믿을 수 있게 된다."""
    line = np.vstack([p.line for p in chain])
    length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    trust = length >= CHAIN_TRUST or (len(chain) == 1 and chain[0].trust)
    return Piece(chain[0].poly, line, length, float(np.median([p.width for p in chain])), trust, list(chain))


def group_all(pieces):
    """조각을 묶고, 묶인 선끼리 다시 묶기를 되풀이한다 (더 이어지지 않을 때까지, 최대 3번)."""
    chains = group(pieces)
    for _ in range(3):
        again = group([merged(c) for c in chains])
        if len(again) == len(chains):
            break
        chains = [[p for m in c for p in m.parts] for c in again]
    return chains


class Lane(Curve):
    """한 프레임에서 묶인 선 하나: 순서 있는 조각들과 거기에 맞춘 곡선."""

    def __init__(self, chain):
        super().__init__(np.vstack([p.line for p in chain]))
        self.pieces = chain
        paint = sum(p.length for p in chain)
        solid = max(p.length for p in chain) >= SOLID_LEN or (self.length > 6 and paint / self.length > 0.85)
        self.kind = "solid" if solid else "dashed"      # 실선 또는 점선


def build_lanes(pieces):
    """조각 목록 → 선 목록. 차량 쪽 끝점의 옆 위치 순(왼쪽부터)으로 정렬한다."""
    lanes = [ln for ln in (Lane(c) for c in group_all(pieces)) if ln.rms <= MAX_RMS]
    return sorted(lanes, key=lambda ln: ln.pts[0, 0])
