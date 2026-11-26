"""종이 블럭으로 쌓는 간단한 휘어진 길(실물 적용 예행용)의 도면에서 필요한 파일을 모두 만든다.

길: 직선 → 왼쪽으로 90도 휘는 굽은 길 → 직선. 치수는 블럭 수에서 나온다(블럭을 자르거나 틈을 두지 않게).
블럭: 가로 BLOCK_L, 두께 BLOCK_T, 높이 BLOCK_H. LAYERS단으로 쌓는다(라이다가 벽을 보려면 벽이 라이다보다 높아야 함).
  직선 벽 4곳은 N_STRAIGHT장씩 일렬. 굽은 구간 안쪽 벽은 N_IN장, 바깥쪽 벽은 N_OUT장을 같은 각도씩 꺾어 놓는다.
  안쪽 벽 블럭은 길 반대쪽(굽은 길의 중심 쪽) 모서리끼리 닿고 길 쪽에는 쐐기 모양 틈이 생긴다. 바깥쪽 벽 블럭은 길 쪽 모서리끼리 닿는다.
차로: 벽 사이를 LANES줄로 똑같이 나눈다(모두 같은 방향). 차로 0이 왼쪽(굽은 길의 안쪽).
  차선 변경 길: 길을 따라 LC_EVERY마다, 옆 차로로 LC_LEN에 걸쳐 부드럽게 넘어가는 점열(양쪽 방향 모두). 첫 엣지에 벌칙 LC_PENALTY를 더해
  넘어갈 이유가 없으면 고르지 않게 한다(굽은 구간은 안쪽 차로가 짧아서 벌칙이 없으면 넘어갔다 돌아오는 길이 뽑힘).
좌표(도면 좌표, m): 출발 쪽 길 가운데가 START, 진행 방향이 +x. 굽은 뒤에는 +y로 간다.
만드는 것(OUT 폴더):
  walls.sdf          시뮬 브리지에 넣는 세계(블럭 자리마다 상자 하나 + 땅속에 둔 장애물 상자). lane_sim_bridge.py --world 로 준다.
  lane_graph.geojson Route Server용 도로 그래프(노드 = 차로 가운데 점, 엣지 = 앞으로만 + 차선 변경 길)
  route_nodes.json   노드 좌표·방향·차로 경계까지 거리, 지도 그림 정보 (goal_plan.py, goal_click.py가 씀). 차선 변경 길의 점은 zone = 1.
  map.jpg            탑뷰 그림(블럭과 번호, 차로 가운데 선, 기준점)
  blocks.json        블럭 자리마다 길 쪽 면의 양 끝 좌표(쌓을 때 봄)
  refs.json          기준점(벽 끝의 길 쪽 모서리 4곳)의 도면 좌표. SLAM 지도와 맞출 때 씀.
사용: python wall_map.py OUT
"""
import json, math, os, sys
import cv2, numpy as np

BLOCK_L, BLOCK_T, BLOCK_H, LAYERS = 0.19, 0.05, 0.095, 2   # 블럭 가로, 두께, 높이(m)와 쌓는 단 수
N_STRAIGHT, N_IN, N_OUT = 4, 2, 6                          # 직선 벽 한 곳, 굽은 구간 안쪽 벽, 바깥쪽 벽의 한 단 블럭 수
LANES = 2                                                  # 차로 수
LC_EVERY, LC_LEN, LC_STEP, LC_PENALTY = 0.1, 0.5, 0.02, 0.3   # 차선 변경 길: 시작 간격, 길이, 점 간격, 벌칙(모두 길 가운데 선 기준 m)
OBSTACLE = (0.12, 0.12, 0.19)                              # 시험용 장애물 상자(앞뒤, 좌우, 높이 m). /lane_sim/obstacle로 놓고 치움
START = (0.3, 0.5)                                         # 출발 쪽 길 가운데의 도면 좌표
STEP = 0.01                                                # 차로 가운데 점 간격(길 가운데 선 기준 m)
PX_PER_M, MARGIN = 400.0, 0.25                             # 탑뷰 그림 크기(픽셀/m)와 둘레 여백(m)

S = N_STRAIGHT * BLOCK_L                                   # 직선 길이
R_BACK = BLOCK_L / (2 * math.sin(math.pi / 4 / N_IN))      # 안쪽 벽 블럭의 길 반대쪽 모서리가 놓이는 반지름
R_IN = R_BACK + BLOCK_T                                    # 직선 구간에서 안쪽 벽의 길 쪽 면까지 반지름
R_OUT = BLOCK_L / (2 * math.sin(math.pi / 4 / N_OUT))      # 바깥쪽 벽의 길 쪽 모서리가 놓이는 반지름
RC, WIDTH = (R_IN + R_OUT) / 2, R_OUT - R_IN               # 차로 가운데 반지름, 직선 구간의 벽 사이 폭
CX, CY = START[0] + S, START[1] + RC                       # 굽은 길의 중심


def on(r, a):
    """굽은 길 중심에서 반지름 r, 각도 a(0 = 굽기 시작, 90도 = 굽기 끝)인 점."""
    return np.array([CX + r * math.sin(a), CY - r * math.cos(a)])


def blocks():
    """블럭 자리 목록. 자리마다 (벽 이름, 길 쪽 면의 시작 점, 끝 점, 길 반대쪽을 가리키는 단위 벡터)."""
    out = []
    for name, r, n, away in (("L", R_IN, N_IN, +1), ("R", R_OUT, N_OUT, -1)):          # 왼쪽 = 안쪽 벽, 오른쪽 = 바깥쪽 벽
        xs = [START[0] + i * BLOCK_L for i in range(N_STRAIGHT + 1)]
        seg = [(np.array([a, CY - r]), np.array([b, CY - r])) for a, b in zip(xs[:-1], xs[1:])]
        ang = np.linspace(0, math.pi / 2, n + 1)
        for a, b in zip(ang[:-1], ang[1:]):
            if away > 0:                                                                # 안쪽: 길 반대쪽 모서리가 닿게 놓고 길 쪽 면은 BLOCK_T만큼 바깥
                m = on(1.0, (a + b) / 2) - np.array([CX, CY]); seg.append((on(R_BACK, a) + m * BLOCK_T, on(R_BACK, b) + m * BLOCK_T))
            else:
                seg.append((on(r, a), on(r, b)))
        ys = [CY + i * BLOCK_L for i in range(N_STRAIGHT + 1)]
        seg += [(np.array([CX + r, a]), np.array([CX + r, b])) for a, b in zip(ys[:-1], ys[1:])]
        for p, q in seg:
            d = (q - p) / np.linalg.norm(q - p); out.append((name, p, q, away * np.array([-d[1], d[0]])))
    return out


TOTAL = 2 * S + RC * math.pi / 2                           # 길 가운데 선의 길이


def point(u, d):
    """길 가운데 선을 따라 u만큼 간 곳에서 왼쪽으로 d만큼 떨어진 점."""
    if u <= S:
        return np.array([START[0] + u, START[1] + d])
    if u <= S + RC * math.pi / 2:
        return on(RC - d, (u - S) / RC)
    return np.array([CX + RC - d, CY + u - S - RC * math.pi / 2])


def graph():
    """노드(좌표, 방향, zone)와 엣지 (시작, 끝, 고정 비용 또는 None, 차선 변경 여부). 차로 k의 i번째 점은 노드 k*n + i."""
    U = np.arange(0.0, TOTAL + 1e-9, STEP); n = len(U); lane_w = WIDTH / LANES
    D = [WIDTH / 2 - lane_w * (k + 0.5) for k in range(LANES)]             # 차로 가운데의 옆 위치(왼쪽 +)
    head = lambda u: float(np.clip((u - S) / RC, 0.0, math.pi / 2))
    P = [point(u, d) for d in D for u in U]; yaw = [head(u) for d in D for u in U]; zone = [0] * (n * LANES)
    E = [(k * n + i, k * n + i + 1, None, False) for k in range(LANES) for i in range(n - 1)]
    m = int(round(LC_LEN / STEP)); T = np.arange(LC_STEP, LC_LEN - 1e-9, LC_STEP) / LC_LEN
    for i0 in range(0, n - m, int(round(LC_EVERY / STEP))):
        for a in range(LANES):
            for b in (a - 1, a + 1):
                if not 0 <= b < LANES:
                    continue
                Q = [point(U[i0] + t * LC_LEN, D[a] + (D[b] - D[a]) * (3 * t * t - 2 * t ** 3)) for t in T]   # 옆 위치를 부드럽게(양 끝 기울기 0)
                ids = [a * n + i0] + list(range(len(P), len(P) + len(Q))) + [b * n + i0 + m]
                full = [P[ids[0]]] + Q + [P[ids[-1]]]
                for q0, q1 in zip(full[:-2], full[2:]):
                    yaw.append(math.atan2(q1[1] - q0[1], q1[0] - q0[0]))
                P += Q; zone += [1] * len(Q)
                E += [(ids[0], ids[1], float(np.linalg.norm(full[1] - full[0])) + LC_PENALTY, True)] + [(p, q, None, True) for p, q in zip(ids[1:-1], ids[2:])]
    return np.array(P), np.array(yaw), zone, E, n, lane_w


def sdf(B):
    """블럭 자리마다 상자 하나(높이는 쌓은 단 전체)를 놓은 SDF 세계."""
    H = BLOCK_H * LAYERS; out = ['<?xml version="1.0"?>', '<sdf version="1.8"><world name="wall_map">']
    for i, (name, p, q, n) in enumerate(B):
        c = (p + q) / 2 + n * BLOCK_T / 2; ang = math.atan2(q[1] - p[1], q[0] - p[0])
        out.append(f'<model name="wall_{name}_{i}"><static>true</static><pose>{c[0]:.4f} {c[1]:.4f} {H / 2:.4f} 0 0 {ang:.5f}</pose>'
                   f'<link name="link"><collision name="c"><geometry><box><size>{BLOCK_L} {BLOCK_T} {H}</size></box></geometry></collision></link></model>')
    out.append(f'<model name="obstacle"><static>true</static><pose>0 0 -1 0 0 0</pose><link name="link"><collision name="c"><geometry><box>'
               f'<size>{OBSTACLE[0]} {OBSTACLE[1]} {OBSTACLE[2]}</size></box></geometry></collision></link></model>')   # 땅속에 둔 장애물(브리지가 옮김)
    return "\n".join(out + ["</world></sdf>"]) + "\n"


def draw(out, B, lanes, refs):
    """탑뷰 그림. 도면 좌표의 (0, 0)이 왼쪽 아래."""
    pts = np.array([v for _, p, q, n in B for v in (p, q, p + n * BLOCK_T, q + n * BLOCK_T)])
    hi = pts.max(0) + MARGIN; H, W = int(math.ceil(hi[1] * PX_PER_M)), int(math.ceil(hi[0] * PX_PER_M))
    px = lambda q: np.c_[np.atleast_2d(q)[:, 0] * PX_PER_M, H - np.atleast_2d(q)[:, 1] * PX_PER_M].astype(np.int32)
    img = np.full((H, W, 3), 40, np.uint8)
    for P in lanes:
        cv2.polylines(img, [px(P)], False, (120, 120, 120), 1, cv2.LINE_AA)
    for i, (name, p, q, n) in enumerate(B):
        box = px(np.array([p, q, q + n * BLOCK_T, p + n * BLOCK_T]))
        cv2.fillPoly(img, [box], (200, 200, 200)); cv2.polylines(img, [box], True, (60, 60, 60), 1, cv2.LINE_AA)
        u, v = px((p + q) / 2 + n * (BLOCK_T + 0.05))[0]; cv2.putText(img, str(i), (int(u) - 8, int(v) + 5), 0, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    for name, q in refs.items():
        u, v = px(q)[0]; cv2.circle(img, (int(u), int(v)), 5, (0, 200, 255), -1)
        cv2.putText(img, name, (int(u) + 8, int(v) - 8), 0, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out + "/map.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92]); return W, H


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    B = blocks(); P, yaw, zone, E, n, lane_w = graph(); N = len(P); r4 = lambda v: [round(float(v[0]), 4), round(float(v[1]), 4)]
    assert N < 65535, "노드가 uint16 범위를 넘음"
    open(out + "/walls.sdf", "w").write(sdf(B))
    feats = [{"type": "Feature", "properties": {"id": i, "frame": "map"}, "geometry": {"type": "Point", "coordinates": r4(p)}} for i, p in enumerate(P)]
    for k, (a, b, cost, lc) in enumerate(E):
        props = {"id": N + k, "startid": a, "endid": b, "metadata": {"lane_vision": not lc, **({"lane_change": True} if lc else {})}}
        if cost is not None:
            props.update(cost=round(cost, 6), overridable=False)
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "MultiLineString", "coordinates": [[r4(P[a]), r4(P[b])]]}})
    json.dump({"type": "FeatureCollection", "name": "wall_map", "features": feats}, open(out + "/lane_graph.geojson", "w"))
    L = [b for b in B if b[0] == "L"]; R = [b for b in B if b[0] == "R"]
    refs = {"left_start": L[0][1], "left_end": L[-1][2], "right_start": R[0][1], "right_end": R[-1][2]}
    W, H = draw(out, B, [P[k * n:(k + 1) * n] for k in range(LANES)], refs)
    json.dump({k: r4(v) for k, v in refs.items()}, open(out + "/refs.json", "w"), indent=1)
    json.dump([{"no": i, "wall": name, "face_from": r4(p), "face_to": r4(q), "away": r4(nv)} for i, (name, p, q, nv) in enumerate(B)],
              open(out + "/blocks.json", "w"), indent=1)
    half = lane_w / 2
    json.dump({"map": {"file": "map.jpg", "px_per_m": PX_PER_M, "height": H},
               "routes": [{"name": f"lane{k}", "kind": "left", "start": k * n, "goal": (k + 1) * n - 1, "half": half} for k in range(LANES)],
               "xy": P.round(4).tolist(), "yaw": np.round(yaw, 5).tolist(), "zone": zone, "wl": [half] * N, "wr": [half] * N, "stops": []},
              open(out + "/route_nodes.json", "w"))
    gap = 2 * BLOCK_T * math.tan(math.pi / 4 / N_IN); lc = sum(1 for e in E if e[2] is not None)
    print(f"길: 직선 {S:.2f} m + 굽은 길 + 직선 {S:.2f} m, 벽 사이 폭 {WIDTH:.3f} m, 차로 {LANES}줄(폭 {lane_w:.4f} m), 길 가운데 선 길이 {TOTAL:.2f} m")
    print(f"반지름: 안쪽 벽 길 쪽 면 {R_IN:.3f}, 바깥쪽 벽 길 쪽 면 {R_OUT:.3f} m / 안쪽 벽 굽은 구간의 길 쪽 틈 {gap * 100:.1f} cm")
    print(f"블럭: 한 단 {len(B)}장(안쪽 벽 {len(L)}, 바깥쪽 벽 {len(R)}) x {LAYERS}단 = {len(B) * LAYERS}장, 벽 높이 {BLOCK_H * LAYERS:.3f} m")
    print(f"노드 {N}(차로마다 {n}), 엣지 {len(E)}, 차선 변경 길 {lc}개(길이 {LC_LEN} m, {LC_EVERY} m마다) / 그림 {W}x{H}")
    print("차로 시작·끝 노드:", [(k * n, (k + 1) * n - 1) for k in range(LANES)])
    print("기준점(도면 좌표 m):", {k: [round(float(v[0]), 3), round(float(v[1]), 3)] for k, v in refs.items()})
    for k in range(LANES):
        print("차로 %d 출발 자세: x %.3f, y %.3f, 방향 0도" % (k, P[k * n][0], P[k * n][1]))


if __name__ == "__main__":
    main()
