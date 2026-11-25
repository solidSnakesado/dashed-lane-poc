"""출발지 → 목적지 주행을 ROS 없이 물리로 먼저 확인한다(젯슨의 ROS2 주행 goal_runner.py와 같은 계산을 씀: pinky/ros/goal_plan.py).

Route Server 대신 같은 그래프 파일(lane_graph.geojson)에서 최소 비용 길을 찾고(다익스트라), 0.05 m 간격의 경로로 만든 뒤,
경로가 지나는 정지선마다 경로를 잘라서: 정지선 앞까지 주행(RPP) → 멈춰서 WAIT초 → 다음 구간. 위치는 엔코더 오도메트리만 쓴다.
출력: 쌍마다 도착 여부, 차로 밖으로 나간 줄 수, 정지선마다 멈췄는지·앞머리에서 선까지 남은 거리. 기록 NN.npy(시간, 실제 x, y, yaw, 추정 x, y, yaw, v, w).
사용: python goal_drive.py GRAPH_DIR OUT 출발 목적지 [출발 목적지 ...]     출발·목적지 = 노드 번호 또는 x,y 또는 x,y,방향(도)
환경 변수: PINKY_PRO, LOOKFIX(고정 주시 거리 m, 기본 0.1), WAIT(정지 시간 초, 기본 1.0)
"""
import json, math, os, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE) + "/ros"]
from phys_drive import PINKY_PRO, CTRL_HZ, place, Physics, RPP, build_model
import goal_plan as G

STOP_TOL = 0.02        # 멈출 곳까지 남은 경로가 이보다 짧으면 그 구간 끝(m). 젯슨에서는 goal checker의 xy_goal_tolerance
DENSITY = 0.05         # Route Server의 path_density와 같은 경로 점 간격(m)


def route(graph_dir, nodes, a, b):
    """그래프 파일에서 a → b 최소 비용 길을 찾아 촘촘한 경로로. 값은 Route Server와 같이: 엣지에 cost가 있으면 그 값, 없으면 길이."""
    f = json.load(open(graph_dir + "/lane_graph.geojson"))["features"]
    E = [(p["startid"], p["endid"], p.get("cost")) for p in (x["properties"] for x in f) if "startid" in p]
    r, c, w = zip(*[(i, j, k if k is not None else float(np.linalg.norm(nodes.xy[i] - nodes.xy[j]))) for i, j, k in E])
    d, pred = dijkstra(csr_matrix((w, (r, c)), shape=(len(nodes.xy),) * 2), indices=a, return_predecessors=True)
    if not np.isfinite(d[b]):
        return None
    ids = [b]
    while ids[-1] != a:
        ids.append(int(pred[ids[-1]]))
    P = nodes.xy[ids[::-1]]; s = G.arc(P); q = np.r_[np.arange(0, s[-1], DENSITY), s[-1]]
    return np.c_[np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])]


def drive(ph, path, segs, over, wait):
    """구간마다 RPP로 달리고 구간 끝에서 wait초 멈춘다. 돌려주는 값: 기록 줄들, 도착 여부."""
    t0 = path[1] - path[0]; yaw0 = math.atan2(t0[1], t0[0]); place(ph, path[0, 0], path[0, 1], yaw0)
    sub, rows, n, c, s_ = round(1.0 / CTRL_HZ / ph.bridge_period), [], 0, math.cos(yaw0), math.sin(yaw0)
    def tick(cmd):
        nonlocal n
        true = ph.pose(); o = ph.odom
        est = (path[0, 0] + c * o[0] - s_ * o[1], path[0, 1] + s_ * o[0] + c * o[1], yaw0 + o[2])
        rows.append([n / CTRL_HZ, *true, *est, *(cmd or (0.0, 0.0))]); n += 1
        return est
    for i, seg in enumerate(segs):
        ctl = RPP(seg, 1.0 / CTRL_HZ, **over); limit = int((ctl.s[-1] / 0.05 + 10) * CTRL_HZ)
        for _ in range(limit):
            est = tick(None); cmd, remain = ctl.step(est, ph.twist[0]); rows[-1][7:9] = cmd
            if remain < STOP_TOL:
                break
            for _ in range(sub):
                ph.step(*cmd)
        else:
            return np.array(rows), False
        for _ in range(int((wait if i < len(segs) - 1 else 0.5) * CTRL_HZ)):   # 정지선 앞에서 멈춤(마지막은 도착 후 잠깐)
            tick((0.0, 0.0))
            for _ in range(sub):
                ph.step(0.0, 0.0)
    return np.array(rows), True


def main():
    graph_dir, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    nodes = G.Nodes(graph_dir + "/route_nodes.json"); wait = float(os.environ.get("WAIT", 1.0))
    over = dict(use_velocity_scaled_lookahead_dist=False, lookahead_dist=float(os.environ.get("LOOKFIX", 0.1)), min_lookahead_dist=0.05)
    model = build_model(PINKY_PRO + "/pinky_mujoco", out + "/model", world=os.path.dirname(HERE) + "/ros/empty_world.yaml", simple_render=True)
    lines = []
    for i, (ta, tb) in enumerate(zip(sys.argv[3::2], sys.argv[4::2])):
        a, b = nodes.pick(ta), nodes.pick(tb); path = route(graph_dir, nodes, a, b)
        if path is None:
            lines.append(f"{i} 노드 {a} → {b}: 길이 없음"); print(lines[-1]); continue
        stops = G.stops_on(path, nodes); segs = G.cut(path, [s for s, _ in stops])
        R, done = drive(Physics(model, PINKY_PRO + "/pinky_mujoco/config/mujoco.yaml"), path, segs, over, wait)
        np.save(f"{out}/{i:02d}.npy", R); np.save(f"{out}/{i:02d}_path.npy", path)
        lines.append(G.report(i, a, b, path, R, "도착" if done else "미도착", nodes, stops)); print(lines[-1], flush=True)
    open(out + "/summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
