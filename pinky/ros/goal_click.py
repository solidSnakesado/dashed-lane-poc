"""지도에서 출발지와 목적지를 찍어 시뮬 주행을 시킨다(ROS2). 로봇 쪽 브리지와 Route Server + Controller Server가 떠 있는 상태에서 실행한다.

창의 왼쪽은 탑뷰(지도, route_export.py가 만든 map.jpg), 오른쪽은 핑키 카메라 화면이다.
  탑뷰를 왼쪽 클릭 1번 = 출발지, 2번 = 목적지. 찍은 곳에서 가장 가까운 교차로 밖 차로 위 점으로 맞춰지고, 그 차로의 진행 방향이 화살표로 보인다.
  목적지를 찍으면 예상 경로(흰 선)와 정지선에서 멈출 곳(주황 점)이 그려진다. 예상 경로는 같은 그래프에서 미리 계산한 것이고, 실제 경로는 Route Server가 준다.
  Enter = 주행 시작, r = 다시 찍기, q 또는 Esc = 끝내기.
주행 중: 탑뷰에 로봇(실제 크기)과 달린 자취(분홍), 지금 차로의 주행 라인(하늘색)이 그려지고, 카메라 화면에도 주행 라인(하늘색)이 겹쳐진다.
주행은 goal_runner.py와 같다(정지선마다 멈췄다가 출발, 위치는 오도메트리). 끝나면 결과 한 줄을 터미널에 낸다.
OUT에 남는 것: 기록 NN.npy·NN_path.npy, 마지막 화면 NN.jpg, 주행 영상 NN.mp4(ffmpeg가 없으면 NN.avi).
사용(젯슨, ~/venvs/lane 파이썬, ROS2를 source한 상태): python goal_click.py GRAPH_DIR OUT [ROAD_DIR]
  ROAD_DIR(floor.png, inst.png, road.json이 든 지도 폴더)을 주면 카메라 화면·주행 라인·영상 기록이 켜진다. 안 주면 탑뷰만.
ROS 없이 저장된 기록으로 영상만 다시 만들기: python goal_click.py GRAPH_DIR OUT ROAD_DIR --replay 기록.npy 경로.npy
환경 변수 WAIT(정지 시간 초, 기본 1.0)
"""
import math, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import goal_plan as G
import goal_view as V

WIN = "lane drive: click START, GOAL on the map / Enter=go  r=reset  q=quit"
FAR = 1.0              # 찍은 곳이 가장 가까운 차로 점에서 차로 폭의 이 배수보다 멀면 받지 않는다
GREEN, RED, ORANGE, PINK, WHITE = (0, 220, 0), (0, 0, 255), (0, 160, 255), (255, 0, 255), (255, 255, 255)


class View:
    """탑뷰(지도) 위에 출발지·목적지·경로·자취·로봇·주행 라인을 그리고, 카메라 화면과 한 그림으로 묶는다. 좌표: 픽셀 = (x·k, 높이 − y·k)."""
    def __init__(self, graph_dir, nodes, dv=None):
        self.base = cv2.imread(os.path.join(graph_dir, nodes.map["file"])); self.k, self.h, self.nodes, self.dv = nodes.map["px_per_m"], nodes.map["height"], nodes, dv
        self.scale = 1.0; self.clear()

    def clear(self):
        self.start = self.goal = self.plan = self.path = self.true = self.cam = None
        self.stops, self.track, self.lane, self.note, self.info = [], [], (None, None), "click START on the map", []

    def px(self, xy):
        xy = np.atleast_2d(xy); return np.c_[xy[:, 0] * self.k, self.h - xy[:, 1] * self.k].astype(np.int32)

    def mark(self, img, node, color, label):
        p = self.px(self.nodes.xy[node])[0]; a = self.nodes.yaw[node]
        tip = (int(p[0] + 34 * math.cos(a)), int(p[1] - 34 * math.sin(a)))
        cv2.circle(img, tuple(map(int, p)), 9, color, -1); cv2.arrowedLine(img, tuple(map(int, p)), tip, color, 3, tipLength=0.4)
        cv2.putText(img, label, (int(p[0]) + 12, int(p[1]) - 12), 0, 0.6, (0, 0, 0), 4); cv2.putText(img, label, (int(p[0]) + 12, int(p[1]) - 12), 0, 0.6, color, 2)

    def top(self):
        img = self.base.copy(); line = self.path if self.path is not None else self.plan
        if line is not None:
            cv2.polylines(img, [self.px(line)], False, WHITE, 1, cv2.LINE_AA); s = G.arc(line)
            for at, _ in self.stops:                                     # 정지선 앞에서 멈출 곳
                cv2.circle(img, tuple(map(int, self.px([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])[0])), 6, ORANGE, -1)
        if len(self.track) > 1:
            cv2.polylines(img, [self.px(np.array(self.track))], False, PINK, 2, cv2.LINE_AA)
        for ln in self.lane:                                             # 주행 라인(하늘색)
            if ln is not None:
                cv2.polylines(img, [self.px(ln)], False, V.SKY, 4, cv2.LINE_AA)
        if self.start is not None:
            self.mark(img, self.start, GREEN, "START")
        if self.goal is not None:
            self.mark(img, self.goal, RED, "GOAL")
        if self.true is not None:
            V.robot(img, self.px, self.true)
        return img

    def draw(self):
        img, self.scale = V.compose(self.top(), self.cam, [self.note] + self.info)
        return img

    def click(self, u, v, graph):
        """찍은 곳(창의 픽셀)을 노드로 맞춘다. 출발지 → 목적지 순서. 목적지를 찍으면 예상 경로를 계산한다."""
        u, v = u / self.scale, v / self.scale
        if self.goal is not None or u >= self.base.shape[1]:
            return
        x, y = u / self.k, (self.h - v) / self.k
        try:
            k = self.nodes.pick("%f,%f" % (x, y))
        except ValueError:
            k = None
        if k is None or np.hypot(*(self.nodes.xy[k] - [x, y])) > FAR * (self.nodes.wl[k] + self.nodes.wr[k]):
            self.note = "no lane near the click"; return                 # 차로에서 먼 곳(도로 밖)을 찍음
        if self.start is None:
            self.start, self.note = k, "click GOAL on the map"; return
        self.goal = k; self.plan = graph.route(self.start, self.goal)
        if self.plan is None:
            self.note = "no route: r=reset"; return
        self.stops = G.stops_on(self.plan, self.nodes)
        self.note = f"route {G.arc(self.plan)[-1]:.2f} m, stop lines {len(self.stops)}: Enter=go  r=reset"

    def step(self, t, true, est):
        """주행 중 한 프레임: 로봇 위치, 주행 라인, 카메라 화면을 고치고 묶은 그림을 돌려준다(영상에도 기록)."""
        self.true = tuple(true); self.info = [f"t {t:6.1f} s"]
        if self.dv is not None:
            out, left, right = self.dv.lane(tuple(est)); self.lane = (left, right)
            self.cam = self.dv.camera(tuple(true), tuple(est), left, right)
            self.info += [f"lane state: {out['state']}", f"offset {out['off'] * 1000:+.0f} mm   left {out['left'] * 1000:.0f}   right {out['right'] * 1000:.0f}"]
        img = self.draw()
        if self.dv is not None:
            self.dv.write(img)
        return img


def replay(view, nodes, out, rec, path):
    """ROS 없이: 저장된 기록(시간, 실제 위치, 추정 위치)으로 같은 화면을 만들어 영상으로 남긴다."""
    R, view.path = np.load(rec), np.load(path); view.stops = G.stops_on(view.path, nodes); view.dv.set_route(view.path); view.note = "replay"
    view.start = int(nodes.tree.query(view.path[0])[1]); view.goal = int(nodes.tree.query(view.path[-1])[1]); last = -1.0
    view.dv.record(out + "/replay", view.draw().shape[1::-1])
    for row in R:
        view.track.append(row[1:3].copy())
        if row[0] - last >= 1.0 / V.FPS:
            last = row[0]; img = view.step(row[0], row[1:4], row[4:7])
    cv2.imwrite(out + "/replay.jpg", img); print("영상:", view.dv.stop(), "/ 마지막 화면:", out + "/replay.jpg")


def main():
    graph_dir, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    road = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
    nodes = G.Nodes(graph_dir + "/route_nodes.json"); dv = V.DriveView(road, out) if road else None
    view = View(graph_dir, nodes, dv)
    if "--replay" in sys.argv:
        i = sys.argv.index("--replay"); return replay(view, nodes, out, sys.argv[i + 1], sys.argv[i + 2])
    import rclpy
    from route_runner import Runner
    from goal_runner import drive_pair
    graph = G.Graph(graph_dir + "/lane_graph.geojson", nodes); wait, last = float(os.environ.get("WAIT", 1.0)), [-1.0]

    class Live(Runner):                                                  # 주행 중 실제 위치가 올 때마다 자취를 쌓고, 시뮬 시간 0.1초마다 화면을 고친다
        path = None
        def on_truth(self, m):
            super().on_truth(m)
            if self.rows is None or self.odom is None or not self.rows:
                return
            view.track.append(self.truth[:2].copy())
            if self.path is not None and view.path is None:
                view.path = self.path; view.stops = G.stops_on(self.path, nodes)
                if dv is not None:
                    dv.set_route(self.path)
            t = self.rows[-1][0] - self.rows[0][0]
            if view.path is not None and t - last[0] >= 1.0 / V.FPS:
                last[0] = t; cv2.imshow(WIN, view.step(t, self.truth, self.odom)); cv2.waitKey(1)

    rclpy.init(); n = Live()
    n.route_ac.wait_for_server(); n.follow_ac.wait_for_server(); n.map_odom(); n.wait(1.0)
    first = view.draw(); cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, first.shape[1], first.shape[0])
    cv2.setMouseCallback(WIN, lambda e, u, v, f, p: view.click(u, v, graph) if e == cv2.EVENT_LBUTTONDOWN else None)
    print("탑뷰에서 출발지, 목적지를 차례로 클릭하고 Enter를 누르세요. r = 다시 찍기, q = 끝내기"); count = 0
    while True:
        cv2.imshow(WIN, view.draw()); key = cv2.waitKey(30) & 0xFF
        for _ in range(20):                                              # 찍는 동안에도 메시지를 받아 둔다(시뮬 시계와 위치가 밀리지 않게)
            rclpy.spin_once(n, timeout_sec=0)
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            view.clear()
        if key in (13, 10) and view.plan is not None and view.path is None:
            view.note, n.path, last[0] = "driving ...", None, -1.0; cv2.imshow(WIN, view.draw()); cv2.waitKey(1)
            if dv is not None:
                dv.record(f"{out}/{count:02d}", first.shape[1::-1])
            path, stops, R, done = drive_pair(n, nodes, view.start, view.goal, wait)
            video = dv.stop() if dv is not None else None
            if R is None:
                print(f"노드 {view.start} → {view.goal}: {done}"); view.note = "failed: r=reset"; continue
            view.path, view.stops = path, stops
            np.save(f"{out}/{count:02d}.npy", R); np.save(f"{out}/{count:02d}_path.npy", path)
            print(G.report(count, view.start, view.goal, path, R, done, nodes, stops) + (f" | 영상 {video}" if video else ""), flush=True)
            view.note = ("arrived" if done == "도착" else "not arrived") + ": r=new  q=quit"
            cv2.imwrite(f"{out}/{count:02d}.jpg", view.draw()); count += 1
    cv2.destroyAllWindows(); n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
