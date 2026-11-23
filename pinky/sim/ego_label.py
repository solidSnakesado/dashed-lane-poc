"""내 차로 선 정답 만들기. 데이터 생성(gen_dataset.py)과 시뮬 주행의 정답 모드(sim_closed.py)가 함께 쓴다.

정답은 로봇의 위치와 방향만으로 정해진다(어느 경로로 달리는 중인지는 쓰지 않는다. 사진에는 그 정보가 없기 때문).
  1) 로봇이 지금 올라타 있다고 볼 수 있는 경로를 모두 찾는다(경로에서 옆으로 lat 이내, 방향 차 yaw 이내).
     좌회전 경로의 교차로 안 원호에서는 이 한계를 넓힌다(옆으로 차로 폭의 TURN_LW배, 방향 TURN_YAW도). 돌 때는 제어만으로도 그만큼 벗어나기 때문이다.
  2) ego_*   = 그 경로들 가운데 직진·일반 도로 경로를 따라가는 선(왼쪽·오른쪽·중심). 교차로를 가로질러 건너편 차로까지 이어진다.
               좌회전 경로는 교차로 앞 차로 구간만 ego_*이고, 직진 경로가 하나도 없을 때(돌고 있는 중)는 돌아 나갈 차로가 ego_*이다.
  3) lturn_lane = 그 경로들 가운데 좌회전 경로의 교차로 안 원호 부분. 가는 선이 아니라 차로 폭만큼의 면(왼쪽 경계와 오른쪽 경계 사이)으로 그린다.
               빈 바닥 위의 가는 선은 배워지지 않았기 때문이다(pinky_v6, pinky_v7). 이미 돌아 나온 뒤에는 그리지 않는다.
칠해진 차로의 선은 어떻게 들어왔든 항상 ego_*이다. 교차로 안의 선은 바닥에 칠해지지 않은 가상 선이다.
바닥 그림은 비트 묶음이다(비트 k = MINE[k]). 겹치는 선도 각자 온전히 남는다.
"""
import cv2
import numpy as np

from gen_roads import offset, strip

EGO = ["ego_left", "ego_right", "ego_center"]
TURN = ["lturn_lane"]
MINE = EGO + TURN
MIN_AREA = 25        # 이보다 작은 조각(픽셀)은 라벨에서 뺌
STEP = 4.0           # 경로를 다시 찍는 간격(mm)
TURN_LW, TURN_YAW = 0.4, 30.0   # 좌회전 원호 위로 보는 한계: 옆으로 차로 폭의 이 배수, 방향 차(도). 사진을 찍을 때도 이 범위로 흔듦


def route(info, pi):
    """경로 pi의 종류("lane" 일반 도로, "straight" 직진, "left" 좌회전), 교차로에 들어가는 점·나오는 점 번호, 같은 차로에서 갈라지는 다른 경로."""
    if info["type"] != "cross":
        return "lane", None, None, None
    p = np.asarray(info["paths"][pi], float)
    c, half = np.array(info["size_mm"]) / 2, info["lanes_per_dir"] * info["lane_w"]
    ins = np.where(np.all(np.abs(p - c) <= half + 1.0, axis=1))[0]              # 교차로 네모 안의 점
    d0, d1 = p[5] - p[0], p[-1] - p[-6]
    turn = abs(d0[0] * d1[1] - d0[1] * d1[0]) / (np.linalg.norm(d0) * np.linalg.norm(d1))
    sib = next((j for j, q in enumerate(info["paths"]) if j != pi and np.allclose(q[0], p[0])), None)
    return ("left" if turn > 0.5 else "straight"), int(ins[0]), int(ins[-1]), sib


class Routes:
    """한 도로의 경로들을 STEP 간격으로 다시 찍어 두고, 로봇 위치·방향에 맞는 정답 선을 만든다."""

    def __init__(self, sc, lat, yaw):
        """lat = 경로에서 옆으로 벗어나도 그 경로 위로 보는 한계(mm), yaw = 방향 차 한계(도). 사진을 찍을 때 흔드는 범위와 같게 준다."""
        self.sc, self.info, self.lat, self.yaw, self.r, self.cache = sc, sc.info, lat + 1.0, np.radians(yaw) + 1e-3, [], {}
        self.lat_t, self.yaw_t = max(self.lat, TURN_LW * sc.info["lane_w"] + 1.0), max(self.yaw, np.radians(TURN_YAW) + 1e-3)
        right = self.info.get("path_right", [True] * len(self.info["paths"]))
        for j, raw in enumerate(self.info["paths"]):
            p = np.asarray(raw, float)
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            ss = np.arange(0.0, s[-1], STEP)
            q = np.c_[np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])]
            t = np.gradient(q, axis=0)
            kind, en, ex, _ = route(self.info, j)
            ien, iex = (len(q), len(q)) if en is None else (int(s[en] / STEP), int(np.ceil(s[ex] / STEP)))
            self.r.append(dict(p=q, ang=np.arctan2(t[:, 1], t[:, 0]), kind=kind, en=ien, ex=iex, right=right[j]))

    def key(self, x, y, yaw, own=None):
        """로봇 위치(mm)·방향(rad)에서 올라타 있는 경로들 → ((경로 번호, 구간), ...). 구간: 0 교차로 앞, 1 교차로 안, 2 나온 뒤.
        own = 반드시 넣을 경로(시뮬 주행의 정답 모드에서 달리는 경로. 멀리 벗어나도 정답이 끊기지 않게 한다)."""
        out = []
        for j, r in enumerate(self.r):
            d = np.hypot(r["p"][:, 0] - x, r["p"][:, 1] - y)
            i = int(np.argmin(d))
            turn = abs((r["ang"][i] - yaw + np.pi) % (2 * np.pi) - np.pi)
            arc = r["kind"] == "left" and r["en"] <= i <= r["ex"]                 # 좌회전 원호 위: 넓힌 한계
            if j == own or (d[i] <= (self.lat_t if arc else self.lat) and turn <= (self.yaw_t if arc else self.yaw)):
                out.append((j, int(i >= r["en"]) + int(i > r["ex"])))
        return tuple(out)

    def floor(self, key):
        """key에 맞는 정답 선을 바닥 그림(비트 묶음)에 그린다. 같은 key는 한 번만 그리고 다시 쓴다."""
        if key not in self.cache:
            sc, half, t = self.sc, self.info["lane_w"] / 2, self.info.get("line_w", 10.0)
            straight = any(self.r[j]["kind"] != "left" for j, _ in key)
            lines, areas, floor = [], [], np.zeros(sc.inst.shape, np.uint16)   # lines: (선, 오른쪽 선을 그릴지), areas: 좌회전 원호
            for j, zone in key:
                r = self.r[j]
                if r["kind"] != "left":
                    lines.append((r["p"], r["right"]))
                    continue
                if zone == 0:
                    lines.append((r["p"][: r["en"] + 1], r["right"]))
                elif not straight:
                    lines.append((r["p"][r["ex"]:], r["right"]))
                if zone < 2:
                    areas.append(r["p"][r["en"]: r["ex"] + 1])
            px = lambda b: [np.c_[b[:, 0] * sc.ppm, floor.shape[0] - b[:, 1] * sc.ppm].round().astype(np.int32)]
            shapes = [(k, strip(offset(p, d), t)) for p, right in lines if len(p) >= 2
                      for k, (d, ok) in enumerate(((half, True), (-half, right), (0.0, True))) if ok]
            shapes += [(3, np.vstack([offset(p, half), offset(p, -half)[::-1]])) for p in areas if len(p) >= 2]
            for k, b in shapes:
                m = np.zeros(floor.shape, np.uint8)
                cv2.fillPoly(m, px(b), 1)
                floor |= m.astype(np.uint16) << k
            self.cache[key] = floor
        return self.cache[key]


def mine_yolo(mine, names):
    """화면으로 옮긴 비트 묶음 그림 → YOLO-seg 줄들 (클래스 번호, 정규화된 다각형)."""
    h, w = mine.shape
    lines = []
    for k, n in enumerate(MINE):
        cs, _ = cv2.findContours(((mine >> k) & 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            c = cv2.approxPolyDP(c, 0.7, True).reshape(-1, 2) if cv2.contourArea(c) >= MIN_AREA else []
            if len(c) >= 3:
                lines.append(f"{names.index(n)} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c))
    return lines
