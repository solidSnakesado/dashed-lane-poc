"""핑키 전방 카메라 시뮬레이션 장면 (MuJoCo). 생성한 도로 바닥을 깔고, 실측 카메라 값으로 화면을 렌더링한다.

  MUJOCO_GL=egl python scene.py ROAD_DIR CAMERA_YAML [--n 8] [--out 폴더]
    → 경로를 따라 n장 렌더링하고, 라벨(같은 카메라로 옮긴 선 번호)을 겹친 확인용 그림을 저장한다.
다른 스크립트에서는 Scene(road_dir, cam_yaml).render(x, y, yaw)로 쓴다.
렌더링 방식: MuJoCo는 왜곡 없는 이상적인 카메라(여유 있게 넓은 화각)로 그리고,
  실측 카메라 행렬·렌즈 왜곡에 맞게 한 번에 다시 매핑한다. 라벨도 같은 카메라 식으로 만들어 그림과 어긋나지 않는다.
핑키 몸체는 전방 카메라에 보이지 않아 넣지 않았다(주행 시험 단계에서 추가).
"""
import argparse
import json
import os

import cv2
import mujoco
import numpy as np
import yaml

MARGIN = 1.25    # 이상적 카메라를 실제보다 이만큼 크게 렌더링(왜곡으로 생기는 가장자리 빈칸 방지)
MAX_DIST = 1.5   # 이보다 먼 바닥(m, 실제 24m)의 선은 너무 흐리고 납작해 라벨에서 뺌


def cam_rotation(yaw, tilt_deg, roll_deg):
    """세계(x 오른쪽, y 앞, z 위) 기준 카메라 축: 오른쪽, 위, 앞. yaw = 진행 방향(라디안, +x에서 반시계)."""
    d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    t, r = np.radians(tilt_deg), np.radians(roll_deg)
    f = np.cos(t) * d + np.sin(t) * np.array([0, 0, -1.0])        # 아래로 숙인 앞 방향
    right = np.array([d[1], -d[0], 0.0])
    up = np.cross(right, f)
    right, up = np.cos(r) * right + np.sin(r) * up, -np.sin(r) * right + np.cos(r) * up   # +: 화면 오른쪽이 위로
    return right, up, f


class Scene:
    def __init__(self, road_dir, cam_yaml):
        with open(cam_yaml) as fp:
            c = yaml.safe_load(fp)
        self.info = json.load(open(os.path.join(road_dir, "road.json")))
        i, e = c["intrinsic"], c["extrinsic"]
        self.W, self.H = c["image_width"], c["image_height"]
        self.K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
        self.dist = np.array(i["dist"], float)
        self.h, self.tilt, self.roll, self.fwd = e["height_m"], e["tilt_down_deg"], e["roll_deg"], e["forward_m"]
        # 이상적 카메라(정사각 픽셀, 중심 정가운데)와, 실제 카메라 픽셀 → 이상적 카메라 픽셀 매핑
        self.iw, self.ih = int(self.W * MARGIN) // 2 * 2, int(self.H * MARGIN) // 2 * 2
        self.f = i["fy"]
        uu, vv = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        n = cv2.undistortPoints(np.c_[uu.ravel(), vv.ravel()].reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        self.mapx = (n[:, 0] * self.f + self.iw / 2).reshape(self.H, self.W).astype(np.float32)
        self.mapy = (n[:, 1] * self.f + self.ih / 2).reshape(self.H, self.W).astype(np.float32)
        sx, sy = [v / 1000 for v in self.info["size_mm"]]
        fovy = np.degrees(2 * np.arctan(self.ih / 2 / self.f))
        xml = f"""<mujoco><visual><global offwidth="{self.iw}" offheight="{self.ih}"/>
          <headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4" specular="0 0 0"/><quality shadowsize="0"/></visual>
          <asset><texture name="floor" type="2d" file="{os.path.abspath(os.path.join(road_dir, 'floor.png'))}"/>
          <material name="floor" texture="floor" texrepeat="1 1" specular="0" shininess="0" reflectance="0"/></asset>
          <worldbody><geom type="plane" size="{sx / 2} {sy / 2} 0.01" pos="{sx / 2} {sy / 2} 0" material="floor"/>
          <camera name="front" pos="0 0 0.1" fovy="{fovy}"/></worldbody></mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, self.ih, self.iw)
        self.inst = cv2.imread(os.path.join(road_dir, "inst.png"), cv2.IMREAD_UNCHANGED)
        self.ppm = self.info["ppm"]

    def pose(self, x, y, yaw, dh=0.0, dtilt=0.0, droll=0.0):
        """로봇 위치(m)·방향으로 카메라 위치와 축을 구한다. d* = 무작위로 흔들 양."""
        right, up, f = cam_rotation(yaw, self.tilt + dtilt, self.roll + droll)
        p = np.array([x + self.fwd * np.cos(yaw), y + self.fwd * np.sin(yaw), self.h + dh])
        return p, right, up, f

    def render(self, x, y, yaw, **jitter):
        """실제 카메라 화면(BGR)과 선 번호 그림(uint16, 0 = 바닥)을 돌려준다."""
        p, right, up, f = self.pose(x, y, yaw, **jitter)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.column_stack([right, up, -f]).ravel())   # MuJoCo 카메라는 -z를 바라봄
        self.model.cam_pos[0], self.model.cam_quat[0] = p, quat
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="front")
        ideal = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        img = cv2.remap(ideal, self.mapx, self.mapy, cv2.INTER_LINEAR)
        # 라벨: 바닥 그림 픽셀 → 이상적 카메라 픽셀 변환(평면이라 정확히 호모그래피) 후 같은 매핑
        R = np.stack([right, -up, f])                                      # 세계 → 카메라(x 오른쪽, y 아래, z 앞)
        Ki = np.array([[self.f, 0, self.iw / 2], [0, self.f, self.ih / 2], [0, 0, 1.0]])
        Hpx = self.inst.shape[0]
        floor_to_world = np.array([[1 / (1000 * self.ppm), 0, 0], [0, -1 / (1000 * self.ppm), Hpx / (1000 * self.ppm)], [0, 0, 1]])
        G = Ki @ np.column_stack([R[:, 0], R[:, 1], -R @ p])               # 바닥 (x, y, 1)[m] → 이상적 화면 (z = 0 평면)
        Hm = G @ floor_to_world
        ideal_inst = cv2.warpPerspective(self.inst, Hm, (self.iw, self.ih), flags=cv2.INTER_NEAREST)
        fl = np.array([f[0], f[1]]) / max(np.hypot(f[0], f[1]), 1e-9)                 # 바닥 위 앞 방향
        side = np.array([fl[1], -fl[0]])
        far = [p[:2] + MAX_DIST * fl + s * side for s in np.linspace(-4, 4, 9)]       # 먼 거리 경계선
        q = np.array([G @ np.r_[pt, 1.0] for pt in far])
        q = q[q[:, 2] > 0]
        if len(q) >= 2:
            q = (q[:, :2] / q[:, 2:]).astype(np.int32)
            q = q[np.argsort(q[:, 0])]
            cv2.fillPoly(ideal_inst, [np.vstack([q, [[q[-1, 0], -10 ** 4], [q[0, 0], -10 ** 4]]]).astype(np.int32)], 0)
        inst = cv2.remap(ideal_inst, self.mapx, self.mapy, cv2.INTER_NEAREST)
        inst[: max(0, int(self.horizon(R))), :] = 0
        return img, inst

    def horizon(self, R):
        """실제 화면에서 지평선 행(대략). 그 위쪽 라벨은 지운다."""
        v = self.K @ (R @ np.array([R[2, 0], R[2, 1], 0.0]) / max(np.hypot(R[2, 0], R[2, 1]), 1e-9))
        return v[1] / v[2] + 2

    def close(self):
        self.renderer.close()


def overlay(img, inst, classes):
    """라벨을 색으로 겹친 확인용 그림."""
    col = {"white_lane": (0, 220, 255), "stop_line": (0, 0, 255), "crosswalk": (255, 120, 0)}
    out = img.copy()
    for k, c in classes.items():
        m = inst == int(k)
        if m.any():
            out[m] = (0.4 * out[m] + 0.6 * np.array(col[c])).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("road")
    ap.add_argument("cam")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sc = Scene(a.road, a.cam)
    out = a.out or os.path.join(a.road, "scene_check")
    os.makedirs(out, exist_ok=True)
    path = np.array(sc.info["paths"][0]) / 1000.0
    idx = np.linspace(5, int(len(path) * 0.75), a.n).astype(int)     # 도로 끝부분은 앞이 비어 있어 제외
    tiles = []
    for k, i in enumerate(idx):
        dx, dy = path[i + 5] - path[i - 5]
        img, inst = sc.render(path[i, 0], path[i, 1], np.arctan2(dy, dx))
        cv2.imwrite(os.path.join(out, f"view_{k:02d}.jpg"), img)
        tiles.append(np.hstack([img, overlay(img, inst, sc.info["instances"])]))
    sheet = np.vstack(tiles[: min(4, len(tiles))])
    cv2.imwrite(os.path.join(out, "sheet.jpg"), cv2.resize(sheet, None, fx=0.6, fy=0.6))
    print(f"저장: {out}/  (sheet.jpg: 왼쪽 렌더링, 오른쪽 라벨 겹침)")
    sc.close()


if __name__ == "__main__":
    main()
