"""도로 규격·카메라에서 후처리 기준값을 계산해 각 모듈에 넣는다 (규격 수치가 바뀌어도 코드를 고치지 않기 위함).

후처리 좌표는 '차로 폭을 3.3으로 맞춘 단위'다(보정 파일의 scale). 차로 폭에 비례하는 기준값은 그대로 통한다.
차로 폭과 따로 변하는 것(점선 길이·간격, 선 굵기, 곡선 반경, 카메라가 보는 거리)에 걸린 기준값만 여기서 다시 계산한다.
각 기준값 = 기준 규격(차로 3.3 m, 점선 3 m, 간격 5 m, 선 굵기 0.15 m)일 때의 값 × (이 도로의 규격 ÷ 기준 규격).

규격은 파일로 받지 않는다. Auto가 주행 화면에서 잰 값을 쓴다(lane_measure.py). 아직 못 잰 항목은 기준 규격의 비율로 채운다.
"""
import numpy as np

import lane_geom
import lane_group
import lane_motion
import lane_roles
import lane_track
from lane_measure import Measure

REF = {"lane_w": 3.3, "dash": 3.0, "gap": 5.0, "line_w": 0.15}   # 기준 규격(m)
SIDE = 0.75        # 기준 거리: 좌우로 차로 폭의 이 배수 지점이 화면에 함께 보이는 가장 가까운 거리
KEEP_K, HOLD_K = 1.0, 0.5   # 안 보이는 점은 점선 한 주기(점선+간격)를 달릴 동안, 통째로 안 보이는 내 차로 선은 반 주기 동안 유지
YAW_TOL = 20.0     # 차로 선이 진행 방향과 벌어질 수 있는 기본 각(도): 차로 안에서 비스듬히 달리는 경우


def seen(cam, size, x, z):
    """바닥 점 (x, z)가 화면 안에 보이는지 (렌즈 왜곡 계산이 터지는 점은 되돌려 보아 걸러 냄)."""
    px = cam.to_image(np.array([[x, z]]))
    if not np.all(np.isfinite(px)) or not (0 <= px[0, 0] < size[0] and 0 <= px[0, 1] < size[1]):
        return False
    back, ok = cam.to_ground(px)
    return bool(ok[0]) and np.linalg.norm(back[0] - [x, z]) < 0.05 * max(z, 1.0)


def view(cam, size, dash):
    """카메라가 보는 범위: (양쪽 선이 함께 보이는 가장 가까운 거리, 믿을 수 있는 가장 먼 거리).

    먼 거리 = 화면 세로 1픽셀이 바닥에서 점선 길이의 절반을 넘게 차지하기 시작하는 거리(그보다 멀면 조각 구분 불가)."""
    half = SIDE * REF["lane_w"]
    near = next((z for z in np.arange(0.5, 40.0, 0.1) if seen(cam, size, -half, z) and seen(cam, size, half, z)), None)
    rows = np.arange(size[1] - 1, 0, -1.0)
    g, ok = cam.to_ground(np.c_[np.full(len(rows), size[0] / 2), rows])
    z = np.where(ok & (g[:, 1] > 0), g[:, 1], 1e9)       # 지평선 위는 아주 먼 것으로 침
    bad = np.where(np.diff(z) > dash / 2)[0]
    far = float(z[bad[0]]) if len(bad) else float(z.max())
    return near, far


def derive(spec, scale, cam, size):
    """규격(mm)과 카메라에서 기준값을 계산한다. {모듈: {이름: 값}}와 설명용 요약을 돌려준다."""
    lane = spec["lane_w"]
    get = lambda k: spec.get(k, lane * REF[k] / REF["lane_w"]) * scale / 1000.0   # 후처리 단위로 바꾼 규격
    d, g, t = get("dash"), get("gap"), get("line_w")
    rad = spec["min_radius"] * scale / 1000.0 if spec.get("min_radius") else None
    near, far = view(cam, size, d)
    rd, rg, rt, rp = d / REF["dash"], g / REF["gap"], t / REF["line_w"], (d + g) / (REF["dash"] + REF["gap"])
    out = {
        lane_geom: {"MAX_WIDTH": 1.0 * rt, "MIN_LEN": 0.4 * rd, "SHORT_LEN": 1.2 * rd, "DIR_SPAN": 3.0 * rd,
                    "Z_MAX": far},
        lane_group: {"MAX_GAP": 15.0 * (2 * g + d) / (2 * REF["gap"] + REF["dash"]), "SHORT_GAP": 3.0 * rg,
                     "SOLID_LEN": 8.0 * rd, "CHAIN_TRUST": 2.5 * rd, "DIR_NOISE": 0.2 * rt,
                     "AHEAD": far / 3, "FAR_W": far / 2},
        lane_motion: {"END_GATE": 2.5 * rp, "EXTEND": 8.0 * rp, "REACH_MAX": far * 5 / 6, "AHEAD": far / 3},
        lane_track: {"KEEP_Z": far / 4, "MIN_NEW": 2.0 * rd, "KEEP_DIST": KEEP_K * (d + g), "HOLD_DIST": HOLD_K * (d + g)},
        lane_roles: {"ROLE_LEN": d, "PAIR_LEN": d / 3, "SIDE_Z": 0.0},
    }
    if near is not None:
        out[lane_roles]["Z_REF"] = near
    if rad:   # 급한 곡선일수록 점선 사이가 더 꺾이고, 기준 거리에서 선이 진행 방향과 더 벌어짐
        out[lane_group]["MAX_BEND"] = max(lane_group.MAX_BEND, 1.25 * np.degrees((d + g) / rad))
        out[lane_roles]["ROLE_TURN"] = YAW_TOL + np.degrees(out[lane_roles].get("Z_REF", lane_roles.Z_REF) / rad)
    note = (f"규격(후처리 단위): 점선 {d:.2f} / 간격 {g:.2f} / 선 굵기 {t:.2f}"
            + (f" / 최소 반경 {rad:.1f}" if rad else "") + f", 카메라: 기준 거리 {near and round(near, 1)}, 먼 한계 {far:.1f}")
    return out, note


def apply(spec, scale, cam, size):
    """규격(mm)으로 기준값을 계산해 각 모듈에 넣는다."""
    out, note = derive(spec, scale, cam, size)
    for mod, vals in out.items():
        for k, v in vals.items():
            assert hasattr(mod, k), f"{mod.__name__}.{k} 없음"
            setattr(mod, k, float(v))
    print(note + "\n  계산한 기준값: " + ", ".join(f"{k}={v:.2f}" for vals in out.values() for k, v in vals.items()))


class Auto:
    """규격을 사람이 넣지 않고, 주행 화면에서 잰 값(lane_measure.py)으로 기준값을 맞춘다.

    차로 폭이 측정되면 좌표 단위를 고정한다(차로 폭 = 3.3이 되게 카메라 변환을 키움). 그 전 프레임은 측정만 한다.
    점선 길이·간격처럼 나중에 측정되는 항목은 측정되는 대로 기준값에 반영한다."""
    def __init__(self, calib_path, size):
        self.real, self.size = lane_geom.Camera(calib_path), size      # 실제 m 단위 카메라(측정용)
        d = np.load(calib_path)
        self.m = Measure(self.real, size, float(d["cam_h"]) if "cam_h" in d else None)
        self.scale, self.known = None, 0

    def feed(self, polys, cam, name=""):
        """한 프레임의 차선 폴리곤을 측정에 넣는다. 좌표 단위가 정해졌으면 True (그때부터 후처리를 돌림)."""
        self.m.add([p for cls, p in polys if cls == "white_lane"])
        spec = self.m.spec()
        if self.scale is None and "lane_w" in spec:
            self.scale = REF["lane_w"] / spec["lane_w"]
            cam.H = np.diag([self.scale, self.scale, 1.0]) @ self.real.H
            cam.Hinv = np.linalg.inv(cam.H)
            print(f"  {name}: 차로 폭 {spec['lane_w'] * 1000:.0f} mm로 측정 → 좌표 단위 고정")
        if self.scale is not None and len(spec) != self.known:          # 새로 측정된 항목이 생기면 기준값 갱신
            self.known = len(spec)
            apply({k: v * 1000.0 for k, v in spec.items()}, self.scale, cam, self.size)
        return self.scale is not None
