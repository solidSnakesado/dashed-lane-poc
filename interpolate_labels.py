"""reviewed로 체크한 키프레임 사이의 프레임 라벨을 선형 보간으로 채우는 스크립트.

사용법:
    python interpolate_labels.py ~/lane/dataset/images 53 59

동작:
    - 지정한 범위(시작~끝 프레임 번호, 양 끝 포함) 안에서 reviewed가 체크된 프레임을 키프레임으로 사용한다.
    - 인접한 두 키프레임 사이에 있는 reviewed가 아닌 프레임의 라벨을 새로 만든다.
    - 두 키프레임 모두에 있는 슬롯(lane0~lane3)만 보간한다. 한쪽에만 있는 슬롯은 만들지 않는다.
    - 덮어쓰기 전에 원래 JSON을 <이름>.json.bak으로 백업한다.
    - 보간된 프레임은 reviewed가 꺼진 상태로 저장되므로, labelme에서 확인한 뒤 직접 체크해야 한다.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

N_POINTS = 12  # 보간 결과 선 하나에 남길 점 개수


def load(path):
    with open(path) as f:
        return json.load(f)


def is_reviewed(data):
    return bool(data.get("flags", {}).get("reviewed", False))


def lanes_of(data):
    # {슬롯 라벨: 점 배열(N,2)} 형태로 변환
    out = {}
    for s in data.get("shapes", []):
        if s.get("shape_type") == "linestrip" and len(s["points"]) >= 2:
            out[s["label"]] = np.asarray(s["points"], dtype=float)
    return out


def orient_near_first(pts):
    # 화면 아래쪽(가까운 쪽, y가 큰 점)이 첫 점이 되도록 방향을 통일
    return pts if pts[0, 1] >= pts[-1, 1] else pts[::-1]


def resample(pts, n=N_POINTS):
    # 선 길이 기준으로 점 n개를 같은 간격으로 다시 뽑음 (두 선의 점 개수를 맞추기 위함)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0, dist[-1], n)
    return np.stack([np.interp(t, dist, pts[:, 0]), np.interp(t, dist, pts[:, 1])], axis=1)


def blend(a, b, w):
    # 두 키프레임의 같은 슬롯 선을 가중치 w(0~1)로 섞음
    a = resample(orient_near_first(a))
    b = resample(orient_near_first(b))
    return (1 - w) * a + w * b


def frame_no(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def main():
    img_dir = os.path.expanduser(sys.argv[1])
    start, end = int(sys.argv[2]), int(sys.argv[3])

    # 범위 안의 JSON 목록 (프레임 번호 순)
    paths = sorted((p for p in glob.glob(os.path.join(img_dir, "*.json"))
                    if start <= frame_no(p) <= end), key=frame_no)
    if not paths:
        print("범위 안에 JSON 파일이 없습니다.")
        return

    keys = [p for p in paths if is_reviewed(load(p))]
    print("키프레임:", [frame_no(p) for p in keys])
    if len(keys) < 2:
        print("reviewed 키프레임이 2개 이상 필요합니다.")
        return

    made = 0
    for ka, kb in zip(keys, keys[1:]):
        fa, fb = frame_no(ka), frame_no(kb)
        la, lb = lanes_of(load(ka)), lanes_of(load(kb))
        common = sorted(set(la) & set(lb))
        for p in paths:
            f = frame_no(p)
            if not (fa < f < fb):
                continue
            data = load(p)
            if is_reviewed(data):
                continue  # 직접 확인한 프레임은 건드리지 않음
            w = (f - fa) / (fb - fa)
            shapes = [{"label": lab, "points": [[round(x, 1), round(y, 1)] for x, y in blend(la[lab], lb[lab], w)],
                       "group_id": None, "description": "", "shape_type": "linestrip", "flags": {}}
                      for lab in common]
            shutil.copy(p, p + ".bak")  # 원래 라벨 백업
            data["shapes"] = shapes
            data["flags"] = {"reviewed": False}
            with open(p, "w") as fo:
                json.dump(data, fo, indent=2)
            made += 1
            print(f"{f:04d}: {fa:04d}~{fb:04d} 보간 (w={w:.2f}, 슬롯 {common})")
    print(f"완료: {made}개 프레임 보간")


if __name__ == "__main__":
    main()
