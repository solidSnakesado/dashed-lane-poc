"""핑키 카메라 내부 보정 (화각·렌즈 왜곡). 체커보드 사진들로 카메라 행렬 K와 왜곡 계수를 구한다.

사용:  python calib_intrinsic.py ~/calib --cols 8 --rows 6 --square 30
  --cols, --rows : 체커보드 안쪽 모서리 수 (가로로 놓았을 때 기준. 세로로 찍힌 사진도 자동으로 찾음)
  --square       : 칸 한 변 길이(mm)
결과:
  ~/calib/camera_intrinsics.yaml  카메라 행렬, 왜곡 계수, 화각, 해상도
  ~/calib/check/                  모서리를 찾은 사진(초록 = 사용, 빨강 = 오차가 커서 뺀 사진)
"""
import argparse
import glob
import os

import cv2
import numpy as np

DROP_FACTOR = 2.5    # 사진별 오차가 전체 중앙값의 이 배수를 넘으면 흔들린 사진으로 보고 뺌
MIN_IMAGES = 10      # 이보다 적게 남으면 결과를 믿기 어려움


def find_corners(gray, size):
    """체커보드 안쪽 모서리를 찾는다. 못 찾으면 None."""
    ok, pts = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    if ok:
        return pts
    ok, pts = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), crit)


def per_image_error(obj, img, rv, tv, K, dist):
    """사진 한 장의 재투영 오차(픽셀, 평균)."""
    proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))


def calibrate(objs, imgs, size):
    """OpenCV 표준 모델(k1, k2, p1, p2, k3)로 보정한다."""
    rms, K, dist, rvs, tvs = cv2.calibrateCamera(objs, imgs, size, None, None, flags=cv2.CALIB_FIX_K3)
    errs = [per_image_error(o, i, r, t, K, dist) for o, i, r, t in zip(objs, imgs, rvs, tvs)]
    return rms, K, dist, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=30.0)
    a = ap.parse_args()
    folder = os.path.expanduser(a.folder)
    files = sorted(glob.glob(os.path.join(folder, "img_*.jpg")) + glob.glob(os.path.join(folder, "img_*.png")))
    check = os.path.join(folder, "check")
    os.makedirs(check, exist_ok=True)

    # 체커보드 위 모서리의 실제 좌표(mm). 판 위가 z = 0 평면
    grid = np.zeros((a.cols * a.rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square

    names, objs, imgs, shown, size = [], [], [], {}, None
    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        pts = find_corners(gray, (a.cols, a.rows))
        if pts is None:                               # 세로로 찍힌 경우를 위해 가로세로를 바꿔 다시 찾음
            pts = find_corners(gray, (a.rows, a.cols))
            g2 = np.zeros_like(grid)
            g2[:, :2] = np.mgrid[0:a.rows, 0:a.cols].T.reshape(-1, 2) * a.square
            obj = g2
        else:
            obj = grid
        name = os.path.basename(f)
        if pts is None:
            print(f"  {name}: 체커보드를 찾지 못함 (화면 밖으로 잘렸거나 흐림)")
            continue
        names.append(name); objs.append(obj); imgs.append(pts); shown[name] = (img, pts, obj.shape[0])
    print(f"사진 {len(files)}장 중 체커보드를 찾은 사진 {len(names)}장")
    if len(names) < MIN_IMAGES:
        print(f"사용할 수 있는 사진이 {MIN_IMAGES}장보다 적어 중단합니다.")
        return

    rms, K, dist, errs = calibrate(objs, imgs, size)
    limit = DROP_FACTOR * float(np.median(errs))
    keep = [i for i, e in enumerate(errs) if e <= limit]
    dropped = [names[i] for i in range(len(names)) if i not in keep]
    if dropped and len(keep) >= MIN_IMAGES:           # 오차가 큰 사진(흔들림, 롤링 셔터)을 빼고 다시 계산
        rms, K, dist, errs = calibrate([objs[i] for i in keep], [imgs[i] for i in keep], size)
        names_used = [names[i] for i in keep]
    else:
        dropped, names_used = [], names

    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = np.degrees(2 * np.arctan(w / (2 * fx)))
    vfov = np.degrees(2 * np.arctan(h / (2 * fy)))
    print(f"사용 {len(names_used)}장, 뺀 사진 {len(dropped)}장 {dropped}")
    print(f"재투영 오차(RMS): {rms:.3f} 픽셀  (0.5 이하면 좋음, 1.0 넘으면 다시 촬영 권장)")
    print(f"해상도 {w}x{h}, fx {fx:.1f} fy {fy:.1f}, 중심 ({cx:.1f}, {cy:.1f})")
    print(f"화각: 가로 {hfov:.1f}도, 세로 {vfov:.1f}도")
    print(f"왜곡 계수 k1 k2 p1 p2 k3: {np.round(dist.ravel(), 4).tolist()}")

    out = os.path.join(folder, "camera_intrinsics.yaml")
    fs = cv2.FileStorage(out, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", w); fs.write("image_height", h)
    fs.write("camera_matrix", K); fs.write("dist_coeffs", dist)
    fs.write("hfov_deg", float(hfov)); fs.write("vfov_deg", float(vfov))
    fs.write("rms_px", float(rms)); fs.write("images_used", len(names_used))
    fs.release()

    for name, (img, pts, n) in shown.items():          # 확인용 그림: 초록 = 사용, 빨강 = 뺌
        pattern = (a.cols, a.rows) if n == a.cols * a.rows and pts.shape[0] == n else (a.rows, a.cols)
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern, pts, True)
        col = (0, 0, 255) if name in dropped else (0, 200, 0)
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), col, 6)
        cv2.imwrite(os.path.join(check, name), vis)
    print(f"저장: {out}")
    print(f"확인용 그림: {check}/")


if __name__ == "__main__":
    main()
