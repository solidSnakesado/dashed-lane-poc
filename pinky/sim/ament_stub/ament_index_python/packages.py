"""ROS2가 없는 환경에서 phys_drive.py가 pinky_pro 패키지 경로(서브모듈 소스)를 찾게 하는 대역."""
import os
ROOT = os.environ.get("PINKY_PRO", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "pinky_pro"))
def get_package_share_directory(name):
    return os.path.join(ROOT, name)
class PackageNotFoundError(KeyError): pass
