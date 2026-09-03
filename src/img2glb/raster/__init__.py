"""삼각형 래스터라이저 (MIT).

nvdiffrast(NVIDIA 비상업 라이선스) 를 대체하기 위해 직접 구현한 모듈이다.
두 가지 용도로 쓴다.

1. **UV 아틀라스 래스터화** — TRELLIS.2 의 ``o_voxel.postprocess.to_glb`` 가
   텍스처 베이킹에 쓰던 nvdiffrast 호출을 대신한다.
2. **미리보기 렌더** — 생성된 GLB 를 눈으로 확인할 때 쓴다.

nvdiffrast 출력 규약을 그대로 따른다 (실측 확인함)::

    rast[..., 0] = λ0   (정점 0 가중치)
    rast[..., 1] = λ1   (정점 1 가중치)
    rast[..., 2] = 깊이 (UV 모드에서는 0)
    rast[..., 3] = 삼각형 인덱스 + 1   (0 = 배경)

    보간식: attr = λ0·a0 + λ1·a1 + (1-λ0-λ1)·a2
    행 0 이 NDC y = -1 에 대응 (세로 뒤집기 없음)
    픽셀 (j,i) 의 중심은 ((i+0.5)/W, (j+0.5)/H)
"""
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "raster.cu")
_ext = None

__all__ = ["rasterize", "rasterize_mesh", "interpolate"]


def _find_python_headers():
    """Python 개발 헤더 위치를 찾는다.

    root 권한이 없어 ``apt install python3-dev`` 를 못 하는 환경에서는
    deb 를 추출해 둔 경로를 ``PYDEV_INCLUDE`` 로 지정한다.
    설치 스크립트가 저장소 루트에 만들어 두는 ``.localdev`` 도 자동으로 찾는다.
    """
    ver = f"python3.{sys.version_info.minor}"
    env = os.environ.get("PYDEV_INCLUDE")
    if env and os.path.isdir(env):
        return env
    # 디렉터리만 있고 Python.h 가 없는 경우가 있다 (libpython3-minimal)
    if os.path.isfile(os.path.join("/usr/include", ver, "Python.h")):
        return "/usr/include"

    # 저장소 루트(또는 현재 디렉터리) 위쪽의 .localdev/root/usr/include
    from pathlib import Path
    roots = [Path.cwd(), *Path.cwd().parents,
             *Path(__file__).resolve().parents]
    for r in roots:
        cand = r / ".localdev" / "root" / "usr" / "include"
        if (cand / ver / "Python.h").is_file():
            return str(cand)
    return None


def _prebuilt_is_usable():
    """설치 시 빌드된 확장을 이 환경에서 써도 되는지 확인한다.

    editable 설치로 여러 venv 가 하나의 소스 트리를 공유하면, 서로 다른 torch
    버전이 같은 ``.so`` 를 쓰게 된다. C++ ABI 가 달라 오작동할 수 있으므로
    빌드 당시 torch 버전과 다르면 JIT 로 폴백한다.
    """
    try:
        from .._build_info import TORCH_VERSION
    except ImportError:
        return True          # 빌드 정보가 없으면 판단 불가 -> 그대로 시도
    if TORCH_VERSION == torch.__version__:
        return True
    print(f"[img2glb] 확장이 torch {TORCH_VERSION} 로 빌드되었는데 현재는 "
          f"{torch.__version__} 입니다. JIT 로 폴백합니다.\n"
          f"          이 환경에서 'pip install -e . --no-build-isolation' 으로 "
          f"다시 빌드하면 폴백이 사라집니다.", file=sys.stderr)
    return False


def _clear_stale_lock(build_dir, timeout=None):
    """torch JIT 의 stale lock 을 제거한다.

    torch 의 ``FileBaton`` 은 빌드 디렉터리에 ``lock`` 파일을 만들고 사라질
    때까지 폴링한다. 빌드 중 프로세스가 죽으면 파일만 남아 이후 모든 실행이
    **무한 대기**한다. 오래된 lock 은 소유자가 없다고 보고 지운다.
    """
    if timeout is None:
        timeout = float(os.environ.get("IMG2GLB_LOCK_TIMEOUT", 300))
    lock = os.path.join(build_dir, "lock")
    try:
        age = time.time() - os.path.getmtime(lock)
    except OSError:
        return
    if age > timeout:
        try:
            os.remove(lock)
            print(f"[img2glb] {age:.0f}초 된 JIT lock 을 제거했습니다: {lock}",
                  file=sys.stderr)
        except OSError:
            pass


def _get_ext():
    """래스터라이저 확장을 반환한다.

    1. 설치 시 컴파일된 ``img2glb._raster_ext`` 를 먼저 쓴다.
    2. 없으면 JIT 로 컴파일한다 (nvcc 와 Python 헤더 필요).
    """
    global _ext
    if _ext is not None:
        return _ext

    if _prebuilt_is_usable():
        try:
            from .. import _raster_ext        # 설치 시 빌드된 확장
            _ext = _raster_ext
            return _ext
        except ImportError:
            pass

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _default_arch())

    venv_bin = os.path.join(sys.prefix, "bin")          # ninja 가 여기에만 있을 수 있다
    if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

    inc = _find_python_headers()
    if inc:
        ver = f"python3.{sys.version_info.minor}"
        want = f"{inc}/{ver}:{inc}"
        cur = os.environ.get("CPATH", "")
        if want not in cur:
            os.environ["CPATH"] = want + (":" + cur if cur else "")

    try:
        from torch.utils.cpp_extension import _get_build_directory
        _clear_stale_lock(_get_build_directory("img2glb_raster", False))
    except Exception:
        pass

    _ext = load(name="img2glb_raster", sources=[_SRC], verbose=False)
    return _ext


def _default_arch():
    """설치된 GPU 의 compute capability 를 그대로 쓴다."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"
    return "8.0"


def _barycentric(tri, uv_or_xy, faces, H, W, ys, xs):
    """커버리지 결과로부터 픽셀별 바리센트릭을 계산한다."""
    f = faces[tri.long() - 1]
    p0, p1, p2 = uv_or_xy[f[:, 0]], uv_or_xy[f[:, 1]], uv_or_xy[f[:, 2]]
    px = (xs.float() + 0.5) / W
    py = (ys.float() + 0.5) / H
    x0, y0 = p0[:, 0], p0[:, 1]
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    inv = torch.where(area != 0, 1.0 / area, torch.zeros_like(area))
    l0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv
    l1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv
    return l0, l1


def rasterize(uv: torch.Tensor, faces: torch.Tensor, resolution) -> torch.Tensor:
    """UV 아틀라스를 래스터화한다 (깊이 없음).

    Args:
        uv: [V, 2] CUDA, UV 좌표 0~1
        faces: [F, 3] CUDA
        resolution: [H, W] 또는 int

    Returns:
        [1, H, W, 4] float32
    """
    H, W = (resolution, resolution) if isinstance(resolution, int) else (
        int(resolution[0]), int(resolution[1]))
    uv = uv.contiguous().float()
    faces = faces.contiguous().int()

    tri = _get_ext().uv_coverage(uv, faces, H, W)
    rast = torch.zeros((1, H, W, 4), dtype=torch.float32, device=uv.device)
    rast[0, ..., 3] = tri.float()

    mask = tri > 0
    if not mask.any():
        return rast
    ys, xs = mask.nonzero(as_tuple=True)
    l0, l1 = _barycentric(tri[mask], uv, faces, H, W, ys, xs)
    rast[0, ys, xs, 0] = l0
    rast[0, ys, xs, 1] = l1
    return rast


def rasterize_mesh(xyz: torch.Tensor, faces: torch.Tensor, resolution) -> torch.Tensor:
    """화면 공간 메시를 래스터화한다 (z 버퍼 적용).

    Args:
        xyz: [V, 3] CUDA. x∈[0,W], y∈[0,H] 픽셀 좌표, z 는 0 이상이며 작을수록 가깝다.
        faces: [F, 3] CUDA
        resolution: [H, W] 또는 int

    Returns:
        [1, H, W, 4] float32. 채널 2 는 보간된 깊이.
    """
    H, W = (resolution, resolution) if isinstance(resolution, int) else (
        int(resolution[0]), int(resolution[1]))
    xyz = xyz.contiguous().float()
    faces = faces.contiguous().int()

    keys = _get_ext().mesh_coverage(xyz, faces, H, W)
    tri = (keys & 0xFFFFFFFF).int()
    tri = torch.where(keys == -1, torch.zeros_like(tri), tri)

    rast = torch.zeros((1, H, W, 4), dtype=torch.float32, device=xyz.device)
    rast[0, ..., 3] = tri.float()

    mask = tri > 0
    if not mask.any():
        return rast
    ys, xs = mask.nonzero(as_tuple=True)

    # 바리센트릭은 픽셀 좌표계에서 계산한다 (정규화 좌표로 환산)
    xy_norm = torch.stack([xyz[:, 0] / W, xyz[:, 1] / H], dim=-1)
    l0, l1 = _barycentric(tri[mask], xy_norm, faces, H, W, ys, xs)
    rast[0, ys, xs, 0] = l0
    rast[0, ys, xs, 1] = l1

    f = faces[tri[mask].long() - 1]
    z = xyz[:, 2]
    rast[0, ys, xs, 2] = l0 * z[f[:, 0]] + l1 * z[f[:, 1]] + (1 - l0 - l1) * z[f[:, 2]]
    return rast


def interpolate(attr: torch.Tensor, rast: torch.Tensor, faces: torch.Tensor):
    """rast 의 바리센트릭으로 정점 속성을 보간한다.

    Args:
        attr: [1, V, C] 또는 [V, C]
        rast: [1, H, W, 4]
        faces: [F, 3]

    Returns:
        (out, None) — nvdiffrast 와 동일하게 튜플로 반환한다.
    """
    if attr.dim() == 3:
        attr = attr[0]
    faces = faces.contiguous().int()
    _, H, W, _ = rast.shape
    out = torch.zeros((1, H, W, attr.shape[-1]), dtype=attr.dtype, device=attr.device)

    tri = rast[0, ..., 3].long()
    mask = tri > 0
    if not mask.any():
        return out, None
    ys, xs = mask.nonzero(as_tuple=True)
    f = faces[tri[mask] - 1].long()
    l0 = rast[0, ys, xs, 0].to(attr.dtype).unsqueeze(-1)
    l1 = rast[0, ys, xs, 1].to(attr.dtype).unsqueeze(-1)
    out[0, ys, xs] = l0 * attr[f[:, 0]] + l1 * attr[f[:, 1]] + (1 - l0 - l1) * attr[f[:, 2]]
    return out, None
