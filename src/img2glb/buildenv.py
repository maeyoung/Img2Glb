"""런타임 컴파일 환경 준비.

이 프로젝트는 실행 중에 C/CUDA 를 컴파일하는 지점이 두 곳 있다.

1. ``img2glb.raster`` 확장을 JIT 로 빌드할 때 (설치 시 빌드했으면 생략된다)
2. **Triton** 이 커널을 컴파일할 때. 백엔드의 sparse conv 가 Triton 커널을
   쓰는데, Triton 은 첫 실행에서 ``cuda_utils.c`` 를 gcc 로 컴파일하며
   ``Python.h`` 를 필요로 한다. 한 번 컴파일하면 ``~/.triton/cache`` 에
   남으므로, 캐시가 있는 기존 환경에서는 이 문제가 드러나지 않는다.

둘 다 gcc 를 하위 프로세스로 띄우므로 ``CPATH`` 를 환경변수로 넘기면 된다.
root 권한이 없어 ``apt install python3-dev`` 를 못 하는 환경을 위해
deb 를 풀어 둔 ``.localdev`` 도 함께 찾는다.
"""
import os
import sys
from pathlib import Path

_prepared = False


def find_python_headers():
    """``Python.h`` 가 있는 include 루트를 찾는다. 없으면 None."""
    ver = f"python3.{sys.version_info.minor}"

    env = os.environ.get("PYDEV_INCLUDE")
    if env and (Path(env) / ver / "Python.h").is_file():
        return env

    if (Path("/usr/include") / ver / "Python.h").is_file():
        return "/usr/include"

    # 저장소 루트나 상위 디렉터리의 .localdev
    roots = [Path.cwd(), *Path.cwd().parents,
             *Path(__file__).resolve().parents]
    for r in roots:
        cand = r / ".localdev" / "root" / "usr" / "include"
        if (cand / ver / "Python.h").is_file():
            return str(cand)
    return None


def prepare(verbose=False):
    """CPATH 와 PATH 를 보강한다. 여러 번 불러도 안전하다."""
    global _prepared
    if _prepared:
        return
    _prepared = True

    # ninja 가 venv 안에만 있을 수 있다
    venv_bin = os.path.join(sys.prefix, "bin")
    if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

    inc = find_python_headers()
    if not inc:
        if verbose:
            print("[img2glb] Python 개발 헤더를 찾지 못했습니다. "
                  "Triton 커널 컴파일이 실패할 수 있습니다.\n"
                  "          PYDEV_INCLUDE 로 경로를 지정하거나 "
                  "python3-dev 를 설치하세요.", file=sys.stderr)
        return

    ver = f"python3.{sys.version_info.minor}"
    want = f"{inc}/{ver}:{inc}"
    cur = os.environ.get("CPATH", "")
    if want not in cur:
        os.environ["CPATH"] = want + (":" + cur if cur else "")
