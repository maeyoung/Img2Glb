"""CUDA 확장을 설치 시점에 빌드한다.

메타데이터는 pyproject.toml 에 있고, 이 파일은 확장 빌드만 담당한다.

torch 가 빌드 환경에 없으면 확장을 건너뛴다. 그 경우 런타임에 JIT 로 컴파일되지만
(img2glb/raster/__init__.py 폴백), 최초 실행이 느려지고 nvcc 와 Python 헤더가
필요하므로 아래처럼 설치하는 편이 낫다.

    pip install -e . --no-build-isolation
"""
import os
import sys

from setuptools import setup

def _python_header_dirs():
    """Python 개발 헤더 경로를 찾는다.

    root 권한이 없어 ``apt install python3-dev`` 를 못 하는 환경에서는
    deb 를 풀어 둔 ``.localdev`` 를 쓴다. 설치 스크립트가 CPATH 를 넘겨주지 않는
    경우(``pip install .`` 직접 실행)에도 동작하도록 여기서 직접 찾는다.

    pyconfig.h 가 ``aarch64-linux-gnu/pythonX.Y/pyconfig.h`` 를 상대경로로
    참조하므로 include 루트도 함께 넣는다.
    """
    ver = f"python3.{sys.version_info.minor}"
    env = os.environ.get("PYDEV_INCLUDE")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if env:
        candidates.append(env)
    candidates.append("/usr/include")
    candidates.append(os.path.join(here, ".localdev", "root", "usr", "include"))

    for base in candidates:
        if os.path.isfile(os.path.join(base, ver, "Python.h")):
            return [os.path.join(base, ver), base]
    return []


ext_modules = []
cmdclass = {}

if os.environ.get("IMG2GLB_SKIP_EXT") != "1":
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        ext_modules = [
            CUDAExtension(
                name="img2glb._raster_ext",
                sources=["src/img2glb/raster/csrc/raster.cu"],
                include_dirs=_python_header_dirs(),
            )
        ]
        cmdclass = {"build_ext": BuildExtension}

        # 어떤 torch 로 빌드했는지 기록한다. editable 설치로 여러 venv 가
        # 같은 소스 트리를 공유할 때 ABI 불일치를 감지하기 위해서다.
        import shutil

        import torch

        # setuptools 는 build/ 의 오브젝트를 재사용한다. 다른 torch 버전으로
        # 빌드된 잔재가 남아 있으면 ABI 가 어긋난 .so 가 설치되어, 파이썬에서는
        # CUDA 텐서인데 C++ 에서는 아니라고 나오는 식으로 조용히 깨진다.
        # torch 버전이 달라졌으면 build/ 를 지운다.
        _here = os.path.dirname(os.path.abspath(__file__))
        _stamp = os.path.join(_here, "build", ".torch-version")
        _prev = None
        if os.path.isfile(_stamp):
            with open(_stamp) as fh:
                _prev = fh.read().strip()
        _build = os.path.join(_here, "build")
        if _prev != torch.__version__ and os.path.isdir(_build):
            # 스탬프가 없는 build/ 는 어떤 torch 로 만든 것인지 알 수 없으므로 지운다
            _why = f"{_prev} -> {torch.__version__}" if _prev else "출처 불명"
            print(f"[img2glb] build/ 를 정리합니다 ({_why}).")
            shutil.rmtree(_build, ignore_errors=True)
        os.makedirs(os.path.join(_here, "build"), exist_ok=True)
        with open(_stamp, "w") as fh:
            fh.write(torch.__version__)

        with open("src/img2glb/_build_info.py", "w") as fh:
            fh.write('"""빌드 시 자동 생성됨. 직접 수정하지 말 것."""\n')
            fh.write(f'TORCH_VERSION = "{torch.__version__}"\n')
        print(f"[img2glb] CUDA 확장을 빌드합니다 (torch {torch.__version__}).")
    except ImportError:
        print("[img2glb] torch 를 찾을 수 없어 확장 빌드를 건너뜁니다 "
              "(런타임 JIT 폴백을 사용합니다).")

setup(ext_modules=ext_modules, cmdclass=cmdclass)
