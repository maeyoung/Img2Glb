#!/usr/bin/env bash
# TRELLIS.2 백엔드 설치.
#
#   bash scripts/setup_trellis2.sh [--cuda cu130] [--arch 12.1]
#
# models/trellis2 에 저장소를 받고, CUDA 확장을 빌드하고,
# nvdiffrast 를 img2glb.raster 로 대체하는 패치를 적용한다.
set -euo pipefail

CUDA_TAG="cu130"
ARCH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA_TAG="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --editable) EDITABLE=1; shift ;;
        *) echo "알 수 없는 옵션: $1" >&2; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
T2="$MODELS/trellis2"
PY="${PYTHON:-python3}"

mkdir -p "$MODELS"
if [[ ! -d "$T2" ]]; then
    echo "==> TRELLIS.2 저장소 clone"
    git clone --recursive https://github.com/microsoft/TRELLIS.2.git "$T2"
else
    echo "==> 기존 저장소 사용: $T2"
    git -C "$T2" submodule update --init --recursive
fi

if [[ ! -d "$T2/.venv" ]]; then
    echo "==> venv 생성"
    "$PY" -m venv "$T2/.venv"
fi
VPY="$T2/.venv/bin/python"
"$VPY" -m pip install -q --upgrade pip setuptools wheel

echo "==> torch 설치 ($CUDA_TAG)"
"$VPY" -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" torch torchvision

echo "==> 기본 의존성"
"$VPY" -m pip install -q imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    ninja trimesh transformers pandas zstandard kornia timm huggingface_hub safetensors scipy pybind11 \
    einops torchvision   # einops/timm: BiRefNet 배경제거(trust_remote_code) 요구사항
"$VPY" -m pip install -q --no-deps \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# ---- Python 개발 헤더 (확장 빌드에 필요) ----
# root 권한이 없어 `apt install python3-dev` 를 못 하는 환경을 위해 deb 를 풀어 쓴다.
PYVER="$("$VPY" -c 'import sys;print(f"python3.{sys.version_info.minor}")')"
if [[ -f "/usr/include/$PYVER/Python.h" ]]; then
    PYDEV_INCLUDE="/usr/include"
elif [[ -f "$ROOT/.localdev/root/usr/include/$PYVER/Python.h" ]]; then
    PYDEV_INCLUDE="$ROOT/.localdev/root/usr/include"
else
    echo "==> Python 개발 헤더가 없어 deb 를 받아 푼다 (root 불필요)"
    tmp="$(mktemp -d)"
    ( cd "$tmp" && apt-get download "lib${PYVER}-dev" "${PYVER}-dev" )
    mkdir -p "$ROOT/.localdev/root"
    for deb in "$tmp"/*.deb; do dpkg-deb -x "$deb" "$ROOT/.localdev/root"; done
    rm -rf "$tmp"
    PYDEV_INCLUDE="$ROOT/.localdev/root/usr/include"
fi
# pyconfig.h 가 aarch64-linux-gnu/... 를 상대경로로 참조하므로 include 루트까지 넣는다
export CPATH="$PYDEV_INCLUDE/$PYVER:$PYDEV_INCLUDE${CPATH:+:$CPATH}"
echo "==> Python 헤더: $PYDEV_INCLUDE"

if [[ -z "$ARCH" ]]; then
    ARCH="$("$VPY" -c 'import torch;m,n=torch.cuda.get_device_capability(0);print(f"{m}.{n}")')"
fi
echo "==> CUDA arch: $ARCH"

echo "==> CUDA 확장 빌드 (FlexGEMM, CuMesh, o-voxel)"
mkdir -p "$T2/extensions"
for repo in FlexGEMM CuMesh; do
    [[ -d "$T2/extensions/$repo" ]] || \
        git clone --recursive --depth 1 "https://github.com/JeffreyXiang/$repo.git" "$T2/extensions/$repo"
    ( cd "$T2/extensions/$repo" && \
      TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-8}" "$VPY" -m pip install . --no-build-isolation )
done
( cd "$T2/o-voxel" && \
  TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-8}" "$VPY" -m pip install . --no-build-isolation )

echo "==> img2glb 설치 (CUDA 확장 포함)"
# --no-build-isolation: setup.py 가 torch 를 봐야 확장을 빌드할 수 있다.
# 확장을 설치 시점에 컴파일해두면 런타임 JIT(락 대기 / 최초 지연) 이 없다.
# 기본은 일반 설치다. venv 마다 확장이 따로 컴파일되어 torch 버전이 달라도
# 충돌하지 않는다. 코드를 고쳐가며 쓸 때만 --editable 을 준다
# (단, editable 은 소스 트리를 공유하므로 venv 를 하나만 쓸 때에 한한다).
PIP_TARGET_ARG="$ROOT"
[[ "${EDITABLE:-0}" == "1" ]] && PIP_TARGET_ARG="-e $ROOT"
TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-8}" \
    "$VPY" -m pip install $PIP_TARGET_ARG --no-build-isolation

echo "==> 패치 적용 (sdpa 백엔드 / nvdiffrast 제거)"
"$VPY" "$ROOT/scripts/apply_patches.py" --trellis2 "$T2"

# venv 를 다른 경로로 옮긴 경우 activate 안의 VIRTUAL_ENV 와 콘솔 스크립트
# shebang 이 옛 경로를 가리켜 "command not found" 가 난다. 여기서 바로잡는다.
echo "==> venv 경로 정합성 확인"
VENV="$T2/.venv"
if ! grep -q "VIRTUAL_ENV=$VENV\b" "$VENV/bin/activate" 2>/dev/null; then
    echo "    activate 의 경로가 어긋나 있어 수정합니다."
    for f in activate activate.csh activate.fish activate.nu activate.ps1; do
        [[ -f "$VENV/bin/$f" ]] && \
            sed -i "s|[A-Za-z0-9_/.-]*/\.venv|$VENV|g" "$VENV/bin/$f"
    done
fi
# 콘솔 스크립트 shebang
for f in "$VENV"/bin/*; do
    [[ -f "$f" ]] || continue
    head -c 2 "$f" 2>/dev/null | grep -q '^#!' || continue
    sed -i "1s|^#!.*/\.venv/bin/python.*|#!$VENV/bin/python|" "$f" 2>/dev/null || true
done

echo "==> 설치 확인"
"$VENV/bin/img2glb" --version
"$VENV/bin/python" -c "
from img2glb.raster import _get_ext
e = _get_ext()
src = getattr(e, '__file__', '')
print('  래스터 확장:', '설치 시 빌드' if '_raster_ext' in src else '런타임 JIT')
"

echo
echo "완료. 사용 예:"
echo "  source $VENV/bin/activate"
echo "  img2glb --model trellis2 --image samples/bird.png"
echo
echo "  (activate 없이:  $VENV/bin/img2glb --image ...)"
