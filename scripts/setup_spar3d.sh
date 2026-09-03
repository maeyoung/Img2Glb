#!/usr/bin/env bash
# SPAR3D 백엔드 설치.
#
#   bash scripts/setup_spar3d.sh [--cuda cu130] [--arch 12.1]
#
# models/spar3d 에 저장소를 받고, 전용 venv 를 만들고,
# texture_baker / uv_unwrapper 확장을 빌드하고, 패치를 적용한다.
#
# 모델 가중치는 gated 다. 먼저 접근 승인을 받고 `hf auth login` 을 해 둘 것.
#   https://huggingface.co/stabilityai/stable-point-aware-3d
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
SP="$MODELS/spar3d"
PY="${PYTHON:-python3}"

mkdir -p "$MODELS"
if [[ ! -d "$SP" ]]; then
    echo "==> SPAR3D 저장소 clone"
    git clone --depth 1 https://github.com/Stability-AI/stable-point-aware-3d.git "$SP"
else
    echo "==> 기존 저장소 사용: $SP"
fi

if [[ ! -d "$SP/.venv" ]]; then
    echo "==> venv 생성"
    "$PY" -m venv "$SP/.venv"
fi
VPY="$SP/.venv/bin/python"
"$VPY" -m pip install -q --upgrade pip wheel
# AlphaCLIP 의 setup.py 가 pkg_resources 를 쓴다 (setuptools 81 에서 제거됨)
"$VPY" -m pip install -q "setuptools<81"

echo "==> torch 설치 ($CUDA_TAG)"
"$VPY" -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" torch torchvision

echo "==> 기본 의존성"
# numpy / transformers 등은 저장소 requirements.txt 의 핀을 따르지 않는다.
# 핀이 낡아 aarch64 + torch cu130 조합과 맞지 않는다. transformers 만 4.x 로 묶는데,
# spar3d 가 복사해 둔 DINOv2 구현이 5.x 에서 사라진 내부 API 를 쓰기 때문이다.
"$VPY" -m pip install -q \
    einops jaxtyping omegaconf "transformers<5" trimesh numpy huggingface_hub \
    safetensors pillow tqdm ninja pybind11 scikit-image opencv-python-headless \
    timm kornia loralib ftfy regex

echo "==> AlphaCLIP / CLIP (image estimator 용)"
"$VPY" -m pip install -q --no-build-isolation \
    "git+https://github.com/openai/CLIP.git" \
    "git+https://github.com/SunzeY/AlphaCLIP.git"

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

echo "==> 확장 빌드 (uv_unwrapper, texture_baker)"
( cd "$SP" && TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-8}" \
    "$VPY" -m pip install --no-build-isolation ./uv_unwrapper/ ./texture_baker/ )

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

echo "==> 패치 적용 (transparent_background 의존 제거)"
"$VPY" "$ROOT/scripts/apply_patches.py" --spar3d "$SP"

echo "==> 모델 가중치 내려받기 (gated - 접근 승인 필요)"
"$VPY" -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/stable-point-aware-3d')
print('ok')
"

# venv 를 다른 경로로 옮긴 경우 activate 안의 VIRTUAL_ENV 와 콘솔 스크립트
# shebang 이 옛 경로를 가리켜 "command not found" 가 난다. 여기서 바로잡는다.
echo "==> venv 경로 정합성 확인"
VENV="$SP/.venv"
if ! grep -q "VIRTUAL_ENV=$VENV\b" "$VENV/bin/activate" 2>/dev/null; then
    echo "    activate 의 경로가 어긋나 있어 수정합니다."
    for f in activate activate.csh activate.fish activate.nu activate.ps1; do
        [[ -f "$VENV/bin/$f" ]] && \
            sed -i "s|[A-Za-z0-9_/.-]*/\.venv|$VENV|g" "$VENV/bin/$f"
    done
fi
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
echo "  img2glb --model spar3d --image samples/bird.png"
echo
echo "  (activate 없이:  $VENV/bin/img2glb --model spar3d -i ...)"
