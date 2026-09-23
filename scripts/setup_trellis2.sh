#!/usr/bin/env bash
# TRELLIS.2 백엔드 설치.
#
#   bash scripts/setup_trellis2.sh [--cuda cu130] [--arch 12.1] [--torch 2.10.0]
#
# models/trellis2 에 저장소를 받고, CUDA 확장을 빌드하고,
# nvdiffrast 를 img2glb.raster 로 대체하는 패치를 적용한다.
set -euo pipefail

CUDA_TAG="cu130"
ARCH=""
# torch 2.13 부터 C++20 을 요구하는데 FlexGEMM / CuMesh / o-voxel 은
# setup.py 에 -std=c++17 을 하드코딩해 두어 빌드가 깨진다.
# 검증된 조합으로 고정한다. 바꾸려면 --torch / --torchvision 을 쓸 것.
TORCH_VER="2.10.0"
TORCHVISION_VER="0.25.0"

# 업스트림 커밋을 고정한다. apply_patches.py 는 문자열 치환으로 패치하므로
# 업스트림이 해당 파일을 고치면 패치가 실패한다. 아래는 검증된 조합이다.
# 올릴 때는 커밋을 바꾸고 전체 설치를 다시 검증할 것.
TRELLIS2_COMMIT="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
FLEXGEMM_COMMIT="6dd94a859c26ee8246888502eada3dd8ad85532e"
CUMESH_COMMIT="12289e1062f0603f2f0d0771b02e1395d247f26f"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA_TAG="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --editable) EDITABLE=1; shift ;;
        --torch) TORCH_VER="$2"; shift 2 ;;
        --torchvision) TORCHVISION_VER="$2"; shift 2 ;;
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
    git clone https://github.com/microsoft/TRELLIS.2.git "$T2"
fi
echo "==> TRELLIS.2 커밋 고정: ${TRELLIS2_COMMIT:0:12}"
git -C "$T2" fetch --quiet origin "$TRELLIS2_COMMIT" 2>/dev/null || git -C "$T2" fetch --quiet origin
git -C "$T2" checkout --quiet "$TRELLIS2_COMMIT"
git -C "$T2" submodule update --init --recursive

if [[ ! -d "$T2/.venv" ]]; then
    echo "==> venv 생성"
    "$PY" -m venv "$T2/.venv"
fi
VPY="$T2/.venv/bin/python"
"$VPY" -m pip install -q --upgrade pip setuptools wheel

echo "==> torch 설치 ($CUDA_TAG, torch==$TORCH_VER)"
"$VPY" -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
    "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER"

# 확장들이 C++17 을 전제하므로 torch 2.13+ 는 쓸 수 없다.
"$VPY" - <<'PYCHK'
import sys, torch
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if (major, minor) >= (2, 13):
    sys.exit(
        f"\n[오류] torch {torch.__version__} 은 C++20 을 요구하는데 "
        "FlexGEMM / CuMesh / o-voxel 은 -std=c++17 로 빌드됩니다.\n"
        "       2.12 이하를 쓰세요:  bash scripts/setup_trellis2.sh --torch 2.10.0\n")
PYCHK

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
    case "$repo" in
        FlexGEMM) commit="$FLEXGEMM_COMMIT" ;;
        CuMesh)   commit="$CUMESH_COMMIT" ;;
    esac
    dir="$T2/extensions/$repo"
    [[ -d "$dir" ]] || git clone "https://github.com/JeffreyXiang/$repo.git" "$dir"
    git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$commit"
    git -C "$dir" submodule update --init --recursive
    echo "    $repo @ ${commit:0:12}"
    ( cd "$dir" && \
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
