# img2glb

이미지 한 장에서 **PBR 텍스처가 입혀진 GLB** 를 생성하는 CLI.

```bash
img2glb --model trellis2 --image cat.png --output cat.glb
img2glb --model spar3d   --image cat.png --output cat.glb   # 빠른 쪽
```

백엔드는 두 개다.

| 백엔드 | 모델 | 라이선스 | 생성 시간 | 특징 |
|---|---|---|---|---|
| `trellis2` (기본) | [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2) | MIT | 75–98 s | 형상·텍스처 품질이 좋다. 20만 면 + 4096² PBR |
| `spar3d` | [SPAR3D](https://github.com/Stability-AI/stable-point-aware-3d) | **Stability AI Community** | 3–4 s | 20배 빠르고 가볍다. 1만 면 + 1024². 점군도 함께 나온다 |

> **라이선스 주의**: `spar3d` 는 MIT 가 **아니다**. 연구·비상업과 연매출
> 100만 달러 미만 조직의 상업적 사용까지만 무상이다. 제약 없는 상업 배포가
> 필요하면 `trellis2` 를 쓸 것. 자세한 내용은 [NOTICE](NOTICE) 참조.

GLB 는 baseColor + metallicRoughness 텍스처를 포함해 Unreal / Unity 의
PBR 머티리얼에 그대로 매핑된다 (`spar3d` 는 baseColor + normal, metallic /
roughness 는 텍스처 대신 머티리얼 스칼라로 추정해 넣는다).

## 특징

- **상업 배포 가능한 라이선스 구성** (`trellis2` 경로) — 원본 TRELLIS.2 는 GLB 생성 경로에서
  `nvdiffrast`(NVIDIA 비상업) 를 쓴다. 본 저장소는 이를 자체 구현
  ([`src/img2glb/raster`](src/img2glb/raster), MIT) 으로 대체했다.
  자세한 내용은 [NOTICE](NOTICE) 참조. (`spar3d` 는 모델 자체가
  Stability AI Community License 라 매출 조건이 붙는다)
- **aarch64 / Blackwell 지원** — flash-attn·xformers 휠이 없는 환경을 위해
  sparse attention 에 SDPA 경로를 추가한다.
- **배경 자동 제거** — 알파 채널이 없는 입력이면 생성 전에 알아서 제거한다.
  BiRefNet(MIT) 을 쓰며, TRELLIS.2 가 기본으로 지정하는
  RMBG-2.0(CC BY-NC, gated) 은 **사용을 차단**했다.
  단독 실행(`remove-bg`) 도 가능하다.
- 외부 렌더러 없이 결과를 확인하는 `preview` / `inspect` 내장.

## 설치

CUDA GPU 와 Python 3.10+ 가 필요하다.

백엔드마다 의존성 핀이 충돌해서 **venv 를 따로 쓴다**. 필요한 쪽만 깔면 된다.

```bash
git clone https://github.com/KETI/img2glb.git
cd img2glb

bash scripts/setup_trellis2.sh          # 저장소 clone + 확장 빌드 + 패치
source models/trellis2/.venv/bin/activate

bash scripts/setup_spar3d.sh            # 또는 이쪽
source models/spar3d/.venv/bin/activate
```

두 스크립트 모두 CUDA 태그와 아키텍처를 자동 판별한다. 필요하면 지정한다.

```bash
bash scripts/setup_trellis2.sh --cuda cu130 --arch 12.1
```

### gated 모델 접근 승인

두 백엔드 모두 gated 모델을 쓴다. 최초 1회 필요하다.

| 백엔드 | 승인이 필요한 저장소 |
|---|---|
| `trellis2` | https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m (이미지 인코더) |
| `spar3d` | https://huggingface.co/stabilityai/stable-point-aware-3d (모델 가중치) |

승인 후 `hf auth login` 으로 토큰을 등록한다.

> `trellis2`: 제품에 내장할 경우 **"Built with DINOv3" 표기가 의무**다.
> `spar3d`: 배포 시 **"Powered by Stability AI" 표기와 라이선스 사본 첨부가
> 의무**이며 매출 조건이 붙는다. [NOTICE](NOTICE) 참조.

### root 권한이 없는 환경

CUDA 확장 빌드에 Python 개발 헤더가 필요하다.
`setup_*.sh` 가 헤더가 없으면 deb 를 받아 저장소 루트의 `.localdev/` 에
풀어 두고, `img2glb` 도 이 경로를 자동으로 찾는다. 수동으로 하려면:

```bash
apt-get download libpython3.12-dev python3.12-dev     # root 불필요
dpkg-deb -x libpython3.12-dev_*.deb .localdev/root
dpkg-deb -x python3.12-dev_*.deb .localdev/root
export PYDEV_INCLUDE=$PWD/.localdev/root/usr/include
```

## 사용법

```bash
# 생성 (--output 생략 시 output/glb/<입력이름>.glb)
img2glb --model trellis2 --image samples/bird.png
img2glb --model spar3d   --image samples/bird.png
img2glb -i samples/bird.png -o out/bird.glb --seed 7 --preview

# spar3d: 중간 산출물인 점군도 함께 저장 (out/bird.points.ply)
img2glb --model spar3d -i samples/bird.png -o out/bird.glb --spar3d-save-points

# 배경 제거만 단독 실행
img2glb remove-bg photo.jpg -o photo_rgba.png
img2glb remove-bg photo.jpg --method u2net --device cpu

# 확인 (--output 생략 시 output/render/<이름>_preview.png)
img2glb preview out/bird.glb --resolution 1024
img2glb preview out/bird.glb --views 0,45,90 --elev 20

# 스펙 출력
img2glb inspect out/bird.glb

# 백엔드 목록
img2glb backends
```

### 주요 옵션

공통:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--model` | `trellis2` | 백엔드 (`trellis2` / `spar3d`) |
| `--seed` | 2025 | 랜덤 시드 |
| `--no-remove-bg` | off | 배경 자동 제거를 끈다 (기본은 켜짐) |
| `--keep-rgba` | off | 배경 제거 결과를 `output/rgba/` 에 남긴다 |
| `--bg-method` | `birefnet` | 배경제거 방식 (`birefnet` / `u2net`) |

`--model trellis2`:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--pipeline-type` | `1024_cascade` | 복셀 해상도 |
| `--steps` | 12 | 샘플러 step 수 |
| `--texture-size` | 4096 | 출력 텍스처 해상도 |
| `--decimation-target` | 200000 | 최종 메시 목표 면 수 |
| `--use-rembg` | off | TRELLIS.2 내장 RMBG-2.0 사용 (**비상업 라이선스**) |

`--model spar3d`:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--spar3d-texture-resolution` | 1024 | 텍스처 아틀라스 해상도 |
| `--spar3d-foreground-ratio` | 1.3 | 전경 크롭 여유 배율 (클수록 여백이 넓다) |
| `--spar3d-save-points` | off | 중간 산출물 점군을 `.points.ply` 로 저장 |
| `--spar3d-low-vram` | off | 모듈을 단계별로 올렸다 내려 VRAM 절약 (느려진다) |
| `--spar3d-remesh` | `none` | 리메시 (`triangle`/`quad`, 아래 참고) |
| `--spar3d-target-count` | 2000 | 리메시 목표 정점 수 |

> **spar3d 의 metallic 값 주의**: metallic / roughness 는 텍스처가 아니라
> 머티리얼 스칼라로 나오는데, metallic 헤드가 베타분포의 **mode** 를 쓰는 탓에
> 0 또는 1 로 튄다 (실측: bird=1.0, human=0.0, doorhandle=0.0).
> 금속이 아닌 물체가 `metallicFactor=1.0` 로 나오면 엔진에서 거울처럼 보이므로
> 임포트 후 확인하고 필요하면 0 으로 덮어쓸 것.

> `--spar3d-remesh` 는 `gpytoolbox` / `pynanoinstantmeshes` 가 필요한데
> **aarch64 에는 휠이 없고 소스 빌드도 실패**한다 (gpytoolbox sdist 에
> CMakeLists.txt 가 빠져 있다). 그래서 기본 설치에서 제외했고, 지정하면
> 명시적인 오류로 알려준다. 면 수를 줄여야 하면 후처리로 데시메이션할 것.

**튜닝 요령**: `--pipeline-type` 과 `--steps` 는 기본값을 권장한다. 실측상
`1536_cascade` 나 `--steps 25` 는 형상은 그대로인데 텍스처가 나빠진다
(로드하는 모델이 `1024_cascade` 와 동일하다). 결과가 마음에 들지 않으면
**`--seed` 를 바꾸는 편이 효과적**이다.

### 입력 이미지

단일 객체 · 정면 · 정사각형에 가까운 비율을 권장한다.
**배경 제거는 자동이다** — 알파 채널이 있으면 그대로 쓰고, 없으면 생성 직전에
제거한 뒤 넘긴다. 중간 RGBA 는 임시파일로만 쓰고 지운다
(`--keep-rgba` 로 남길 수 있다).

| 방식 | 라이선스 | 특징 |
|---|---|---|
| `birefnet` (기본) | MIT (ZhengPeng7/BiRefNet) | 품질 우선. GPU 권장 |
| `u2net` | MIT + Apache-2.0 | 가볍고 CPU 가능. `pip install "rembg[cpu]"` 필요 |

> TRELLIS.2 는 `pipeline.json` 에서 배경제거 모델로 `briaai/RMBG-2.0` 을
> 지정하는데 **CC BY-NC 4.0(비상업)** 이고 gated 다. 본 CLI 는 이를
> **로드 자체를 차단**하며 우회 옵션도 두지 않았다.
> 참고로 TRELLIS.2 의 래퍼 코드는 원래 MIT 인 `ZhengPeng7/BiRefNet` 을
> 기본값으로 두고 있고, 설정 파일이 이를 덮어쓰는 구조다.

## 성능 (NVIDIA GB10, 1024×1024 입력)

`--model trellis2`

| 단계 | 시간 | peak GPU |
|---|---|---|
| 모델 로드 | ~60 s | — |
| 생성 | 75–98 s | 3.0–3.8 GB |
| GLB 변환 | 23–34 s | |

출력은 약 20만 면 + 4096² baseColor(RGBA) + 4096² metallicRoughness.

`--model spar3d`

| 단계 | 시간 | peak GPU |
|---|---|---|
| 모델 로드 | 10–13 s | — |
| 생성 + 베이킹 | 3–4 s | 10.7 GB |

출력은 약 1만 면 + 1024² baseColor + 1024² normal.
배경 제거까지 포함한 전체 실행이 20초 안쪽이라 반복 시도에 적합하다.

## 구조

```
img2glb/
├── src/img2glb/
│   ├── cli.py            # 명령줄 진입점
│   ├── backends/         # 모델 백엔드 (trellis2, spar3d)
│   ├── bg/               # 배경 제거 (birefnet / u2net)
│   ├── raster/           # 자체 래스터라이저 (CUDA, MIT)
│   └── tools/            # preview / inspect
├── scripts/
│   ├── setup_trellis2.sh # 백엔드 설치
│   ├── setup_spar3d.sh   #   〃
│   └── apply_patches.py  # 백엔드 저장소 패치 (멱등)
├── samples/
├── models/               # 백엔드 저장소 (gitignore)
└── output/               # glb/ , render/ , rgba/ (gitignore)
```

### 백엔드 추가

`Backend` 를 상속하고 `backends/__init__.py` 의 `BACKENDS` 에 등록하면
CLI 에 자동 노출된다.

```python
class MyBackend(Backend):
    name = "mymodel"
    description = "..."

    @staticmethod
    def add_arguments(parser): ...

    def generate(self, image_path, output_path, args) -> Path: ...
```

## 적용된 패치

`scripts/apply_patches.py` 가 백엔드 저장소에 아래를 적용한다. 멱등이며
원본은 `*.orig` 로 백업된다.

`--trellis2`

1. **sparse attention SDPA 경로** — aarch64 에 flash-attn/xformers 휠이 없다.
2. **nvdiffrast 제거** — `o_voxel.postprocess` 를 `img2glb.raster` 로 전환.
3. **transformers 5.x 호환** — DINOv3 레이어 경로 변경 대응.

`--spar3d`

1. **transparent_background 의존 제거** — `spar3d/utils.py` 가 모듈 최상단에서
   `Remover` 를 import 한다. 배경 제거는 백엔드와 무관하게 `img2glb.bg`
   (BiRefNet, MIT) 가 담당하므로 이 패키지를 설치하지 않고 import 를
   선택적으로 바꾼다.

> `o_voxel` 은 site-packages 에 복사본으로 설치되므로 재설치하면 패치가 사라진다.
> 그때는 `apply_patches.py` 를 다시 실행하면 된다.

### spar3d 에서 핀을 따르지 않는 의존성

저장소 `requirements.txt` 의 핀(`numpy==1.26.4`, `transformers==4.42.3`,
`trimesh==4.4.1` 등)은 낡아서 aarch64 + torch cu130 조합과 맞지 않는다.
`setup_spar3d.sh` 는 최신 버전을 쓰되 아래 두 가지만 제약한다.

- `transformers<5` — spar3d 가 복사해 둔 DINOv2 구현이 5.x 에서 사라진
  내부 API(`transformers.pytorch_utils`, `utils.backbone_utils`)를 쓴다.
- `setuptools<81` — AlphaCLIP 의 `setup.py` 가 `pkg_resources` 를 쓴다.
  81 에서 제거되어 빌드가 깨진다. 설치도 `--no-build-isolation` 이 필요하다.

## 문제 해결

### 실행이 멈춘 채 진행되지 않는다

CUDA 확장을 런타임 JIT 로 컴파일하는 경우, 빌드 디렉터리의 `lock` 파일이
남으면 torch 의 JIT 로더가 **파일이 사라질 때까지 무한 대기**한다.
빌드 중 프로세스가 죽으면(Ctrl+C 등) 이후 모든 실행이 멈춘다.

정상 설치라면 확장이 **설치 시점에 컴파일**되어 JIT 를 타지 않는다. 확인:

```bash
python -c "from img2glb.raster import _get_ext; print(_get_ext().__file__)"
# .../img2glb/_raster_ext...so   -> 설치 시 빌드 (정상)
```

JIT 폴백을 쓰는 상태라면 해당 venv 에서 다시 설치한다.

```bash
pip install . --no-build-isolation
```

`--no-build-isolation` 이 없으면 setup.py 가 torch 를 못 찾아 확장 빌드를 건너뛴다.

> **백엔드마다 venv 가 따로일 때는 일반 설치(`pip install .`) 를 쓴다.**
> `-e`(editable) 는 여러 venv 가 하나의 소스 트리를 공유하게 되는데, 확장은
> 소스 트리에 한 벌만 빌드되므로 torch 버전이 다른 venv 끼리 충돌한다.
> 이 경우 나중에 빌드한 쪽만 prebuilt 를 쓰고 나머지는 JIT 로 폴백한다
> (빌드 당시 torch 버전을 기록해 두고 런타임에 대조한다).
> 개발용으로 editable 이 필요하면 `setup_*.sh --editable` 을 쓰되 venv 하나만
> 사용할 것.

300초 이상 된 lock 은 자동 제거된다(`IMG2GLB_LOCK_TIMEOUT` 으로 조정).
즉시 지우려면:

```bash
rm -rf ~/.cache/torch_extensions/*/img2glb_raster
```

### `img2glb: command not found`

venv 폴더를 옮기면 `activate` 안의 `VIRTUAL_ENV` 와 콘솔 스크립트 shebang 이
옛 경로를 가리킨다. `setup_trellis2.sh` 를 다시 돌리면 자동으로 고쳐진다.
activate 없이 절대경로로 실행해도 된다.

```bash
models/trellis2/.venv/bin/img2glb --image samples/bird.png
```

### `Python.h: No such file or directory`

Python 개발 헤더가 없다. [root 권한이 없는 환경](#root-권한이-없는-환경) 참고.

### `Access to model ... is restricted` (403)

gated 모델 접근 승인이 필요하다. `hf auth login` 만으로는 부족하고
모델 페이지에서 별도 요청·승인이 있어야 한다.

## 라이선스

MIT ([LICENSE](LICENSE)). 제3자 구성요소 조건은 [NOTICE](NOTICE) 를 반드시 확인할 것.
