"""SPAR3D 백엔드.

stabilityai/stable-point-aware-3d 로 이미지 한 장에서 GLB 를 만든다.
TRELLIS.2 대비 훨씬 빠르고(수 초) 가벼우며, 중간 산출물로 점군을 내놓아
점군을 손봐 다시 넣는 재생성이 가능하다. 대신 형상 디테일은 떨어진다.

라이선스 관련 처리
------------------
* 모델 가중치는 **Stability AI Community License** 다. MIT 가 아니다.
  연구/비상업은 자유롭고, 연매출 100만 달러 미만 조직의 상업적 사용도
  허용되지만 그 이상은 별도 계약이 필요하다. 자세한 내용은 NOTICE 참고.
* 저장소가 기본 배경제거로 쓰는 ``transparent_background`` 는 설치하지 않는다.
  CLI 가 생성 전에 ``img2glb.bg`` (BiRefNet, MIT) 로 알파를 입혀 넘긴다.
  (``scripts/apply_patches.py --spar3d`` 가 해당 import 를 선택적으로 바꾼다)
"""
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from .base import Backend
from ..paths import find_model_root


class Spar3dBackend(Backend):
    name = "spar3d"
    description = ("stabilityai/stable-point-aware-3d (Stability AI Community "
                   "License). 수 초 만에 생성되고 점군도 함께 얻는다.")

    @staticmethod
    def add_arguments(parser):
        g = parser.add_argument_group("spar3d 옵션")
        g.add_argument("--spar3d-model-id", default="stabilityai/stable-point-aware-3d",
                       help="HuggingFace 모델 ID (gated - 접근 승인 필요)")
        g.add_argument("--spar3d-texture-resolution", type=int, default=1024,
                       help="텍스처 아틀라스 해상도")
        g.add_argument("--spar3d-foreground-ratio", type=float, default=1.3,
                       help="전경 크롭 여유 배율 (클수록 여백이 넓다)")
        g.add_argument("--spar3d-remesh", default="none",
                       choices=["none", "triangle", "quad"],
                       help="리메시 방식 (triangle=gpytoolbox, quad=pynanoinstantmeshes "
                            "설치 필요. aarch64 에서는 둘 다 빌드되지 않는다)")
        g.add_argument("--spar3d-target-count", type=int, default=2000,
                       help="리메시 목표 정점 수 (--spar3d-remesh 사용 시)")
        g.add_argument("--spar3d-low-vram", action="store_true",
                       help="모듈을 단계별로 올렸다 내려 VRAM 사용을 줄인다 (느려진다)")
        g.add_argument("--spar3d-save-points", action="store_true",
                       help="중간 산출물인 점군을 GLB 옆에 .ply 로 저장한다")

    def generate(self, image_path: Path, output_path: Path, args) -> Path:
        root = find_model_root("spar3d", "SPAR3D_ROOT")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import torch
        from PIL import Image

        from spar3d.system import SPAR3D
        from spar3d.utils import foreground_crop

        # remesh 는 선택 의존성(gpytoolbox / pynanoinstantmeshes)이 있어야 한다.
        if args.spar3d_remesh != "none":
            from spar3d.models.mesh import (QUAD_REMESH_AVAILABLE,
                                            TRIANGLE_REMESH_AVAILABLE)
            available = {"triangle": TRIANGLE_REMESH_AVAILABLE,
                         "quad": QUAD_REMESH_AVAILABLE}[args.spar3d_remesh]
            if not available:
                pkg = {"triangle": "gpytoolbox",
                       "quad": "pynanoinstantmeshes==0.0.3"}[args.spar3d_remesh]
                raise RuntimeError(
                    f"--spar3d-remesh {args.spar3d_remesh} 에는 {pkg} 가 필요합니다.\n"
                    f"  pip install {pkg}  (aarch64 에는 휠이 없어 빌드가 실패할 수 있습니다)")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        print(f"[env] torch {torch.__version__} / {dev_name}")

        image = Image.open(image_path).convert("RGBA")
        has_alpha = image.getchannel("A").getextrema()[0] < 255
        print(f"[prep] {image.size} (알파={has_alpha})")
        if not has_alpha:
            print("[warn] 알파 채널이 없습니다. 배경이 형상에 포함될 수 있습니다.")
        # 전경을 정사각형으로 크롭한다. SPAR3D 는 512x512 조건 이미지를 쓴다.
        image = foreground_crop(image, args.spar3d_foreground_ratio)

        t0 = time.time()
        model = SPAR3D.from_pretrained(
            args.spar3d_model_id,
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=args.spar3d_low_vram,
        )
        model.to(device)
        model.eval()
        print(f"[load] 완료 ({time.time() - t0:.1f}s)")

        # 원본 run.py 의 reduction_count_type=keep 에 해당한다 (-1 = 리메시 안 함)
        vertex_count = -1 if args.spar3d_remesh == "none" else args.spar3d_target_count

        print(f"[gen] texture_resolution={args.spar3d_texture_resolution} "
              f"remesh={args.spar3d_remesh} seed={args.seed}")
        t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # 점군 확산 샘플러가 전역 RNG 를 쓰므로 시드는 여기서 고정한다.
        torch.manual_seed(args.seed)
        with torch.no_grad():
            ctx = (torch.autocast(device_type=device, dtype=torch.bfloat16)
                   if device == "cuda" else nullcontext())
            with ctx:
                mesh, glob = model.run_image(
                    image,
                    bake_resolution=args.spar3d_texture_resolution,
                    remesh=args.spar3d_remesh,
                    vertex_count=vertex_count,
                    return_points=args.spar3d_save_points,
                )
        print(f"[gen] 완료 ({time.time() - t0:.1f}s)")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path), include_normals=True)

        if args.spar3d_save_points:
            points_path = output_path.with_suffix(".points.ply")
            glob["point_clouds"][0].export(str(points_path))
            print(f"[pts] {points_path}")

        if device == "cuda":
            print(f"[mem] peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return output_path
