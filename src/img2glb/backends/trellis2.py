"""TRELLIS.2 백엔드.

microsoft/TRELLIS.2-4B 로 이미지 한 장에서 PBR 텍스처가 입혀진 GLB 를 만든다.

라이선스 관련 처리
------------------
* TRELLIS.2 의 ``pipeline.json`` 은 배경제거 모델로 ``briaai/RMBG-2.0`` 을
  지정하는데 **CC BY-NC 4.0(비상업)** 이고 gated 다. 본 백엔드는 이를
  **항상 차단한다**. 대신 CLI 가 생성 전에 ``img2glb.bg`` (BiRefNet, MIT) 로
  알파를 입혀 넘기므로 파이프라인의 rembg 경로는 실행되지 않는다.
  (``preprocess_image`` 의 has_alpha 분기)
* ``o_voxel.postprocess`` 의 nvdiffrast 의존은 ``img2glb.raster`` 로 대체한다.
  자세한 내용은 저장소 README 참고.
"""
import os
import sys
import time
from pathlib import Path

from .base import Backend
from ..paths import find_model_root


class _NoRembg:
    """RMBG-2.0 로드를 막는 더미. 알파 채널 있는 입력에서만 안전하다."""

    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def __call__(self, *a, **k):
        raise RuntimeError(
            "TRELLIS.2 내장 배경제거(briaai/RMBG-2.0) 는 CC BY-NC 4.0(비상업)"
            " 이라 사용하지 않습니다.\n"
            "  알파 채널이 있는 이미지를 넣거나, --no-remove-bg 를 빼고 실행하세요"
            " (img2glb 가 BiRefNet(MIT) 으로 알아서 처리합니다)."
        )


class Trellis2Backend(Backend):
    name = "trellis2"
    description = "microsoft/TRELLIS.2-4B (MIT). 형상 품질이 좋고 VRAM 사용이 적다."

    @staticmethod
    def add_arguments(parser):
        g = parser.add_argument_group("trellis2 옵션")
        g.add_argument("--model-id", default="microsoft/TRELLIS.2-4B",
                       help="HuggingFace 모델 ID")
        g.add_argument("--pipeline-type", default="1024_cascade",
                       choices=["512", "1024", "1024_cascade", "1536_cascade"],
                       help="복셀 해상도 (기본 1024_cascade 권장)")
        g.add_argument("--steps", type=int, default=12,
                       help="샘플러 step 수 (기본 12 권장; 올리면 텍스처가 나빠진다)")
        g.add_argument("--shape-guidance", type=float, default=7.5,
                       help="형상 guidance strength")
        g.add_argument("--texture-size", type=int, default=4096,
                       help="출력 텍스처 해상도")
        g.add_argument("--decimation-target", type=int, default=200000,
                       help="최종 메시 목표 면 수")
        g.add_argument("--max-num-tokens", type=int, default=49152)

    def generate(self, image_path: Path, output_path: Path, args) -> Path:
        root = find_model_root("trellis2", "TRELLIS2_ROOT")

        # aarch64 에는 flash-attn/xformers 휠이 없어 torch 기본 SDPA 를 쓴다.
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import torch
        from PIL import Image

        # RMBG-2.0(CC BY-NC) 로드를 원천 차단한다. 배경 제거는 CLI 가
        # img2glb.bg (BiRefNet, MIT) 로 미리 처리해 알파를 넣어 넘긴다.
        import trellis2.pipelines.rembg as _rembg
        for n in [n for n in dir(_rembg) if not n.startswith("_")]:
            if isinstance(getattr(_rembg, n), type):
                setattr(_rembg, n, _NoRembg)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel

        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[env] torch {torch.__version__} / {dev}")

        image = Image.open(image_path)
        has_alpha = (image.mode in ("RGBA", "LA")
                     and image.getchannel("A").getextrema()[0] < 255)
        print(f"[prep] {image.size} {image.mode} (알파={has_alpha})")
        if not has_alpha:
            print("[warn] 알파 채널이 없습니다. 배경이 형상에 포함될 수 있습니다.")

        t0 = time.time()
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_id)
        pipeline.cuda()
        load_sec = time.time() - t0
        print(f"[load] 완료 ({load_sec:.1f}s)")

        sampler = {"steps": args.steps, "guidance_strength": args.shape_guidance}
        print(f"[gen] pipeline_type={args.pipeline_type} steps={args.steps}")
        t0 = time.time()
        torch.manual_seed(args.seed)
        mesh = pipeline.run(
            image, seed=args.seed,
            pipeline_type=args.pipeline_type,
            max_num_tokens=args.max_num_tokens,
            sparse_structure_sampler_params=sampler,
            shape_slat_sampler_params=sampler,
            tex_slat_sampler_params={"steps": args.steps},
        )[0]
        gen_sec = time.time() - t0
        print(f"[gen] 완료 ({gen_sec:.1f}s)")

        mesh.simplify(16777216)   # 래스터라이저 인덱스 한계

        t0 = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces,
            attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            remesh=True, remesh_band=1, remesh_project=0,
            verbose=args.verbose,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(output_path))
        glb_sec = time.time() - t0
        print(f"[glb] 완료 ({glb_sec:.1f}s)")

        peak = None
        if torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print(f"[mem] peak GPU {peak} GB")

        self.stats = {
            "backend": self.name,
            "model_id": args.model_id,
            "seed": args.seed,
            "params": {
                "pipeline_type": args.pipeline_type,
                "steps": args.steps,
                "shape_guidance": args.shape_guidance,
                "texture_size": args.texture_size,
                "decimation_target": args.decimation_target,
                "max_num_tokens": args.max_num_tokens,
            },
            "durations_sec": {
                "load": round(load_sec, 1),
                "generate": round(gen_sec, 1),
                "export": round(glb_sec, 1),
            },
            "peak_gpu_gb": peak,
        }
        return output_path
