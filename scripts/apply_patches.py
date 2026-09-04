#!/usr/bin/env python
"""TRELLIS.2 저장소에 img2glb 용 패치를 적용한다.

세 가지를 고친다.

1. **sparse attention 에 sdpa 경로 추가**
   aarch64 에는 flash-attn / xformers 휠이 없다. varlen 구조를 시퀀스별로
   잘라 torch 기본 SDPA 로 처리한다.

2. **nvdiffrast 제거**
   ``o_voxel.postprocess.to_glb`` 는 nvdiffrast 를 미분 렌더링이 아니라
   UV 공간 2D 래스터화에만 쓴다. NVIDIA 비상업 라이선스를 피하려고
   ``img2glb.raster`` (MIT) 로 대체한다.
   ``OVOXEL_RASTERIZER=nvdiffrast`` 로 원래 동작을 되돌릴 수 있다.

3. **transformers 5.x 호환**
   DINOv3 레이어 경로가 ``model.layer`` 에서 ``model.model.layer`` 로 바뀌었다.

o_voxel 은 site-packages 에 복사본으로 설치되므로 그쪽에도 반영한다.

멱등하게 동작하므로 여러 번 실행해도 안전하다.
"""
import argparse
import importlib.util
import io
import shutil
import sys
from pathlib import Path

SDPA_BRANCH = '''    elif config.ATTN == 'sdpa':
        # aarch64 에는 flash-attn / xformers 휠이 없어 torch 기본 SDPA 로 대체한다.
        # varlen 이므로 시퀀스별로 잘라 배치 1 로 돌린 뒤 다시 이어붙인다.
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        outs = []
        q_off = kv_off = 0
        for i in range(len(q_seqlen)):
            ql, kl = q_seqlen[i], kv_seqlen[i]
            qi = q[q_off:q_off + ql].transpose(0, 1).unsqueeze(0)
            ki = k[kv_off:kv_off + kl].transpose(0, 1).unsqueeze(0)
            vi = v[kv_off:kv_off + kl].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi)
            outs.append(oi.squeeze(0).transpose(0, 1))
            q_off += ql
            kv_off += kl
        out = torch.cat(outs, dim=0)
'''

RASTER_IMPORT = '''import os

# nvdiffrast 는 NVIDIA 비상업 라이선스라 기본 경로에서 제외한다.
# to_glb 는 UV 공간 2D 래스터화에만 쓰므로 MIT 구현으로 대체했다.
# OVOXEL_RASTERIZER=nvdiffrast 로 원본 동작을 되돌릴 수 있다 (A/B 비교용).
_RASTERIZER = os.environ.get("OVOXEL_RASTERIZER", "img2glb")
if _RASTERIZER == "nvdiffrast":
    import nvdiffrast.torch as dr
else:
    from img2glb import raster as dr
'''

RASTER_BLOCK_NEW = '''    # UV 공간 래스터화 -> 텍셀마다 (바리센트릭, 삼각형 id)
    if _RASTERIZER == "nvdiffrast":
        ctx = dr.RasterizeCudaContext()
        uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+100000],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i
            rast = torch.where(mask_chunk, rast_chunk, rast)
    else:
        rast = dr.rasterize(out_uvs, out_faces, [texture_size, texture_size])
'''


def _read(p):
    return io.open(p, encoding="utf-8").read()


def _write(p, s):
    io.open(p, "w", encoding="utf-8").write(s)


def _backup(p: Path):
    orig = p.with_suffix(p.suffix + ".orig")
    if not orig.exists():
        shutil.copy2(p, orig)


def patch_sparse_attention(root: Path) -> bool:
    cfg = root / "trellis2/modules/sparse/config.py"
    attn = root / "trellis2/modules/sparse/attention/full_attn.py"
    changed = False

    s = _read(cfg)
    old = "in ['xformers', 'flash_attn', 'flash_attn_3']:"
    if old in s:
        _backup(cfg)
        _write(cfg, s.replace(old, "in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']:", 1))
        changed = True

    s = _read(attn)
    if "config.ATTN == 'sdpa'" not in s:
        _backup(attn)
        marker = "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")"
        assert marker in s, "full_attn.py 구조가 예상과 다릅니다"
        s = s.replace(marker, SDPA_BRANCH + marker, 1)
        if "import torch.nn.functional as F" not in s:
            s = s.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n", 1)
        _write(attn, s)
        changed = True
    return changed


def patch_dinov3(root: Path) -> bool:
    p = root / "trellis2/modules/image_feature_extractor.py"
    s = _read(p)
    old = "        for i, layer_module in enumerate(self.model.layer):"
    if old not in s:
        return False
    _backup(p)
    new = ("        # transformers 5.x 에서 model.layer -> model.model.layer 로 변경됨\n"
           "        _layers = getattr(self.model, 'layer', None)\n"
           "        if _layers is None:\n"
           "            _layers = self.model.model.layer\n"
           "        for i, layer_module in enumerate(_layers):")
    _write(p, s.replace(old, new, 1))
    return True


def patch_ovoxel(root: Path) -> bool:
    p = root / "o-voxel/o_voxel/postprocess.py"
    s = _read(p)
    if "_RASTERIZER" in s:
        return False
    _backup(p)
    assert "import nvdiffrast.torch as dr\n" in s, "postprocess.py 구조가 예상과 다릅니다"
    s = s.replace("import nvdiffrast.torch as dr\n", RASTER_IMPORT, 1)

    start = s.index("    # Setup differentiable rasterizer context")
    end = s.index("    # Mask of valid pixels in texture")
    s = s[:start] + RASTER_BLOCK_NEW + "\n" + s[end:]
    _write(p, s)
    return True


def sync_installed_ovoxel(root: Path) -> bool:
    """site-packages 의 o_voxel 복사본에도 반영한다.

    ``import o_voxel`` 로 경로를 찾으면 안 된다. 패치 전 복사본은 최상단에서
    nvdiffrast 를 import 하므로, 바로 그 이유로 import 가 실패해 동기화가
    조용히 건너뛰어진다. find_spec 은 모듈을 실행하지 않고 위치만 알려준다.
    """
    spec = importlib.util.find_spec("o_voxel")
    if spec is None or not spec.origin:
        print("  o_voxel 이 설치되어 있지 않아 동기화를 건너뜁니다.")
        return False
    dst = Path(spec.origin).parent / "postprocess.py"
    src = root / "o-voxel/o_voxel/postprocess.py"
    if not dst.is_file() or dst.resolve() == src.resolve():
        return False
    if _read(dst) == _read(src):
        return False
    _backup(dst)
    shutil.copy2(src, dst)
    return True


def main():
    ap = argparse.ArgumentParser(description="TRELLIS.2 저장소 패치")
    ap.add_argument("--trellis2", required=True, type=Path)
    args = ap.parse_args()

    root = args.trellis2
    if not (root / "trellis2").is_dir():
        sys.exit(f"TRELLIS.2 저장소가 아닙니다: {root}")
    for label, fn in (("sparse attention (sdpa)", patch_sparse_attention),
                      ("DINOv3 (transformers 5.x)", patch_dinov3),
                      ("o_voxel (nvdiffrast 제거)", patch_ovoxel),
                      ("o_voxel 설치본 동기화", sync_installed_ovoxel)):
        done = fn(root)
        print(f"  [{'적용' if done else '이미 적용됨'}] {label}")

    print("패치 완료.")


if __name__ == "__main__":
    main()
