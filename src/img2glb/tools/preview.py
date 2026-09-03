"""GLB 미리보기 렌더.

픽셀마다 UV 를 보간해 텍스처를 바이리니어 샘플링한다. 정점 컬러로 근사하지 않으므로
실제 텍스처 화질이 그대로 보인다. 래스터화는 ``img2glb.raster`` (MIT) 를 쓴다.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from ..raster import interpolate, rasterize_mesh


def _load_mesh(glb_path):
    scene = trimesh.load(str(glb_path))
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        if not geoms:
            raise ValueError(f"지오메트리가 없습니다: {glb_path}")
        mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(tuple(geoms))
    else:
        mesh = scene
    return mesh


def _project(vertices, elev, azim, resolution, margin=0.92):
    """정규 직교 투영. 반환값의 x,y 는 픽셀 좌표, z 는 0 이상(작을수록 가까움)."""
    v = vertices - vertices.mean(dim=0, keepdim=True)
    v = v / v.abs().max().clamp(min=1e-8)

    e, a = math.radians(elev), math.radians(azim)
    ry = torch.tensor([[math.cos(a), 0, math.sin(a)],
                       [0, 1, 0],
                       [-math.sin(a), 0, math.cos(a)]], dtype=v.dtype, device=v.device)
    rx = torch.tensor([[1, 0, 0],
                       [0, math.cos(e), -math.sin(e)],
                       [0, math.sin(e), math.cos(e)]], dtype=v.dtype, device=v.device)
    v = v @ ry.T @ rx.T

    H = W = resolution
    x = (v[:, 0] * 0.5 * margin + 0.5) * W
    y = (0.5 - v[:, 1] * 0.5 * margin) * H       # 화면 위쪽이 +y
    z = v[:, 2] - v[:, 2].min() + 1e-3           # 0 이상으로 이동
    return torch.stack([x, y, z], dim=-1)


def render(glb_path, output_path, resolution=768, views=(0, 90, 180, 270),
           elev=0.0, background=(255, 255, 255), device="cuda"):
    """GLB 를 여러 각도에서 렌더해 한 장의 PNG 로 저장한다."""
    mesh = _load_mesh(glb_path)
    verts = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)

    material = getattr(mesh.visual, "material", None)
    tex_img = getattr(material, "baseColorTexture", None) if material else None
    uv = getattr(mesh.visual, "uv", None)

    tex = None
    if tex_img is not None and uv is not None:
        tex_np = np.asarray(tex_img.convert("RGB")).astype(np.float32) / 255.0
        tex = torch.as_tensor(tex_np, device=device).permute(2, 0, 1).unsqueeze(0)
        uv = torch.as_tensor(np.asarray(uv), dtype=torch.float32, device=device)

    print(f"[in] faces={len(mesh.faces)} vertices={len(mesh.vertices)} "
          f"texture={tex_img.size if tex_img else None}")

    bg = torch.tensor([c / 255.0 for c in background], device=device)
    tiles = []
    for azim in views:
        xyz = _project(verts, elev, azim, resolution)
        rast = rasterize_mesh(xyz, faces, resolution)
        visible = (rast[0, ..., 3:4] > 0).float()

        if tex is not None:
            uv_map, _ = interpolate(uv.unsqueeze(0), rast, faces)
            # trimesh 는 glTF 임포트 시 V 축을 뒤집어 OBJ 규약(좌하단 원점)으로 준다.
            # PIL 이미지는 행 0 이 위쪽이므로 되돌려서 샘플링한다.
            grid = torch.stack([uv_map[0, ..., 0], 1.0 - uv_map[0, ..., 1]], dim=-1)
            grid = (grid * 2.0 - 1.0).unsqueeze(0)
            color = F.grid_sample(tex, grid, mode="bilinear",
                                  padding_mode="border", align_corners=False)
            color = color[0].permute(1, 2, 0)
        else:
            # 텍스처가 없으면 깊이 음영으로 형태만 보여준다
            d = rast[0, ..., 2:3]
            dm = d[visible > 0]
            norm = (d - dm.min()) / (dm.max() - dm.min() + 1e-8) if dm.numel() else d
            color = (1.0 - norm).repeat(1, 1, 3) * 0.75 + 0.15

        color = color * visible + bg * (1 - visible)
        tiles.append(Image.fromarray((color.clamp(0, 1) * 255).byte().cpu().numpy()))

    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (resolution * cols, resolution * rows), background)
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % cols) * resolution, (i // cols) * resolution))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"[out] {output_path}  ({canvas.size[0]}x{canvas.size[1]})")
    return output_path
