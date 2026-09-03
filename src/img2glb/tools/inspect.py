"""GLB 의 메시/머티리얼 스펙 요약."""
from pathlib import Path

import numpy as np
import trimesh


def summarize(path) -> None:
    path = Path(path)
    scene = trimesh.load(str(path))
    geoms = (list(scene.geometry.values())
             if isinstance(scene, trimesh.Scene) else [scene])

    print(f"\n=== {path}  ({path.stat().st_size / 1e6:.2f} MB) ===")
    for i, g in enumerate(geoms):
        tag = f"[{i}] " if len(geoms) > 1 else ""
        print(f"{tag}faces {len(g.faces):,} | vertices {len(g.vertices):,} "
              f"| watertight {g.is_watertight}")
        print(f"{tag}bounds  {np.round(g.bounds[0], 3).tolist()} ~ "
              f"{np.round(g.bounds[1], 3).tolist()}")

        uv = getattr(g.visual, "uv", None)
        print(f"{tag}uv      {None if uv is None else tuple(uv.shape)}")

        mat = getattr(g.visual, "material", None)
        if mat is None:
            print(f"{tag}material 없음 (정점 컬러이거나 미지정)")
            continue
        print(f"{tag}material {type(mat).__name__}")
        for attr in ("baseColorTexture", "metallicRoughnessTexture",
                     "normalTexture", "emissiveTexture"):
            tex = getattr(mat, attr, None)
            print(f"{tag}  {attr:24s} {(tex.size, tex.mode) if tex else '-'}")

        mr = getattr(mat, "metallicRoughnessTexture", None)
        if mr is not None:
            a = np.asarray(mr.convert("RGB"))
            print(f"{tag}  roughness(G)  {a[..., 1].min()}~{a[..., 1].max()} "
                  f"(평균 {a[..., 1].mean():.1f})")
            print(f"{tag}  metallic (B)  {a[..., 2].min()}~{a[..., 2].max()} "
                  f"(평균 {a[..., 2].mean():.1f})")
