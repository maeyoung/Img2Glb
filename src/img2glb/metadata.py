"""에셋 메타데이터 생성.

GLB 옆에 ``<이름>.json`` 사이드카를 만든다. 에셋 관리 웹에서 목록을 보여주고
"이 에셋을 쓸지" 판단하는 데 필요한 값만 담는다.

설계 원칙
---------
* **표준 명칭** — 일반 메타데이터는 schema.org(CreativeWork/MediaObject/3DModel),
  머티리얼 속성은 glTF 2.0 스펙의 이름을 그대로 쓴다. camelCase 로 통일한다.
* **평탄한 구조** — 중첩이 없어 DB 컬럼에 그대로 매핑된다.
* **측정값만** — 임계값으로 분류하거나 추정한 값은 넣지 않는다.
  (예: "high-poly" 같은 등급, 압축 방식에 좌우되는 VRAM 추정치)
  등급이 필요하면 웹에서 ``triangleCount`` 같은 수치로 판단한다.
* **설명은 사용자 몫** — 자동 생성하지 않는다. 없으면 빈 문자열이다.
"""
import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

SCHEMA_VERSION = "1.0"
GLB_MIME = "model/gltf-binary"      # IANA 등록 미디어 타입


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _gltf_json(glb_path):
    """GLB 컨테이너에서 glTF JSON 청크를 꺼낸다."""
    with open(glb_path, "rb") as f:
        if f.read(4) != b"glTF":
            return {}
        f.read(8)                                   # version, length
        length, _kind = struct.unpack("<II", f.read(8))
        return json.loads(f.read(length))


def _surface_samples(mesh, tex):
    """정점 UV 위치에서 텍스처를 샘플한다.

    UV 아틀라스 전체를 평균내면 차트 사이의 빈 공간까지 섞여 실제 표면 값과
    달라진다. 실제 표면에 대응하는 정점 UV 에서만 읽는다.
    """
    uv = getattr(mesh.visual, "uv", None)
    if tex is None or uv is None or len(uv) == 0:
        return None
    img = np.asarray(tex)
    h, w = img.shape[:2]
    uv = np.asarray(uv)
    # trimesh 는 glTF 임포트 시 V 를 뒤집어 준다 -> 이미지 좌표로 되돌린다
    xs = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
    ys = np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
    return img[ys, xs]


def _dominant_colors(mesh, tex, n=3):
    """표면에서 가장 많이 나타나는 색을 hex 로 반환한다.

    채널당 8단계로 양자화해 최빈값을 고른다. 색 이름은 붙이지 않는다
    (경계가 자의적이라 오분류가 생긴다).
    """
    samples = _surface_samples(mesh, tex.convert("RGB") if tex else None)
    if samples is None:
        return []
    q = (samples // 32) * 32 + 16
    uniq, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)[:n]
    return ["#%02x%02x%02x" % tuple(int(v) for v in uniq[i]) for i in order]


def _keywords(material, backend, extra):
    """사실만 담은 태그. 임계값 기반 등급은 넣지 않는다."""
    tags = list(extra or [])
    if material["baseColorTextureSize"]:
        tags.append("textured")
    if material["metallicRoughnessTextureSize"]:
        tags.append("pbr")
    if material["normalTextureSize"]:
        tags.append("normal-map")
    if material["alphaMode"] and material["alphaMode"] != "OPAQUE":
        tags.append("transparent")
    if material["doubleSided"]:
        tags.append("double-sided")
    if backend:
        tags.append(backend)
    return sorted(set(tags))


def _material(mesh, gltf):
    """머티리얼 정보. 키 이름은 glTF 2.0 스펙을 따른다."""
    mats = gltf.get("materials") or [{}]
    g = mats[0]
    pbr = g.get("pbrMetallicRoughness", {})

    vis_mat = getattr(mesh.visual, "material", None)
    base = getattr(vis_mat, "baseColorTexture", None) if vis_mat else None
    mr = getattr(vis_mat, "metallicRoughnessTexture", None) if vis_mat else None
    nrm = getattr(vis_mat, "normalTexture", None) if vis_mat else None

    info = {
        # glTF 스펙 이름 그대로
        "alphaMode": g.get("alphaMode", "OPAQUE"),
        "doubleSided": bool(g.get("doubleSided", False)),
        "metallicFactor": pbr.get("metallicFactor"),
        "roughnessFactor": pbr.get("roughnessFactor"),
        "baseColorTextureSize": base.size[0] if base is not None else None,
        "metallicRoughnessTextureSize": mr.size[0] if mr is not None else None,
        "normalTextureSize": nrm.size[0] if nrm is not None else None,
        # 표면에서 실측한 평균. Factor(배율) 와는 다른 값이라 이름을 구분한다.
        "metallicAverage": None,
        "roughnessAverage": None,
    }
    if mr is not None:
        s = _surface_samples(mesh, mr.convert("RGB"))
        if s is not None:
            info["roughnessAverage"] = round(float(s[..., 1].mean() / 255), 3)
            info["metallicAverage"] = round(float(s[..., 2].mean() / 255), 3)
    return info


def build(glb_path, *, source_image=None, generation=None, description=None,
          tags=None, name=None, preview_image=None):
    """GLB 를 분석해 메타데이터 딕셔너리를 만든다."""
    glb_path = Path(glb_path).resolve()
    generation = dict(generation or {})
    gltf = _gltf_json(glb_path)

    scene = trimesh.load(str(glb_path))
    geoms = (list(scene.geometry.values())
             if isinstance(scene, trimesh.Scene) else [scene])
    mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(tuple(geoms))

    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)
    material = _material(mesh, gltf)

    vis_mat = getattr(mesh.visual, "material", None)
    base_tex = getattr(vis_mat, "baseColorTexture", None) if vis_mat else None

    # 정확히 퇴화한 삼각형만 센다 (같은 정점을 두 번 참조). 임계값을 쓰지 않는다.
    degenerate = int((
        (faces[:, 0] == faces[:, 1]) |
        (faces[:, 1] == faces[:, 2]) |
        (faces[:, 0] == faces[:, 2])
    ).sum())

    dims = mesh.bounds[1] - mesh.bounds[0]
    sha = _sha256(glb_path)
    src = Path(source_image).resolve() if source_image else None
    prev = Path(preview_image).resolve() if preview_image else None

    return {
        "schemaVersion": SCHEMA_VERSION,

        # --- schema.org (CreativeWork / MediaObject / 3DModel) ---
        "identifier": sha[:16],
        "name": name or glb_path.stem,
        "description": description or "",
        "keywords": _keywords(material, generation.get("backend"), tags),
        "dateCreated": datetime.now(timezone.utc)
                               .astimezone().isoformat(timespec="seconds"),
        "contentUrl": glb_path.name,
        "contentSize": glb_path.stat().st_size,          # 바이트
        "encodingFormat": GLB_MIME,
        # JSON 사이드카 위치 기준 상대경로. 보통 같은 폴더라 파일명만 나온다.
        "thumbnailUrl": (os.path.relpath(prev, glb_path.parent)
                         if prev and prev.is_file() else None),
        "sha256": sha,

        # --- 지오메트리 (실측) ---
        "triangleCount": int(len(faces)),
        "vertexCount": int(len(verts)),
        "degenerateTriangleCount": degenerate,
        "isWatertight": bool(mesh.is_watertight),
        "hasUv": getattr(mesh.visual, "uv", None) is not None,
        "hasNormals": bool(getattr(mesh, "vertex_normals", None) is not None),
        "surfaceArea": round(float(mesh.area), 4),
        "boundingSphereRadius": round(
            float(np.linalg.norm(verts - verts.mean(axis=0), axis=1).max()), 4),
        # 생성 모델이 단위 큐브에 맞춰 출력하므로 실제 치수가 아니라 비율이다
        "dimensionsNormalized": [round(float(v), 3) for v in dims],

        # --- 드로우콜 판단 (glTF 구조 실측) ---
        "primitiveCount": sum(len(m.get("primitives", []))
                              for m in gltf.get("meshes", [])),
        "materialCount": len(gltf.get("materials", [])),
        "textureCount": len(gltf.get("textures", [])),

        # --- 머티리얼 (glTF 2.0 이름) ---
        **material,
        "dominantColors": _dominant_colors(mesh, base_tex),

        # --- 출처 ---
        "backend": generation.get("backend"),
        "modelId": generation.get("model_id"),
        "seed": generation.get("seed"),
        "sourceImage": src.name if src else None,
        "sourceImageSha256": _sha256(src) if src and src.is_file() else None,
    }


def write(meta, glb_path):
    """메타데이터를 ``<glb 경로>.json`` 으로 저장하고 경로를 반환한다."""
    out = Path(glb_path).with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    return out
