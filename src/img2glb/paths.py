"""모델 저장소 위치 해석.

모델 저장소(TRELLIS.2 등)는 용량이 커서 패키지에 포함하지 않는다.
아래 순서로 찾는다.

1. 환경변수 (``IMG2GLB_MODELS`` 또는 백엔드별 ``TRELLIS2_ROOT``)
2. 현재 작업 디렉터리 기준 ``models/<name>``
3. 패키지 설치 위치에서 거슬러 올라가며 ``models/<name>``
"""
import os
from pathlib import Path


def find_model_root(name: str, env_var: str) -> Path:
    """모델 저장소 경로를 찾는다. 못 찾으면 FileNotFoundError."""
    env = os.environ.get(env_var) or os.environ.get("IMG2GLB_MODELS")
    if env:
        p = Path(env)
        p = p / name if p.name != name and (p / name).is_dir() else p
        if p.is_dir():
            return p

    candidates = [Path.cwd() / "models" / name, Path.cwd() / name]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "models" / name)
    for c in candidates:
        if c.is_dir():
            return c

    raise FileNotFoundError(
        f"{name} 저장소를 찾을 수 없습니다.\n"
        f"  {env_var} 환경변수로 경로를 지정하거나,\n"
        f"  scripts/setup_{name}.sh 로 설치하세요."
    )
