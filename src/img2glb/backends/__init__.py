"""생성 백엔드 레지스트리.

새 모델을 추가하려면 ``Backend`` 를 상속한 클래스를 만들고
``BACKENDS`` 에 등록하면 CLI 에 자동으로 노출된다.
"""
from .base import Backend
from .spar3d import Spar3dBackend
from .trellis2 import Trellis2Backend

BACKENDS = {
    "spar3d": Spar3dBackend,
    "trellis2": Trellis2Backend,
}

__all__ = ["Backend", "BACKENDS"]
