"""배경 제거 백엔드 공통 인터페이스."""
from abc import ABC, abstractmethod


class BgRemover(ABC):
    """이미지 -> RGBA (배경 투명)."""

    name: str = ""
    description: str = ""
    license: str = ""

    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def __call__(self, image):
        """PIL 이미지를 받아 알파가 적용된 RGBA 이미지를 반환한다."""
