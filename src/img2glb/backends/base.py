"""백엔드 공통 인터페이스."""
from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    """이미지 -> GLB 백엔드."""

    name: str = ""
    description: str = ""

    @staticmethod
    def add_arguments(parser) -> None:
        """백엔드 전용 CLI 옵션을 등록한다 (선택)."""

    @abstractmethod
    def generate(self, image_path: Path, output_path: Path, args) -> Path:
        """GLB 를 생성하고 저장된 경로를 반환한다."""
