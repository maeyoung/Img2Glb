"""rembg / U-2-Net 기반 배경 제거 (rembg MIT, U-2-Net Apache-2.0).

BiRefNet 보다 가볍고 CPU 에서도 쓸 만하다. 설치는 선택 사항이다::

    pip install "rembg[cpu]"
"""
from .base import BgRemover


class U2NetRemover(BgRemover):
    name = "u2net"
    description = "rembg / U-2-Net — 가볍고 CPU 에서도 동작. 별도 설치 필요."
    license = "MIT (rembg) + Apache-2.0 (U-2-Net)"

    def __init__(self, device: str = "cuda", model_name: str = "u2net"):
        super().__init__(device)
        self.model_name = model_name
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        try:
            from rembg import new_session
        except ImportError as e:
            raise ImportError(
                "rembg 가 설치되어 있지 않습니다.\n"
                '  pip install "rembg[cpu]"'
            ) from e
        self._session = new_session(self.model_name)

    def __call__(self, image):
        from rembg import remove

        self._load()
        return remove(image.convert("RGB"), session=self._session,
                      bgcolor=[255, 255, 255, 0]).convert("RGBA")
