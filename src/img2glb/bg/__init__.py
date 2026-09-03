"""배경 제거 — 생성 모델과 독립적으로 동작한다.

이미지에 알파 채널을 입혀 반환한다. 생성 전처리로도 쓰고,
``img2glb remove-bg`` 로 단독 실행도 한다.

라이선스
--------
기본 백엔드는 **BiRefNet (ZhengPeng7/BiRefNet, MIT)** 이다.
TRELLIS.2 가 기본으로 쓰는 ``briaai/RMBG-2.0`` 은 같은 아키텍처를 BRIA 가
파인튜닝한 것으로 **CC BY-NC 4.0(비상업)** 이고 gated 다. 상업적 배포를
막지 않기 위해 원본 MIT 가중치를 쓴다.

``u2net`` 백엔드는 rembg(MIT) + U-2-Net(Apache-2.0) 을 쓰며 더 가볍다.
"""
from .base import BgRemover
from .birefnet import BiRefNetRemover
from .u2net import U2NetRemover

REMOVERS = {
    "birefnet": BiRefNetRemover,
    "u2net": U2NetRemover,
}

DEFAULT = "birefnet"

__all__ = ["BgRemover", "REMOVERS", "DEFAULT", "remove_background", "has_alpha"]


def has_alpha(image) -> bool:
    """이미 유효한 알파 채널이 있는지 확인한다."""
    if image.mode not in ("RGBA", "LA"):
        return False
    return image.getchannel("A").getextrema()[0] < 255


def remove_background(image, method: str = DEFAULT, device: str = "cuda", **kwargs):
    """이미지에서 배경을 제거해 RGBA 로 반환한다.

    Args:
        image: PIL.Image
        method: ``REMOVERS`` 의 키
        device: "cuda" 또는 "cpu"

    Returns:
        PIL.Image (RGBA)
    """
    if method not in REMOVERS:
        raise ValueError(f"알 수 없는 배경제거 방식: {method} "
                         f"(가능: {', '.join(sorted(REMOVERS))})")
    return REMOVERS[method](device=device, **kwargs)(image)
