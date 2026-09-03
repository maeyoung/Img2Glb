"""BiRefNet 기반 배경 제거 (ZhengPeng7/BiRefNet, MIT)."""
from .base import BgRemover

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class BiRefNetRemover(BgRemover):
    name = "birefnet"
    description = "BiRefNet — 품질 우선. GPU 권장."
    license = "MIT (ZhengPeng7/BiRefNet)"

    def __init__(self, device: str = "cuda", model_id: str = "ZhengPeng7/BiRefNet",
                 input_size: int = 1024):
        super().__init__(device)
        self.model_id = model_id
        self.input_size = input_size
        self._model = None
        self._tf = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            self.model_id, trust_remote_code=True)
        model.eval()
        if self.device == "cuda" and torch.cuda.is_available():
            model.to("cuda")
        else:
            self.device = "cpu"
        self._model = model
        self._tf = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def __call__(self, image):
        import torch
        from torchvision import transforms

        self._load()
        rgb = image.convert("RGB")
        # 체크포인트가 half 로 저장된 경우가 있어 모델 dtype 에 입력을 맞춘다
        dtype = next(self._model.parameters()).dtype
        x = self._tf(rgb).unsqueeze(0).to(self.device, dtype=dtype)
        with torch.no_grad():
            pred = self._model(x)[-1].sigmoid().float().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(rgb.size)
        out = rgb.convert("RGBA")
        out.putalpha(mask)
        return out
