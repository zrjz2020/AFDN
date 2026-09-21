import os
import numpy as np
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as T
from PIL import Image

_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "weights", "dinov2_vitb14_pretrain.pth")

EMB_DIM = 768
_IMG_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

_model = None
_device = None
_tf = None


def _interpolate_pos_embed(sd, img_size, patch_size=14):
    pos = sd.get("pos_embed")
    if pos is None:
        return sd
    cls_pe = pos[:, :1, :]
    patch_pe = pos[:, 1:, :]
    gs_src = int(round(patch_pe.shape[1] ** 0.5))
    gs_dst = img_size // patch_size
    if gs_src == gs_dst:
        return sd
    dim = patch_pe.shape[-1]
    patch_pe = patch_pe.reshape(1, gs_src, gs_src, dim).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(patch_pe, size=(gs_dst, gs_dst),
                             mode="bicubic", align_corners=False, antialias=True)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, gs_dst * gs_dst, dim)
    sd["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)
    return sd


def _build():
    global _model, _device, _tf
    if _model is not None:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=False,
                              num_classes=0, img_size=_IMG_SIZE)
    if not os.path.isfile(_WEIGHTS):
        raise FileNotFoundError(
            f"DINOv2 权重不存在: {_WEIGHTS}\n"
            "请先下载: curl.exe -L -o weights/dinov2_vitb14_pretrain.pth "
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth")
    sd = torch.load(_WEIGHTS, map_location="cpu")
    sd = _interpolate_pos_embed(sd, _IMG_SIZE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    extra = [k for k in unexpected if "mask_token" not in k]
    print(f"[DINOv2] weights={os.path.basename(_WEIGHTS)} device={_device} "
          f"missing={len(missing)} unexpected(useful)={extra}")
    model.eval().to(_device)
    _model = model
    _tf = T.Compose([
        T.Lambda(_letterbox),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


def _letterbox(img, size=None):
    size = size or _IMG_SIZE
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    fill = tuple(int(round(m * 255)) for m in _MEAN)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


@torch.no_grad()
def extract(pil_img):
    _build()
    if not isinstance(pil_img, Image.Image):
        pil_img = Image.fromarray(pil_img)
    x = _tf(pil_img.convert("RGB")).unsqueeze(0).to(_device)
    emb = _model(x)
    emb = F.normalize(emb, dim=-1)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)
