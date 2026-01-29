from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers.gee import GEEProvider
from .base import EmbedderBase
from .meta_utils import build_meta, temporal_to_range, temporal_midpoint_str

# ------------------------------------------------------------------------------
# 1. GEE Data Fetcher (复用 RemoteCLIP 的 RGB 逻辑，但在本地定义以解耦)
# ------------------------------------------------------------------------------
def _fetch_s2_rgb_chw(
    provider: GEEProvider,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
    import ee
    
    region = provider.get_region_3857(spatial)
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(temporal.start, temporal.end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloudy_pct))
    )

    if composite == "median":
        img = col.median()
    elif composite == "mosaic":
        img = col.mosaic()
    else:
        raise ModelError(f"Unknown composite='{composite}'. Use 'median' or 'mosaic'.")

    # SkySense 通常在 RGB 模式下需要标准的 RGB 波段
    img = img.select(["B4", "B3", "B2"]).reproject(crs="EPSG:3857", scale=scale_m)

    rect = img.sampleRectangle(region=region, defaultValue=0.0).getInfo()
    props = rect["properties"]

    # S2 SR 缩放 0..10000 -> 0..1.0
    r = np.array(props["B4"], dtype=np.float32) / 10000.0
    g = np.array(props["B3"], dtype=np.float32) / 10000.0
    b = np.array(props["B2"], dtype=np.float32) / 10000.0
    
    # Clip to 0-1
    return np.clip(np.stack([r, g, b], axis=0), 0.0, 1.0)


# ------------------------------------------------------------------------------
# 2. SkySense 模型加载器 (Wrapper)
# ------------------------------------------------------------------------------
def _load_skysense_model(config_path: str, checkpoint_path: str, device: str = "auto"):
    """
    加载 SkySense 模型。依赖 mmcv, mmpretrain 等库。
    """
    try:
        import torch
        from mmengine.config import Config
        from mmengine.runner import load_checkpoint
        # 假设用户环境中有 SkySense 的代码库或相关依赖
        # 这里尝试通用的 MMPretrain 构建方式，SkySense 大多基于此架构
        from mmpretrain.models import build_classifier, build_backbone
    except ImportError as e:
        raise ModelError(
            "SkySense 依赖 OpenMMLab 组件。请安装: pip install mmengine mmcv mmpretrain"
        ) from e

    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)

    if not os.path.exists(config_path):
        raise ModelError(f"SkySense config not found at: {config_path}")
    
    if not os.path.exists(checkpoint_path):
        raise ModelError(f"SkySense checkpoint not found at: {checkpoint_path}")

    cfg = Config.fromfile(config_path)
    
    # 构建模型
    # SkySense 的官方代码通常是一个 Backbone 或者一个 Classifier 结构
    try:
        if "model" in cfg:
            model = build_classifier(cfg.model)
        elif "backbone" in cfg:
            model = build_backbone(cfg.backbone)
        else:
            # 最后的尝试：直接构建整个 Config
            from mmengine.registry import MODELS
            model = MODELS.build(cfg.model)
    except Exception as e:
        raise ModelError(f"Failed to build SkySense model from config: {e}")

    # 加载权重
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to(dev)
    model.eval()
    
    return model, dev, cfg

# ------------------------------------------------------------------------------
# 3. SkySense Embedder 类
# ------------------------------------------------------------------------------
@register("skysense_plus_s2")
class SkySensePlusS2Embedder(EmbedderBase):
    """
    SkySense++ (Sentinel-2 RGB) On-the-fly Embedder.
    
    需要用户通过环境变量或 sensor.collection 指定本地的 Config 和 Checkpoint 路径。
    
    Usage:
        sensor = SensorSpec(collection="path/to/config.py::path/to/checkpoint.pth")
    """

    DEFAULT_SCALE = 10
    
    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["gee"],
            "inputs": {"collection": "COPERNICUS/S2_SR_HARMONIZED", "bands": ["B4", "B3", "B2"]},
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "scale_m": 10,
                "composite": "median",
            },
            "notes": "Requires 'mmcv' and local SkySense weights. Pass 'config::ckpt' in sensor.collection.",
        }

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
    ) -> Embedding:
        
        if backend.lower() != "gee":
            raise ModelError("skysense_plus_s2 currently only supports 'gee' backend.")

        # 1. 解析配置
        # 我们利用 sensor.collection 来传递 "config_path::checkpoint_path"
        # 例如: "configs/skysense_swin_l.py::checkpoints/skysense_v2.pth"
        col_str = sensor.collection if sensor and sensor.collection else ""
        if "::" in col_str:
            config_path, ckpt_path = col_str.split("::", 1)
        else:
            # 尝试从环境变量读取默认值
            config_path = os.environ.get("RS_EMBED_SKYSENSE_CONFIG")
            ckpt_path = os.environ.get("RS_EMBED_SKYSENSE_CKPT")
            
        if not config_path or not ckpt_path:
            raise ModelError(
                "SkySense requires paths to config and checkpoint. "
                "Set RS_EMBED_SKYSENSE_CONFIG and RS_EMBED_SKYSENSE_CKPT env vars, "
                "or pass 'config::ckpt' as the 'collection' name."
            )

        # 2. 获取数据 (GEE)
        t = temporal_to_range(temporal)
        provider = GEEProvider(auto_auth=True)
        provider.ensure_ready()
        
        scale_m = sensor.scale_m if sensor else self.DEFAULT_SCALE
        cloudy_pct = sensor.cloudy_pct if sensor else 30
        composite = sensor.composite if sensor else "median"

        # 获取 Sentinel-2 RGB [3, H, W] in [0, 1]
        img_chw = _fetch_s2_rgb_chw(
            provider, spatial, t, scale_m=scale_m, cloudy_pct=cloudy_pct, composite=composite
        )
        
        # 3. 加载模型
        model, dev, cfg = _load_skysense_model(config_path, ckpt_path, device)
        
        # 4. 推理 (Inference)
        import torch
        from torchvision import transforms
        
        # SkySense 预处理通常需要 Resize 到特定大小 (e.g. 224, 384) 或者支持任意大小
        # 这里假设使用模型 Config 中的 Input Size，如果没有则默认 224
        # 注意：这里为了简单，我们强制 Resize，这对于 Embedding 提取是常见的做法
        input_size = (224, 224) 
        if hasattr(cfg, 'img_size'):
             input_size = (cfg.img_size, cfg.img_size) if isinstance(cfg.img_size, int) else cfg.img_size

        preprocess = transforms.Compose([
            transforms.Resize(input_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet norm
        ])
        
        # 转换为 Tensor: [1, 3, H, W]
        x_tensor = torch.from_numpy(img_chw).float().unsqueeze(0)
        x_tensor = preprocess(x_tensor).to(dev)

        with torch.no_grad():
            # 不同的 OpenMMLab 模型提取 Feature 的方式不同
            # 通常 backbone(x) 返回一个 tuple (feat1, feat2, ..., feat_last)
            if hasattr(model, "extract_feat"):
                feats = model.extract_feat(x_tensor)
            elif hasattr(model, "backbone"):
                feats = model.backbone(x_tensor)
            else:
                feats = model(x_tensor)
            
            # 取最后一层特征
            if isinstance(feats, (tuple, list)):
                last_feat = feats[-1]
            else:
                last_feat = feats

        # last_feat 形状通常是 [B, C, H_grid, W_grid] (例如 Swin Output)
        # 或者 [B, N, C] (ViT Output)
        
        feat_np = last_feat.detach().cpu().numpy()[0] # [C, H, W] or [N, C]
        
        # 5. 格式化输出
        meta = build_meta(
            model="skysense_plus",
            kind="on_the_fly",
            backend="gee",
            source="Sentinel-2",
            temporal=t,
            input_time=temporal_midpoint_str(t),
            extra={
                "config": config_path,
                "ckpt": ckpt_path,
                "feat_shape": feat_np.shape
            }
        )

        # 处理 Grid 输出
        if output.mode == "grid":
            if feat_np.ndim == 3: # [C, H, W]
                # 已经是 Grid
                data = feat_np # [D, H, W]
            elif feat_np.ndim == 2: # [N, C]
                # 需要 Reshape (假设是方阵)
                n, c = feat_np.shape
                h = w = int(np.sqrt(n))
                if h * w != n:
                     # 尝试去掉 CLS token
                     if int(np.sqrt(n-1))**2 == n-1:
                         feat_np = feat_np[1:]
                         h = w = int(np.sqrt(n-1))
                data = feat_np.reshape(h, w, c).transpose(2, 0, 1) # [C, H, W]
            
            da = xr.DataArray(
                data,
                dims=("d", "y", "x"),
                coords={"d": np.arange(data.shape[0]), "y": np.arange(data.shape[1]), "x": np.arange(data.shape[2])},
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        # 处理 Pooled 输出
        elif output.mode == "pooled":
            if feat_np.ndim == 3:
                # Global Average Pooling over H, W
                vec = np.mean(feat_np, axis=(1, 2))
            else:
                # Global Average Pooling over Tokens
                vec = np.mean(feat_np, axis=0)
            
            return Embedding(data=vec, meta=meta)

        else:
            raise ModelError(f"Unknown output mode: {output.mode}")