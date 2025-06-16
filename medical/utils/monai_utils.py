
import torch
import torch
import numpy as np
import monai
from monai.data import ImageDataset, DataLoader, Dataset
from monai.transforms import EnsureChannelFirstd, Compose, RandRotate90d, Resized, ScaleIntensityd, MaskIntensityd, LoadImaged
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable, RandomizableTransform, Transform
from monai.transforms.utils import extreme_points_to_image, get_extreme_points
from monai.transforms.utils_pytorch_numpy_unification import concatenate
from monai.utils import deprecated, deprecated_arg, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix, TraceKeys, TransformBackends
from monai.utils.type_conversion import convert_to_dst_type
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import (
    TraceKeys,
    convert_data_type,
    convert_to_cupy,
    convert_to_numpy,
    convert_to_tensor,
    deprecated,
    deprecated_arg,
    ensure_tuple,
    look_up_option,
    min_version,
    optional_import,
)
from monai.config import DtypeLike, KeysCollection

class Clamp(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, out_min: Optional[float] = None, out_max: Optional[float] = None, **kwargs) -> None:
        self.out_min, self.out_max = out_min, out_max

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if isinstance(img, torch.Tensor):
            out: NdarrayOrTensor = torch.clamp(img, min=self.out_min, max=self.out_max)
        if isinstance(img, np.ndarray):
            out: NdarrayOrTensor = np.clip(img, a_min=self.out_min, a_max=self.out_max)
        return out
    
class Clampd(MapTransform):
    backend = Clamp.backend

    def __init__(self, keys: KeysCollection, out_min: Optional[float] = None, out_max: Optional[float] = None, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = Clamp(out_min=out_min, out_max=out_max)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
    
class Windowing(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __init__(self, w: float, l: float, **kwargs) -> None:
        self.w = w
        self.l = l
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if isinstance(img, torch.Tensor):
            px = img.clone()
            px_min = self.l - self.w//2
            px_max = self.l + self.w//2
            px[px<px_min] = px_min
            px[px>px_max] = px_max
            out: NdarrayOrTensor = (px - px_min) / (px_max - px_min)
        elif isinstance(img, np.ndarray):
            px = img.copy()
            px_min = self.l - self.w//2
            px_max = self.l + self.w//2
            px[px<px_min] = px_min
            px[px>px_max] = px_max
            out: NdarrayOrTensor = (px - px_min) / (px_max - px_min)
        else:
            raise ValueError(f"Unsupported data type for windowing: {type(img)}")
        return out
    
class Windowingd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`utils.monai_utils.Windowing`,
    """
    backend = Windowing.backend
    def __init__(self, keys: KeysCollection, w: float, l: float, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys (KeysCollection): _description_
            w (float): window width
            l (float): window level
            allow_missing_keys (bool, optional): don't raise exception if key is missing. Defaults to False.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = Windowing(w=w, l=l)
        
    def __call__(self, data:Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Apply clip transformation to input data.
        """  
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
    