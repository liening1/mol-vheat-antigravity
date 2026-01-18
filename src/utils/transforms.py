import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image

class RandomRotation:
    """
    Randomly rotate the image by an angle.
    For molecules, 0-360 degrees is valid as properties are invariant.
    """
    def __init__(self, degrees=360):
        self.degrees = degrees
        self.transform = T.RandomRotation(degrees)
        
    def __call__(self, x):
        return self.transform(x)


class ReplaceWhiteBackground:
    """Replace near-white background pixels with a neutral color.

    Motivation: RDKit molecule renders use a white background. If left as-is, the model
    can overfit to background pixels even though they are chemically meaningless.

    By filling background with the ImageNet mean color, the subsequent Normalize() makes
    background approximately zero, reducing spurious cues.
    """

    def __init__(
        self,
        mode: str = 'mean',
        threshold: int = 250,
        jitter_std: float = 0.02,
        rng_seed: int = 42,
    ):
        self.mode = mode
        self.threshold = int(threshold)
        self.jitter_std = float(jitter_std)
        self.rng = np.random.default_rng(int(rng_seed))

    def _fill_rgb_uint8(self):
        # ImageNet mean in [0, 255]
        mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)

        if self.mode == 'mean':
            rgb = mean_rgb
        elif self.mode == 'mean_jitter':
            rgb = mean_rgb + self.rng.normal(0.0, self.jitter_std, size=(3,)).astype(np.float32)
            rgb = np.clip(rgb, 0.0, 1.0)
        else:
            # Unknown mode => no-op by returning None
            return None

        return (rgb * 255.0).round().astype(np.uint8)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return img
        if self.mode not in ('mean', 'mean_jitter'):
            return img

        arr = np.array(img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return img

        bg = (arr[:, :, 0] >= self.threshold) & (arr[:, :, 1] >= self.threshold) & (arr[:, :, 2] >= self.threshold)
        fill = self._fill_rgb_uint8()
        if fill is None:
            return img

        out = arr.copy()
        out[bg] = fill
        return Image.fromarray(out, mode='RGB')

def get_transforms(split='train', img_size=224, background: str = 'none'):
    transforms = []
    if split == 'train':
        transforms.append(T.RandomRotation(360))
    
    transforms.append(T.Resize((img_size, img_size)))

    # Optional: neutralize the white background so it doesn't become a shortcut feature.
    # Use background='mean_jitter' for train (recommended) and background='mean' for val/test.
    if background in ('mean', 'mean_jitter'):
        transforms.append(ReplaceWhiteBackground(mode=background))

    transforms.append(T.ToTensor())
    # Normalize with ImageNet stats or calculate our own. 
    # For now, standard ImageNet is a safe bet for transfer learning or general vision models.
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)
