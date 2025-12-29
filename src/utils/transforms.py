import torchvision.transforms as T
import torch

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

def get_transforms(split='train', img_size=224):
    transforms = []
    if split == 'train':
        transforms.append(T.RandomRotation(360))
    
    transforms.append(T.Resize((img_size, img_size)))
    transforms.append(T.ToTensor())
    # Normalize with ImageNet stats or calculate our own. 
    # For now, standard ImageNet is a safe bet for transfer learning or general vision models.
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)
