import random
from torch.utils.data import Subset
from pathlib import Path
import pickle
import numpy as np
import albumentations as A
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import torch

def save_split_filenames(dataset, train_indices, val_indices,
                         train_pkl="train_filenames.pkl",
                         val_pkl="val_filenames.pkl",
                         path = None):
    train_filenames = [str(dataset.samples[i][0].name) for i in train_indices]
    val_filenames = [str(dataset.samples[i][0].name) for i in val_indices]

    with open(path + train_pkl, "wb") as f:
        pickle.dump(train_filenames, f)

    with open(path + val_pkl, "wb") as f:
        pickle.dump(val_filenames, f)

    print(f"Saved {len(train_filenames)} train filenames to {train_pkl}")
    print(f"Saved {len(val_filenames)} val filenames to {val_pkl}")


def load_split_filenames(train_pkl="train_filenames.pkl",
                         val_pkl="val_filenames.pkl",
                         path = '/home/chandraharsha.rachabathuni-umw/Competitions/RARE26_challenge/data_for_modeling/'):

    
    with open(path + train_pkl, "rb") as f:
        train_filenames = pickle.load(f)

    with open(path + val_pkl, "rb") as f:
        val_filenames = pickle.load(f)

    return train_filenames, val_filenames


def split_dataset(dataset, val_split=0.2, seed=42,
                  save_split=False,
                  train_pkl="train_filenames.pkl",
                  val_pkl="val_filenames.pkl",
                  use_existing_train_val = False,
                  path = '/home/chandraharsha.rachabathuni-umw/Competitions/RARE26_challenge/data_for_modeling/'):
    """
    Split dataset into train and validation sets by class.
    """
    if  use_existing_train_val:
        with open(path + train_pkl, "rb") as f:
            train_saved = set(pickle.load(f))

        with open(path + val_pkl, "rb") as f:
            val_saved = set(pickle.load(f))

        train_indices, val_indices = [], []

        for i, (img_path, _) in enumerate(dataset.samples):
            rel_path = Path(img_path).name  # or .name if you saved names only
            if rel_path in train_saved:
                train_indices.append(i)
            elif rel_path in val_saved:
                val_indices.append(i)

        print("Number of train data points: ",len(train_indices))

        return train_indices, val_indices

    else:

        random.seed(seed)

        neoplasia_indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == "neoplasia"
        ]
        ndbe_indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == "nondysplastic"
        ]

        random.shuffle(neoplasia_indices)
        random.shuffle(ndbe_indices)

        def split(indices):
            val_size = int(len(indices) * val_split)
            return indices[val_size:], indices[:val_size]  # train, val

        train_neo, val_neo = split(neoplasia_indices)
        train_ndbe, val_ndbe = split(ndbe_indices)

        train_indices = train_neo + train_ndbe
        val_indices = val_neo + val_ndbe

        random.shuffle(train_indices)
        random.shuffle(val_indices)

        if save_split:
            save_split_filenames(
                dataset,
                train_indices,
                val_indices,
                train_pkl=train_pkl,
                val_pkl=val_pkl,
                path = path)

        return Strain_indices, val_indices


INTERPOLATION_MODES = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "box": InterpolationMode.BOX,
    "hamming": InterpolationMode.HAMMING,
    "lanczos": InterpolationMode.LANCZOS,
}


def resolve_interpolation_mode(name):
    if isinstance(name, InterpolationMode):
        return name
    try:
        return INTERPOLATION_MODES[name]
    except KeyError:
        raise ValueError(
            f"Unsupported interpolation mode: {name}. "
            f"Choose from {sorted(INTERPOLATION_MODES.keys())}"
        )

class AlbumentationsBlurNoise:
    """Randomly applies one of a blur/noise transform, via albumentations' A.OneOf.

    Inserted right after Resize (on the still-uint8 image, before ToTensor), since
    albumentations transforms operate on HWC numpy arrays rather than torch tensors.
    """

    def __init__(self, p=0.3):
        self.transform = A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(3.0, 18.0)),
        ], p=p)

    def __call__(self, img):
        return self.transform(image=np.array(img))["image"]


class AlbumentationsDistortion:
    """Randomly applies one of an optical/grid/elastic distortion, via albumentations' A.OneOf.

    Inserted right after Resize (on the still-uint8 image, before ToTensor), since
    albumentations transforms operate on HWC numpy arrays rather than torch tensors.
    """

    def __init__(self, p=0.3):
        self.transform = A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.0),
            A.ElasticTransform(alpha=3),
        ], p=p)

    def __call__(self, img):
        return self.transform(image=np.array(img))["image"]


class RandomBlackBoxes:
    def __init__(self, p=0.3, box_height=2, box_width=2, min_count=1, max_count=10):
        self.p = p
        self.box_height = box_height
        self.box_width = box_width
        self.min_count = min_count
        self.max_count = max_count

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: Tensor of shape [C, H, W], expected in [0, 1].
        """
        if random.random() >= self.p:
            return img

        _, height, width = img.shape
        if height < self.box_height or width < self.box_width:
            return img

        img = img.clone()  # Do not modify the original tensor in-place
        n_boxes = random.randint(self.min_count, self.max_count)

        for _ in range(n_boxes):
            top = random.randint(0, height - self.box_height)
            left = random.randint(0, width - self.box_width)

            img[:, top:top + self.box_height, left:left + self.box_width] = 0.0

        return img


class RandomZoom:
    """Randomly zooms in on the image center by a factor in [1.0, 1 + max_zoom].

    Applied with probability p. Scales the image up by the random factor and
    center-crops back to the original size, so the effect is a random 0% to
    max_zoom% zoom-in around the image center (never zooms out). Operates on
    the resized image tensor (after ToTensor), like RandomBlackBoxes.
    """

    def __init__(self, p=0.3, max_zoom=0.1):
        self.p = p
        self.max_zoom = max_zoom

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img

        _, height, width = img.shape
        zoom_factor = 1.0 + random.uniform(0.0, self.max_zoom)
        new_h, new_w = int(round(height * zoom_factor)), int(round(width * zoom_factor))

        zoomed = TF.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
        # Bicubic interpolation can ring/overshoot outside [0, 1] near hard edges
        # (e.g. RandomBlackBoxes' flat zeroed rectangles), which downstream steps
        # like ColorJitter's hue adjustment assume never happens -- an overshoot
        # that straddles exactly 0 across channels hits a division-by-zero in
        # torchvision's RGB->HSV conversion and produces NaN pixels.
        zoomed = zoomed.clamp(0.0, 1.0)
        return TF.center_crop(zoomed, [height, width])


class RandomBlackBorder:
    """Randomly shrinks the image to `scale` of its size and pads the rest with a
    black border, applied with probability p. Operates on the resized image tensor
    (after ToTensor), like RandomBlackBoxes/RandomZoom. This is the inverse of
    RandomZoom: instead of zooming in, it zooms out and pads the vacated border
    with black so the tensor shape is unchanged.
    """

    def __init__(self, p=0.3, scale=0.9):
        self.p = p
        self.scale = scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img

        _, height, width = img.shape
        new_h, new_w = int(round(height * self.scale)), int(round(width * self.scale))

        shrunk = TF.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
        shrunk = shrunk.clamp(0.0, 1.0)

        pad_top = (height - new_h) // 2
        pad_bottom = height - new_h - pad_top
        pad_left = (width - new_w) // 2
        pad_right = width - new_w - pad_left

        return TF.pad(shrunk, [pad_left, pad_top, pad_right, pad_bottom], fill=0.0)


class AlbumentationsZoomBlurDistortion:
    """With probability p, applies exactly one of zoom / blur+noise / distortion,
    chosen uniformly among the three categories, via a top-level albumentations
    A.OneOf. Unlike --use_zoom/--use_blur_noise/--use_distortion (which each fire
    independently), this combines all three families into one mutually-exclusive
    choice per call. Reuses the same candidate ops as AlbumentationsBlurNoise/
    AlbumentationsDistortion; the zoom candidate is a scale-only (>=1.0) A.Affine,
    which -- since it never zooms out -- fills the canvas without needing a border
    fill, matching RandomZoom's zoom-in-and-crop behavior. Inserted right after
    Resize (on the still-uint8 image, before ToTensor), like AlbumentationsBlurNoise/
    AlbumentationsDistortion.
    """

    def __init__(self, p=0.4, max_zoom=0.08):
        zoom = A.Affine(scale=(1.0, 1.0 + max_zoom), p=1.0)
        blur = A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(3.0, 18.0)),
        ], p=1.0)
        distortion = A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.0),
            A.ElasticTransform(alpha=3),
        ], p=1.0)
        self.transform = A.OneOf([zoom, blur, distortion], p=p)

    def __call__(self, img):
        return self.transform(image=np.array(img))["image"]


def get_train_transforms(resize_image_dim, interpolation="bilinear", antialias=True, use_blackbox=False,
                          use_blur_noise=False, use_distortion=False, use_zoom=False, use_black_border=False,
                          use_zoom_blur_distortion=False, mean=None, std=None):
    transform_list = [
        transforms.Resize(
            size=(resize_image_dim, resize_image_dim),
            interpolation=resolve_interpolation_mode(interpolation),
            antialias=antialias,
        ),
    ]
    if use_blur_noise:
        transform_list.append(AlbumentationsBlurNoise(p=0.3))
    if use_distortion:
        transform_list.append(AlbumentationsDistortion(p=0.3))
    if use_zoom_blur_distortion:
        transform_list.append(AlbumentationsZoomBlurDistortion(p=0.5))
    transform_list.append(transforms.ToTensor())
    if use_blackbox:
        transform_list.append(
            RandomBlackBoxes(
                p=0.4,
                box_height=4,
                box_width=10,
                min_count=5,
                max_count=10,
            )
        )
    if use_zoom:
        transform_list.append(RandomZoom(p=0.3, max_zoom=0.08))
    if use_black_border:
        transform_list.append(RandomBlackBorder(p=0.3, scale=0.9))
    transform_list += [
        transforms.RandomRotation(degrees=(15, 355)),
        transforms.RandomVerticalFlip(p=0.35),
        transforms.RandomApply(
        [transforms.ColorJitter(
            brightness=(0.80, 1.20),
            contrast=(0.90, 1.10),
            saturation=(0.95, 1.05),
            hue=(-0.01, 0.01),
        )],
        p=0.7,),
        transforms.Normalize(
            mean=mean if mean is not None else [0.485, 0.456, 0.406],
            std=std if std is not None else [0.229, 0.224, 0.225]
        ),
    ]
    return transforms.Compose(transform_list)

def get_val_transforms(resize_image_dim, interpolation="bilinear", antialias=True, mean=None, std=None):
    return transforms.Compose([
        transforms.Resize(
            size=(resize_image_dim, resize_image_dim),
            interpolation=resolve_interpolation_mode(interpolation),
            antialias=antialias,
        ),
        transforms.ToTensor(),
        # SimulateNBI(),
        transforms.Normalize(
            mean=mean if mean is not None else [0.485, 0.456, 0.406],
            std=std if std is not None else [0.229, 0.224, 0.225]
        ),
    ])