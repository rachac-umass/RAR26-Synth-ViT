from pathlib import Path

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset




class RareDataset(Dataset):
    def __init__(self, root_dir, use_extradata_Endovis = False, use_extradata_edd2020 = False,
    use_extradata_hyper_short_segment = False, use_extradata_GastroVision = False,
    use_extradata_synthetic_data = False, use_extradata_barett_archive = False,
    use_extradata_red_patch = False,
    transform=None, resize_img_dim = 224, **kwargs):
        """
        Args:
            root_dir (str): Path to the main dataset folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.resize_img_dim = resize_img_dim
        self.transform = transform if transform else self.default_transforms()
        self.classes = {"neo": "neoplasia", "ndbe": "nondysplastic"}
        self.class_counts = {"neoplasia": 0, "nondysplastic": 0}
        self.use_extradata_Endovis = use_extradata_Endovis
        self.use_extradata_edd2020 = use_extradata_edd2020
        self.use_extradata_hyper_short_segment = use_extradata_hyper_short_segment
        self.use_extradata_GastroVision = use_extradata_GastroVision
        self.use_extradata_synthetic_data = use_extradata_synthetic_data
        self.use_extradata_barett_archive = use_extradata_barett_archive
        self.use_extradata_red_patch = use_extradata_red_patch

        self.samples = self.load_samples()
        

        # Print counts after loading
        print(
            f"Loaded dataset with {self.class_counts['neoplasia']} 'neo' (neoplasia) images and "
            f"{self.class_counts['nondysplastic']} 'ndbe' (nondysplastic) images."
        )

    def default_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.resize_img_dim, self.resize_img_dim)),
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=(15,135)),
                transforms.RandomVerticalFlip(p=0.35),
                transforms.ColorJitter(
                brightness=(0.85, 1.15),  # Dims down to 85% or brightens up to 115%
                contrast=(0.90, 1.10),    # Strictly tight to preserve fine vascular lines
                saturation=(0.95, 1.05),  # GI tissue relies heavily on true pink/red tones
                hue=(-0.01, 0.01)         # Almost zero shift to prevent NBI artifacts
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_samples(self):
        samples = []
        valid_exts = {'.png', '.jpg', '.jpeg'}
        for center in self.root_dir.iterdir():
            if (not self.use_extradata_Endovis and center.name == "external_data_EVCBarrett"):
                continue
            if (not self.use_extradata_edd2020 and center.name == "external_edd2020"):
                continue
            if (not self.use_extradata_hyper_short_segment and center.name == "hyper_short_segment"):
                continue
            if (not self.use_extradata_GastroVision and center.name == "GastroVision"):
                continue
            if (not self.use_extradata_synthetic_data and center.name == "synthetic_data"):
                continue
            if (not self.use_extradata_barett_archive and center.name == "barett_archive"):
                continue
            if (not self.use_extradata_red_patch and center.name == "red_patch"):
                continue
            if center.is_dir():
                for class_folder in ["neo", "ndbe"]:
                    class_dir = center / class_folder
                    if class_dir.exists():
                        for img_path in class_dir.iterdir():
                            if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                                label = self.classes[class_folder]
                                samples.append((img_path, label))
                                self.class_counts[label] += 1  # Count the sample
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = 1 if label == "neoplasia" else 0
        return image, label


class RareTestSet(Dataset):
    def __init__(self, root_dir, transform=None,resize_img_dim = 224, return_paths=False):
        """
        Args:
            root_dir (str): Path to the main dataset folder (contains 'neo/' and 'ndbe/').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.resize_img_dim = resize_img_dim
        self.transform = transform if transform else self.default_transforms()
        self.classes = {"neo": "neoplasia", "ndbe": "nondysplastic"}
        self.class_counts = {"neoplasia": 0, "nondysplastic": 0}
        self.samples = self.load_samples()
        self.return_paths = return_paths

        # Print counts after loading
        print(
            f"Loaded test set with {self.class_counts['neoplasia']} 'neo' (neoplasia) images and "
            f"{self.class_counts['nondysplastic']} 'ndbe' (nondysplastic) images."
        )

    def default_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_samples(self):        
        samples = []
        valid_exts = {'.png', '.jpg', '.jpeg'}
        for class_folder in ["neo", "ndbe"]:
            class_dir = self.root_dir / class_folder
            if class_dir.exists():
                        for img_path in class_dir.iterdir():
                            if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                                label = self.classes[class_folder]
                                samples.append((img_path, label))
                                self.class_counts[label] += 1  # Count the sample
        return samples

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = 1 if label == "neoplasia" else 0
        if self.return_paths:
            return image, label, str(img_path)
        else:
            return image, label