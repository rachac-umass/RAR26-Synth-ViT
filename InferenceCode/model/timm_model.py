import torch
import timm
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms

class config:
    repo_path = 'facebookresearch/dinov3'
# Add classification head (replaces num_classes=1 from timm)
class DINOv3Classifier(nn.Module):
    def __init__(self, backbone, embed_dim=1024, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # returns [CLS] token by default
        return self.head(features)
        

class TimmClassificationModel:
    def __init__(self, model_name: str, weights: None, num_classes: int = 1, device: torch.device = None,):
        """
        Wrapper for creating and managing a classification model using timm.

        :param model_name: Name of the model architecture from timm.
        :param device: PyTorch device to move the model to. Defaults to 'cuda' if available.
        :param num_classes: Number of output classes. Default is 1.
        :param pretrained: Whether to load pretrained weights. Default is True.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DINOv3Classifier(torch.hub.load(
                                        repo_or_dir='./resources/dinov3/',
                                        model="dinov3_vitl16",
                                        source="local",
                                        pretrained=False          # prevents trying to fetch weights online
                                    ), num_classes = 1)
        # self.model.load_state_dict(torch.load(weights, map_location=self.device), strict=True)
        # self.model = DINOv3Classifier(timm.create_model(
        #     model_name,#"vit_large_patch16_dinov3.lvd1689m",
        #     pretrained=False,   # offline
        #     num_classes=0       # return features instead of classifier logits
        # ), num_classes = 1)
        self.model.load_state_dict(torch.load(weights, map_location=self.device), strict=True)
        self.model.to(self.device).eval()
        self.transform = self.default_transforms()


    def predict(self, images: list[np.ndarray]):
        """
        Accepts a list of numpy images (HWC, uint8 or float),
        converts them to PIL Images, applies transforms, and runs inference.
        """
        pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        probs = []
        for img in pil_images:
            img = self.transform(img).unsqueeze(0).to(self.device)  # Add batch dimension
            with torch.no_grad():
                logit = self.model(img)
                prob = torch.sigmoid(logit).squeeze().cpu().item()

            probs.append(prob)

        return probs

    @staticmethod
    def default_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


