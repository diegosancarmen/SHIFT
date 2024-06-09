import torch
import torch.nn as nn
import torchvision.models as models
from segmentation_models_pytorch import DeepLabV3Plus

class SegNet(nn.Module):
    """Segmentation Network for generating visiblity masks.

    Args:
    Module: SegNet

    """
    def __init__(self, num_classes=21, pretrained=True):
        super(SegNet, self).__init__()
        self.segmentation_model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=21)

    def forward(self, x):
        return self.segmentation_model(x)['out']

def get_segmentation_masks(model, images):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    return outputs