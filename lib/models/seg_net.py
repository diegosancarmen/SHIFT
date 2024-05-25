# seg_net.py
import torch
import torch.nn as nn
import torchvision.models as models
from segmentation_models_pytorch import DeepLabV3Plus

class SegNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(SegNet, self).__init__()
        self.segmentation_model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        self.segmentation_model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.segmentation_model(x)['out']

def get_segmentation_masks(model, images):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    return outputs


# class SegNet(nn.Module):
#     def __init__(self, num_classes=21, model=None, pretrained=True):
#         super(SegNet, self).__init__()
#         if model == 'deeplabv3plus_r50':
#             self.segmentation_model = DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', classes=num_classes, activation=None)
#         elif model == 'deeplabv3_r50':
#             print('Using deeplabv3_r50')
#             self.segmentation_model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
#             self.segmentation_model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

#     def forward(self, x):
#         return self.segmentation_model(x)

# def get_segmentation_masks(model, images):
#     model.eval()
#     with torch.no_grad():
#         outputs = model(images)
#     return outputs