import torch
import torch.nn as nn

class KeypointToSegmentationEncoder(nn.Module):
    def __init__(self, num_classes=21, num_keypoints=16, output_size=256, nz=100, ngf=64):
        super(KeypointToSegmentationEncoder, self).__init__()
        self.nz = nz
        self.fc = nn.Linear(num_keypoints * 2, nz)

        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, num_classes, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size. num_classes x 64 x 64
        )

        self.output_size = output_size
        self.upsample = nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=False)

    def forward(self, keypoints):
        """
        Forward pass to map keypoints to a segmentation map.

        Args:
            keypoints (torch.Tensor): Keypoint coordinates (B, num_keypoints, 2)
        
        Returns:
            torch.Tensor: Segmentation map (B, num_classes, output_size, output_size)
        """
        B, num_keypoints, _ = keypoints.shape
        x = keypoints.view(B, -1)  # Flatten keypoints
        x = self.fc(x)
        x = x.view(B, self.nz, 1, 1)  # Reshape to (B, nz, 1, 1)
        x = self.main(x)
        x = self.upsample(x)  # Upsample to the desired output size
        return x
    
def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)