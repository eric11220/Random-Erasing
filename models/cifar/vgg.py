from torch.autograd import Variable
import math
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, n_conv=2):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.n_conv = n_conv
        if n_conv >= 3:
            self.conv3 = nn.Conv2d(planes, planes, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        if self.n_conv >= 3:
            out = F.relu(self.conv3(out))
        out = self.pool(out)
        return out


class VGG(nn.Module):
    def __init__(self, num_classes=100, h=32, w=32, c=3, final_feat_map_num=128):
        super(VGG, self).__init__()

        fc_w_dim, fc_h_dim = w, h
        for _ in range(3):
            fc_w_dim = int(fc_w_dim / 2)
            fc_h_dim = int(fc_h_dim / 2)
        self.fc_input_dim = fc_w_dim * fc_h_dim * final_feat_map_num

        self.conv_block1 = ConvBlock(c, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, final_feat_map_num, n_conv=3)

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        print("Number of trainable parameters: %d" % sum(p.numel() for p in self.parameters() if p.requires_grad))
        print(self)
        input("vgg model structure")

    # Input would be (n * t * w * h * c)
    def forward(self, x, drop_rate=0., training=True):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=training, p=drop_rate)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, training=training, p=drop_rate)
        x = self.fc3(x)
        return x

def vgg(**kwargs):
    """
    Constructs a VGG model.
    """
    return VGG(**kwargs)
