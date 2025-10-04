import torch
import torch.nn as nn


class LDB(nn.Module):
    def __init__(self, in_channel: int, t: float = 0.5):
        super(LDB, self).__init__()
        mid_channel = int(in_channel * t)

        # 1x1 Conv expansion
        self.conv1x1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.relu = nn.ReLU(inplace=True)

        # Four 3x3 conv blocks
        self.conv3x3_1 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)

        self.conv3x3_2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channel)

        self.conv3x3_3 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channel)

        self.conv3x3_4 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channel)

    def forward(self, x):
        # Expand
        out = self.relu(self.bn1(self.conv1x1(x)))
        
        out1 = (self.conv3x3_1(out))
        # First conv block
        out2 = self.conv3x3_2(self.relu(self.bn2(out1)))
        out3 = self.conv3x3_3(self.relu(self.bn2(out1+out2)))
        out4 = self.conv3x3_4(self.relu(self.bn2(out1+out2+out3)))
        


        # Concatenate input and all outputs
        out_final = torch.cat([x, out1, out2, out3, out4], dim=1)

        return out_final
        
        
        
class Transition(nn.Module):
    def __init__(self, in_channel, out_channel=32):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CDenseNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=3, t=0.5, n=16):
        super(CDenseNet, self).__init__()
        
        # Initial Conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        

        
        # Stack of (LDB + Transition)
        layers = []
        in_ch = 32
        for _ in range(n):
            ldb = LDB(in_ch, t=t)
            in_ch = int(in_ch * (1 + 4 * t))  # update channels after LDB
            trans = Transition(in_ch, 32)
            in_ch = 32  # transition resets channels to 32
            layers += [ldb, trans]
        
        self.blocks = nn.Sequential(*layers)
        
        # GAP + FC layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu_forfc = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu_forfc(x)
        x = self.fc2(x)
        return x
