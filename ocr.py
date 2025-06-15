import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一块
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 16x64
            # 第二块 
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 8x32
            # 第三块
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 第四块
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> 4x32
            # 第五块
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 第六块
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> 2x32
        )
    # 前向传播
    def forward(self, x):
        return self.conv_layers(x)

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        self.cnn = CNNBackbone()
        # 第一 LSTM 层
        self.rnn1 = nn.LSTM(input_size=512 * 2, hidden_size=256, bidirectional=True, batch_first=True)
        # 第二 LSTM 层
        self.rnn2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(512, num_classes)
    # 初始化权重
    def forward(self, x):
        x = self.cnn(x)  # [B, 512, 2, W]
        b, c, h, w = x.size()
        assert h == 2
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.contiguous().view(b, w, c * h)  # [B, W, 1024]

        x, _ = self.rnn1(x)  # -> [B, W, 512]
        x, _ = self.rnn2(x)  # -> [B, W, 512]
        x = self.fc(x)       # -> [B, W, C]
        x = x.permute(1, 0, 2)  # -> [T, B, C]
        return x
