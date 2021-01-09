import torch
from torch import nn
from torch import functional as F
class FeatrueExtractor(nn.Module):
    def __init__(self):
        super(FeatrueExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=96, kernel_size=7, stride=2),
            nn.BatchNorm2d(num_features=96),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4608, out_features=200, bias=True),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.Tanh()
        )

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1, )
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = FeatrueExtractor()
    print(model)
    X = torch.randint(0, 256, size=[1,3, 256, 256], dtype=torch.float32)
    y = model(X)
    print(y.shape)