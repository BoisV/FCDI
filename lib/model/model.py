import torch
from torch import nn
from torch import functional as F


class FeatrueExtractor(nn.Module):
    def __init__(self):
        super(FeatrueExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=96, kernel_size=7,
                      stride=2, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64,
                      kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # nn.MaxPool2d(kernel_size=2)
        )
        # self.conv4a = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64,
        #               kernel_size=5, stride=1),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.Tanh(),
        #     # nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # nn.AvgPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(in_features=4608, out_features=200, bias=True),
            nn.Linear(in_features=3200, out_features=200, bias=True),
            # nn.Linear(in_features=9216, out_features=200, bias=True),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.Tanh()
        )

    def cal(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1, )
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv4a(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1, )
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class FeatrueExtractor(nn.Module):
    def __init__(self):
        super(FeatrueExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=96, kernel_size=7,
                      stride=2, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64,
                      kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # nn.MaxPool2d(kernel_size=2)
        )
        # self.conv4a = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64,
        #               kernel_size=5, stride=1),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.Tanh(),
        #     # nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # nn.AvgPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(in_features=4608, out_features=200, bias=True),
            nn.Linear(in_features=3200, out_features=200, bias=True),
            # nn.Linear(in_features=9216, out_features=200, bias=True),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.Tanh()
        )

    def cal(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1, )
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv4a(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1, )
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SimilarityNet(nn.Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()
        self.fcA = nn.Sequential(
            nn.Linear(in_features=200, out_features=2048, bias=True),
            nn.Tanh()
        )
        self.fcB = nn.Sequential(
            nn.Linear(in_features=2048*3, out_features=64, bias=True),
            nn.Tanh()
        )
        self.sim_neuron = nn.Sequential(
            nn.Linear(in_features=64, out_features=2, bias=True),
            # nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        x1_inter = self.fcA(x1)
        x2_inter = self.fcA(x2)
        X = torch.cat([x1_inter, x2_inter, x1_inter*x2_inter],
                      dim=1).flatten(start_dim=1)
        X = self.fcB(X)
        return self.sim_neuron(X)


if __name__ == "__main__":
    featureEx = FeatrueExtractor()
    similarityNet = SimilarityNet()
    print(featureEx)
    X1 = torch.randint(0, 256, size=[1, 3, 256, 256], dtype=torch.float32)
    X2 = torch.randint(0, 256, size=[1, 3, 256, 256], dtype=torch.float32)
    X1, X2 = featureEx(X1, X2)
    print(X1.shape, X2.shape)
    y = similarityNet(X1, X2)
    print(y)
