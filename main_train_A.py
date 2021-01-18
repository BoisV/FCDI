import os

import torch
from torch import cuda, device, nn, optim
from torch.nn import init
from torch.nn.parallel.data_parallel import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from lib import util
from lib.dataset.FCDIDataset import FCDIDaset
from lib.model import model
from lib.model.model import FeatrueExtractor

logger = util.get_logger()


class PhaseANet(nn.Module):
    def __init__(self, num_classes):
        super(PhaseANet, self).__init__()
        self.features = FeatrueExtractor()
        self.classifier = nn.Sequential(
            # nn.Linear(in_features=200, out_features=100, bias=True),
            # nn.ReLU(),
            nn.Linear(in_features=200, out_features=num_classes, bias=True),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


def cal_acc(pred, Y):
    return (pred.argmax(dim=1) == Y).sum()/pred.shape[0]


model_root = './models'
root = './data/patches/phaseA128'
num_classes = 10
MAX_EPOCHES = 300

if __name__ == "__main__":
    device = device('cuda' if cuda.is_available() else 'cpu')
    model = PhaseANet(num_classes=num_classes)
    model = model.to(device)
    model = DataParallel(module=model, device_ids=[0, 1])

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    init.xavier_normal_(param,)
                    # print(name, param.data)
                elif 'bias' in name:
                    init.constant_(param, 0)
                    # print(name, param.data)

    initial_epoch, state_dict = util.findLastCheckpoint(model_root)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    logger.info(model.device_ids)
    logger.info(model)
    transform = transforms.Compose([transforms.ToTensor(), ])
    train_dataset = FCDIDaset(root=root,
                              train=True,
                              transform=transform)
    train_iter = DataLoader(train_dataset, batch_size=10, shuffle=False)

    for epoch in range(initial_epoch, MAX_EPOCHES, 1):
        loss_sum = 0
        batch_idx = 0
        acc_sum = 0
        for X, Y in train_iter:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc = cal_acc(pred, Y)
            acc_sum += acc
            loss_sum += loss
            batch_idx += 1
            if batch_idx % 30 == 0:
                logger.info(
                    'Epoch:[{}/{}] batch_idx:{} loss={:.5f} accuracy={:.5f}'.format(epoch+1, MAX_EPOCHES, batch_idx, loss_sum/30, acc_sum/30))
                loss_sum = 0
                acc_sum = 0

        # logger.info(
        #     'Epoch:[{}/{}] loss={:.5f} accuracy={:.5f}'.format(epoch+1, MAX_EPOCHES, loss_sum/batch_idx, acc_sum/batch_idx))
        # torch.save(model.state_dict(),
        #            './models/model_{:03d}.pth'.format(epoch))
