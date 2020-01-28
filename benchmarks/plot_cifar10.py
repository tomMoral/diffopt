import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


DEVICE = 'cuda'


class Model_Implicit(nn.Module):

    def __init__(self, apply_last=True):
        super().__init__()

        self.apply_last = apply_last

        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if apply_last:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x, y=None, t=None):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.apply_last:
            x = self.fc3(x)
        return x


class Model_Autodiff(nn.Module):

    def __init__(self, n_iter=3):
        super().__init__()
        self.n_iter = n_iter

        # Generate the original model
        self.implicit = Model_Implicit(apply_last=False)
        self.W = torch.ones(84, 10, device=DEVICE)

    def forward(self, x, target, t=1):
        output = self.implicit(x)
        target = torch.nn.functional.one_hot(target, num_classes=10)

        if self.training:
            step = 1e-3  # 1 / (10000 + 10*t)
            W = self.W.clone().detach()
            for it in range(self.n_iter):
                logit = torch.softmax(output.mm(W), axis=-1)
                grad = output.t().mm(logit - target)
                W = W - step * grad

            self.W = W
        return output.mm(self.W)


if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model_im = Model_Implicit()
    model_ad = Model_Autodiff(n_iter=1)
    model_ad.to(device=DEVICE)
    model_im.to(device=DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer_im = optim.SGD(model_im.parameters(), lr=0.001, momentum=0.9)
    optimizer_ad = optim.SGD(model_ad.parameters(), lr=0.001, momentum=0.9)

    # in your training loop:
    results = {}
    for net, optimizer, name in [(model_ad, optimizer_ad, 'autodiff'),
                                 (model_im, optimizer_im, 'implicit'),
                                 ]:
        pobj = []
        step = 0
        for epoch in range(10):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                step += 1
                outputs = net(inputs, labels, step)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                pobj.append(loss)

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i+1:5}] loss: '
                          f'{running_loss / 2000:.3f}')
                    running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            net = net.eval()
            for data in testloader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = net(images, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network {name} on the 10000 test images: ',
              100 * correct / total)
        results[name] = pobj

    print('Finished Training')

    pobj = np.array(pobj)
    for name in ['implicit', 'autodiff']:
        plt.semilogy(results[name], label=name)
    plt.legend()
    plt.savefig('cifar_training.pdf')
    plt.show()

    import IPython; IPython.embed(colors='neutral')