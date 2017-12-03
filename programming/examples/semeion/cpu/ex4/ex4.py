import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets
import torchvision.models as models
from semeionTransforms import TransformLabel, ExpandTo3D
from sampler import SemeionSampler

transformsData = transforms.Compose([transforms.Scale(250), transforms.ToTensor(), ExpandTo3D()])
transformsTarget = transforms.Compose([TransformLabel()])

dataset = torchvision.datasets.SEMEION(root='./data', download=True, transform=transformsData, target_transform=transformsTarget)

sampler = SemeionSampler(dataset)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=20, sampler=sampler.trainSampler(), num_workers=2)
testloader = torch.utils.data.DataLoader(dataset, batch_size=20, sampler=sampler.testSampler(), num_workers=2)

net = models.alexnet(num_classes=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Start training")
for epoch in range(300):

    running_loss = 0.0
    for data in trainloader:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(loss.data[0])

    correct = 0
    total = 0

    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the test images: %f %%' % (100 * correct / total))

print('Finished Training')
