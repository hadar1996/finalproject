import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from math import floor, ceil


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout())
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout())
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout())
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout())
        self.enc5 = nn.Sequential(
            nn.Linear(128 * floor(pixels/16) * floor(pixels/16), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 32))

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = x.reshape(x.size(0), -1)
        return nn.functional.log_softmax(self.enc5(x), dim=1)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec1 = nn.Linear(in_features=32, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=pixels * pixels)

    def forward(self, z):
        z = F.relu(self.dec1(z))
        z = F.relu(self.dec2(z))
        z = F.relu(self.dec3(z))
        z = F.relu(self.dec4(z))
        return nn.functional.log_softmax(self.dec5(z), dim=1)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout())
        self.layer5 = nn.Sequential(
            nn.Linear(128 * floor(pixels / 16) * floor(pixels / 16), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.layer5(x)
        return nn.functional.log_softmax(x, dim=1)


def data_reader():
    path = './cell_images/C-NMC_Leukemia/training_data'
    dataset_train = ImageFolder(path, transform=Compose([Resize((pixels, pixels)), ToTensor(), Grayscale()]))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    path = './cell_images/C-NMC_Leukemia/validation_data'
    dataset_validation = ImageFolder(path, transform=Compose([Resize((pixels, pixels)), ToTensor(), Grayscale()]))
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=64, shuffle=True)
    print("read the data")
    return train_loader, validation_loader


def classification_train():
    model.train()
    for e in range(epoch):
        classification_loss = 0
        correct = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            print("im here")
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, labels, reduction='sum')
            classification_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).cpu().sum()
            loss.backward()
            optimizer.step()
        print("Classification Train set: Epoch num:{}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)"
              .format(e, classification_loss/len(train_loader.dataset), correct, len(train_loader.dataset),
                      100. * correct / len(train_loader.dataset)))


def classifier_validation():
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            valid_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            print("prediction:" + str(pred) + '\t' + "target:" + str(target))
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    valid_loss /= len(test_loader.dataset)
    print('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(valid_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def encoder_train():
    encoder_loss = nn.MSELoss()
    model.train()
    for e in range(epoch):
        encoding_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = encoder_loss(output, data.reshape(data.size(0), -1))
            encoding_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Train set: Epoch num:{}, Average loss: {:.4f}"
              .format(e, encoding_loss/len(train_loader.dataset)))


def encoder_validation():
    test_examples = None
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features
            recon = model(test_examples)
            break

    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].numpy().reshape(pixels, pixels))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # plot reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(recon[index].numpy().reshape(pixels, pixels))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


# main
if __name__ == '__main__':
    # Parameters Box
    epoch = 4
    learning_rate = 0.1
    pixels = 128
    reconstruction_mode = False
    classification_mode = True

    train_loader, test_loader = data_reader()
    if reconstruction_mode:
        model = Autoencoder()
    elif classification_mode:
        model = Classifier()
    else:
        model = None

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if reconstruction_mode:
        encoder_train()
        encoder_validation()

    elif classification_mode:
        classification_train()
        classifier_validation()
    else:
        print("im out")
