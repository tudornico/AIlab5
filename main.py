import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import transforms

import Trainer
from ImagineClassifierDataset import ImageClassifierDataset
from load_images import load_images
from network import SimpleNetwork


def save_models(epoch, model):
    torch.save(model.state_dict(), f"cifar10model_{epoch}.model")
    print("Checkpoint saved!")


def test(model, test_loader, cuda_avail):
    model.eval()
    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(images.cuda())
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += torch.sum(torch.eq(prediction, labels.data))
    test_accuracy /= 9
    return test_accuracy


def adjust_learning_rate(epoch, optimizer):
    lr = 0.001

    if epoch > 180:
        lr /= 10 ** 6
    elif epoch > 150:
        lr /= 10 ** 5
    elif epoch > 120:
        lr /= 10 ** 4
    elif epoch > 90:
        lr /= 10 ** 3
    elif epoch > 60:
        lr /= 10 ** 2
    elif epoch > 30:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_all():
    faces_train_path = "./dataset/train/faces"
    no_faces_train_path = "./dataset/train/no_faces"
    faces_test_path = "./dataset/test/faces"
    no_faces_test_path = "./dataset/test/no_faces"

    batch_size = 32
    images_train, classes_train = load_images(faces=faces_train_path, no_faces=no_faces_train_path)
    train_set = ImageClassifierDataset(image_list=images_train, image_classes=classes_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    images_test, classes_test = load_images(faces=faces_test_path, no_faces=no_faces_test_path)
    test_set = ImageClassifierDataset(image_list=images_test, image_classes=classes_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    cuda_avail = torch.cuda.is_available()

    model = SimpleNetwork(num_classes=2)
    if cuda_avail:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    train(num_epochs=1000, model=model, loss_function=loss_fn, optimizer=optimizer, cuda_avail=cuda_avail,
          train_loader=train_loader, test_loader=test_loader)


def train(num_epochs, model, loss_function, optimizer, cuda_avail, train_loader, test_loader):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if cuda_avail:
                images = Variable(images.cuda)
                labels = Variable(labels.cuda)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().data.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_accuracy += torch.sum(prediction == labels.data)
        adjust_learning_rate(epoch=epoch, optimizer=optimizer)
        train_accuracy /= 141
        train_loss /= 141
        test_accuracy = test(model=model, test_loader=test_loader, cuda_avail=cuda_avail)
        if test_accuracy > best_acc:
            save_models(epoch=epoch, model=model)
            best_acc = test_accuracy
        print(f"Epoch {epoch}, Train Accuracy: {train_accuracy}, Train Loss: {train_loss}, Test Accuracy: {test_accuracy}")


def test_on_image(path):
    model_name = "./cifar10model_8.model"
    test_transformations = transforms.Compose([
            transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(path).convert('RGB')
    image = test_transformations(image)
    image = Variable(image.unsqueeze(0))

    model = SimpleNetwork()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    output = model(image)
    _, prediction = torch.max(output.data, 1)
    result = prediction[0].item()
    if result:
        print("Face found")
    else:
        print("No face found")


if __name__ == "__main__":
    while True:
        filename = input("filepath:\n>")
        test_on_image(filename)