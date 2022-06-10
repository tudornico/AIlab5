import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import transforms, models

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

    faces_train_path_amalia = "./Amalia"
    faces_train_path_coco = "./Coco"
    faces_train_path_nico = "./Nico"
    faces_test_path_ama = "./AmaliaTest"
    faces_test_path_coco = "./CocoTest"
    faces_test_path_nico = "./NicoTest"
    amalia_face_train = load_images(faces_train_path_amalia)
    # create labels for training

    batch_size = 32
    images_train, labels_train = load_images(ama_faces=faces_train_path_amalia, coco_faces=faces_train_path_coco,
                                             nico_faces=faces_train_path_nico)
    # todo create the paths for the testing faces
    images_test, labels_test = load_images(ama_faces=faces_test_path_ama, coco_faces=faces_test_path_coco,
                                           nico_faces=faces_test_path_nico)

    train_dataset = ImageClassifierDataset(images_train, labels_train)
    test_dataset = ImageClassifierDataset(images_test, labels_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    cuda_avail = torch.cuda.is_available()

    # use a Resnet18 model
    model = models.resnet18(pretrained=False)
    if cuda_avail:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    train(num_epochs=100, model=model, loss_function=loss_fn, optimizer=optimizer, cuda_avail=cuda_avail,
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
        print(
            f"Epoch {epoch}, Train Accuracy: {train_accuracy}, Train Loss: {train_loss}, Test Accuracy: {test_accuracy}")


def test_on_image(path):
    model_name = "./cifar10model_0.model"
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
    # model trained
    while True:
        filename = input("filepath:\n>")
        test_on_image(filename)
