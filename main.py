import requests
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
# take data from my drive
import pickle

def save_models(epoch, model):
    # rewrite the model with the best accuracy
    pickle.dump(model, open("finalized_model_resnet.sav", "wb"))
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
    # reduce 105 to 70 somthing or change the 28 to higher
    if epoch > 180:
        lr /= 13 ** 6
    elif epoch > 140:
        lr /= 13 ** 5
    elif epoch > 90:
        lr /= 13 ** 4
    elif epoch > 65:
        lr /= 13 ** 3
    elif epoch > 40:
        lr /= 13 ** 2
    elif epoch > 20:
        lr /= 13

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_all():

    faces_train_path_amalia = "./Ama"
    faces_train_path_coco = "./Coco"
    faces_train_path_nico = "./Nico"
    faces_test_path_ama = "./AmaTest"
    faces_test_path_coco = "./CocoTest"
    faces_test_path_nico = "./NicoTest"

    #load images from my drive

    # create labels for training

    batch_size = 32
    images_train, labels_train = load_images(ama_faces=faces_train_path_amalia, coco_faces=faces_train_path_coco,
                                             nico_faces=faces_train_path_nico)

    images_test, labels_test = load_images(ama_faces=faces_test_path_ama, coco_faces=faces_test_path_coco,
                                           nico_faces=faces_test_path_nico)

    train_dataset = ImageClassifierDataset(images_train, labels_train)
    test_dataset = ImageClassifierDataset(images_test, labels_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    cuda_avail = torch.cuda.is_available()

    # use a Resnet18 model
    model = models.resnet18(pretrained=True)

    if cuda_avail:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    train(num_epochs=150, model=model, loss_function=loss_fn, optimizer=optimizer, cuda_avail=cuda_avail,
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
    model_name = "finalized_model_customNetwork.sav"
    test_transformations = transforms.Compose([
        transforms.Resize((200 , 200)), transforms.CenterCrop((150,150)), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(path)
    image = image.convert('RGB')

    image = test_transformations(image)
    image = Variable(image.unsqueeze(0))


    model = pickle.load(open(model_name, "rb"))
    model.eval()

    output = model(image)
    #
    _, prediction = torch.max(output.data, 1)
    result = prediction[0].item()

    vector = [0,0,0,0]
    for item in prediction:
        vector[item.item()] += 1

    result = vector.index(max(vector))
    # print the labels of the image


    if (result == 0):
        print("Amalia")
    elif (result == 1):
        print("Nico")
    elif (result == 2):
        print("Coco")
    else:
        print("Error")

def sendMessagetoFirebase(message):
    url = "https://fcm.googleapis.com/fcm/send"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=AAAAX-_X_yY:APA91bG-_X_yY'
    }
    payload = {
        "to": "/topics/all",
        "notification": {
            "title": "Face Recognition",
            "body": message
        }
    }
    response = requests.request("POST", url, headers=headers, json=payload)
    print(response.text)

def retrieveMessageFromFirebase():
    url = "https://fcm.googleapis.com/fcm/send"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=AAAAX-_X_yY:APA91bG-_X_yY'
    }
    payload = {
        "to": "/topics/all",
        "notification": {
            "title": "Face Recognition",
            "body": "Hello"
        }
    }
    response = requests.request("POST", url, headers=headers, json=payload)
    print(response.text)
if __name__ == "__main__":
      # sendMessagetoFirebase("Please look at the camera")
      # retrieveMessageFromFirebase()
      #train_all()
      while True:
          filename = input("filepath:\n>")
          image = Image.open(filename)
          image.show()
          test_on_image(filename)

      # resize to smaller but maintain aspect ratio





# connect rest api to flutter
# send a string to firebase server
# use  rest api to send requests
# retrain models to see the results