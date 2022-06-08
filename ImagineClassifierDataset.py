import torch

from torchvision import transforms
from torch.utils.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ImageClassifierDataset(Dataset):
    def __init__(self, image_list, image_classes):
        self.images = []
        self.labels = ['amalia', 'coco', 'nico','no_face']
        self.classes = list(set(image_classes))
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        self.image_size = 32

        # what is this
        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size), transforms.CenterCrop(self.image_size), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        for image, image_class in zip(image_list, image_classes):
            transformed_image = self.transforms(image)
            self.images.append(transformed_image)
            label = self.class_to_label[image_class]
            self.labels.append(label)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)