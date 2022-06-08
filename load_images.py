import pathlib

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_images(faces, no_faces):
    images = []
    classes = []
    for path in pathlib.Path(faces).iterdir():
        if path.is_file():
            images.append(Image.open(path).convert('RGB'))
            classes.append(1)

    for path in pathlib.Path(no_faces).iterdir():
        if path.is_file():
            images.append(Image.open(path).convert('RGB'))
            classes.append(0)
    return images, classes
