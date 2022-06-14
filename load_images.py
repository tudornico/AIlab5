import pathlib

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True




def load_images(ama_faces,coco_faces,nico_faces):
    images = []
    classes = []

    for path in pathlib.Path(ama_faces).iterdir():
        if path.is_file():
            image = Image.open(path)
            image.thumbnail((150, 150))
            images.append(image.convert('RGB'))
            classes.append('ama')

    for path in pathlib.Path(coco_faces).iterdir():
        if path.is_file():
            image = Image.open(path)
            image.thumbnail((150, 150))
            images.append(image.convert('RGB'))
            classes.append('coco')

    for path in pathlib.Path(nico_faces).iterdir():
        if path.is_file():
            image = Image.open(path)
            image.thumbnail((150, 150))
            images.append(image.convert('RGB'))
            classes.append('nico')

    return images, classes
