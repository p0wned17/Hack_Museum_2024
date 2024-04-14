import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data
from turbojpeg import TurboJPEG


turbo_jpeg = TurboJPEG()


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError("Failed to read {}".format(image_file))
    return img


def set_labels_to_range(labels):
    """
    set the labels so it follows a range per level of semantic
    usefull for example for CSLLoss
    """
    new_labels = []
    print(labels.shape)

    unique_group = sorted(set(labels[:, 0]))
    print("unique_group", len(unique_group))

    conversion = {x: i for i, x in enumerate(unique_group)}
    new_lvl_labels = [conversion[x] for x in labels[:, 0]]
    new_labels.append(new_lvl_labels)

    unique_classes = sorted(set(labels[:, 1]))
    print("unique_classes", len(unique_classes))
    conversion = {x: i for i, x in enumerate(unique_classes)}
    new_lvl_labels = [conversion[x] for x in labels[:, 1]]
    new_labels.append(new_lvl_labels)

    return np.stack(new_labels, axis=1)


class WBFlexDataset(data.Dataset):
    def __init__(self, root, annotation_file, transform, val=None):
        self.root = root
        self.val = val

        table = pd.read_csv(annotation_file, delimiter=";")

        img_names = table["img_name"].tolist()
        roots = table["object_id"].tolist()

        self.paths = [f"{self.root}/{x}/{y}" for x, y in zip(roots, img_names)]
        labels = table[["group", "object_id"]].to_numpy()

        self.classes_count = len(table["object_id"].unique())
        print("длина датасета", len(table))
        self.groups_count = len(table["group"].unique())

        self.labels = set_labels_to_range(labels)
        self.transform = transform

    def __getitem__(self, index):
        cv2.setNumThreads(80)

        full_imname = self.paths[index]
        category_id, class_id = self.labels[index]

        try:
            with open(file=full_imname, mode="rb") as image_file:
                img = turbo_jpeg.decode(image_file.read(), pixel_format=0)

        except Exception:
            print(full_imname)

        img = self.transform(image=img)["image"]
        if self.val:
            return img, category_id
        return img, category_id, class_id

    def __len__(self):
        return len(self.paths)
