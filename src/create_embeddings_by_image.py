import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class ModelWrapper:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.half().eval()

    def predict(self, image: np.ndarray):
        img_tensor = self._preprocess_image(image)

        output = self._infer(img_tensor)
        return output

    def _preprocess_image(self, image: np.ndarray, input_size=256):
        image = cv2.resize(
            image, (input_size, input_size), interpolation=cv2.INTER_LANCZOS4
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image).transpose((2, 0, 1))
        image = torch.from_numpy(image).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        return image.unsqueeze(0)

    @torch.inference_mode()
    def _infer(self, img_tensor):
        output = self.model(img_tensor.half().to(self.device))

        return output


# Создание экземпляра класса
model_path = "model"
model_wrapper = ModelWrapper(model_path)

df = pd.read_csv("path_to_csv", delimiter=";")

main_path = "path_to_folder_with_images"

output_list = []

for i, v in tqdm(df.iterrows(), total=df.shape[0]):
    img_path = os.path.join(main_path, str(v["object_id"]), str(v["img_name"]))
    try:
        image = cv2.imread(img_path)

        output = model_wrapper.predict(image).float().cpu().numpy()[0]

        output_list.append(
            {
                "object_id": v["object_id"],
                "name": v["name"],
                "output": output.tolist(),
                "img_name": v["img_name"],
                "description": v["description"],
                "group": v["group"],
            }
        )
    except:
        output_list.append(
            {
                "object_id": v["object_id"],
                "output": [0] * 2048,
                "img_name": v["img_name"],
                "description": v["description"],
                "group": v["group"],
            }
        )

with open("output.json", "w") as json_file:
    json.dump(output_list, json_file)
