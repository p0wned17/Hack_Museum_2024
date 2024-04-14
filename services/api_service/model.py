import cv2
import numpy as np
import torch


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
