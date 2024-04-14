import io
import os

import numpy as np
import uvicorn
from classifier import classify
from fastapi import FastAPI, File, UploadFile
from generate_description_by_photo.py import get_pred
from model_inference import ModelWrapper
from PIL import Image
from pymilvus import Collection, connections
import cv2

# Создание экземпляра класса
model_path = "out_model"
model_wrapper = ModelWrapper(model_path)

TOKEN = "tlgr_token"

connections.connect(host="uri", port="port")

app = FastAPI()


@app.post("/predict/", tags=["Predict"], summary="Predict")
async def upload(
    file: UploadFile = File(
        ...,
        description="Выберите файл для загрузки",
    ),
):
    data = await file.read()

    pil_image = Image.open(io.BytesIO(data))
    image11 = np.array(pil_image)

    output = model_wrapper.predict(image11).float().cpu().numpy()

    data = output[0]

    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10},
    }

    collection = Collection("example")

    similarity_search_result = collection.search(
        data=[data],
        anns_field="output",
        param=search_params,
        limit=10,
        output_fields=["img_name", "description", "group", "object_id", "name"],
    )

    bb = []
    for idx, hit in enumerate(similarity_search_result[0]):
        score = hit.distance
        description = hit.entity.description
        object_id = hit.entity.object_id
        name = hit.entity.name
        img_name = hit.entity.img_name
        group = hit.entity.group

        base = {}  # Создаем новый словарь для каждой итерации

        base["id"] = idx + 1
        base["name"] = name
        base["description"] = description
        base["object_id"] = object_id
        base["img_name"] = img_name
        base["group"] = group
        base["score"] = score

        image = cv2.imread(
            os.path.join("/home/gaus/projects/base/ml/train", str(object_id), img_name)
        )
        maxsize = (256, 256)
        imRes = cv2.resize(image, maxsize, interpolation=cv2.INTER_AREA)
        success, encoded_image = cv2.imencode(".png", imRes)
        content2 = encoded_image.tobytes()

        base["bytes_image"] = base64.b64encode(content2)

        bb.append(base)

    classification_result = classify(bb)

    result = {"classifier": classification_result, "topk": bb}
    return result


@app.post(
    "/predict_desc_by_ai/", tags=["predict_desc_by_ai"], summary="predict_desc_by_ai"
)
async def upload(
    file: UploadFile = File(
        ...,
        description="Выберите файл для загрузки",
    ),
):
    data = await file.read()
    pil_image = Image.open(io.BytesIO(data))
    prediction_on_request_image = get_pred(pil_image)
    bb = {"generated_by_ai": prediction_on_request_image}

    return bb

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8099))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
        workers=2,
    )
