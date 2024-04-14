import io
import os

import cv2

# import face_lib_pytorch
import numpy as np
import telebot
from classifier import classify
from generate_description_by_photo.py import get_pred
from PIL import Image
from pymilvus import Collection, connections
import time


# Создание экземпляра класса
model_path = "out model"
model_wrapper = ModelWrapper(model_path)

TOKEN = "token_telegram_bot"

bot = telebot.TeleBot(TOKEN)

connections.connect(host="uri", port="port")


def process_image(file_info, message):
    
    downloaded_file = bot.download_file(file_info.file_path)
    fp = io.BytesIO(downloaded_file)
    image_pill = Image.open(fp)

    image1 = np.array(image_pill)

    output = model_wrapper.predict(image1).float().cpu().numpy()
   
    face1_data = output[0] 

    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10},
    }

    collection = Collection("example_new_2")

    start_time = time.time()
    similarity_search_result = collection.search(
        data=[face1_data],
        anns_field="output",
        param=search_params,
        limit=10,
        output_fields=["img_name", "description", "group", "object_id", "name"],
    )

    end_time = time.time() - start_time
    print("Время работы поиска в бд: ", end_time)
    bb = []

    for idx, hit in enumerate(similarity_search_result[0]):
        score = hit.distance
        description = hit.entity.description
        object_id = hit.entity.object_id
        img_name = hit.entity.img_name
        group = hit.entity.group
        name = hit.entity.name
        print(
            f"{idx + 1}. {description} {object_id} {img_name} {group} (distance: {score})"
        )

        base = {}

        base["id"] = idx + 1
        base["name"] = name
        base["description"] = description
        base["object_id"] = object_id
        base["img_name"] = img_name
        base["group"] = group
        base["score"] = score

        image1 = cv2.imread(
            os.path.join("/home/gaus/projects/base/ml/train", str(object_id), img_name)
        )

        # Изменение размера изображения до 1200x1200 пикселей
        height, width = image1.shape[:2]

        # Определяем масштаб для изменения размера с сохранением пропорций
        max_dimension = max(height, width)
        scale_factor = 1200 / max_dimension

        # Изменение размера изображения с сохранением пропорций
        resized_image = cv2.resize(
            image1, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
        )

        # Путь для сохранения обработанного изображения

        # Сохраняем обработанное изображение
        cv2.imwrite(
            os.path.join("/home/gaus/projects/base/ml/train", str(object_id), img_name),
            resized_image,
        )
        # Отправляем обработанное изображение обратно пользователю

        with open(
            os.path.join("/home/gaus/projects/base/ml/train", str(object_id), img_name),
            "rb",
        ) as photo:
            bot.send_photo(message.chat.id, photo)

        bot.send_message(
            message.chat.id,
            f"<b>Имя:</b>{name}\n<b>Описание предмета:</b>\n{description}\n<b>Группа:</b> {group}\n<b>Сходство: {score}%</b>",
            parse_mode="HTML",
        )

        bb.append(base)

    desc = get_pred(image_pill)

    classification_result = classify(bb)

    bot.send_message(
        message.chat.id,
        f"<b>Классификатор категорий</b>: {classification_result['group']}\n<b>Описание предмета сгенерированное AI для отправленной вами фоткой:</b> {desc[0]}",
        parse_mode="HTML",
    )

    return score, description, object_id, img_name, group


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне изображение, и я его обработаю.")


# Обработчик сообщений с изображениями
@bot.message_handler(content_types=["photo"])
def handle_image(message):
    # Получаем объект фотографии с максимальным разрешением
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    # downloaded_file = bot.download_file(file_info.file_path)
    # Получаем путь к файлу изображения

    # Обрабатываем изображение
    score, description, object_id, img_name, group = process_image(file_info, message)


bot.polling()

# Импортируем бибилиотеку
