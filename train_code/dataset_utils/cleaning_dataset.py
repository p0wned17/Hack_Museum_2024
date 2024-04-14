import pandas as pd
from turbojpeg import TurboJPEG

# from .augmentations import (
#     get_gallery_transform,
# )
turbo_jpeg = TurboJPEG()

# Путь к файлу с разметкой и папке с изображениями
data_path = "/mnt/hack_museums/dataset_utils/training_data.csv"
image_folder = "/mnt/hack_museums/train_dataset"
clean_data_path = "очищенный_файл.csv"

# Загрузка данных
data = pd.read_csv(data_path, delimiter=";")


# Функция для проверки изображения
def check_image(file_path):
    try:
        with open(file=file_path, mode="rb") as image_file:
            img = turbo_jpeg.decode(image_file.read(), pixel_format=0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                return False
            return True
    except Exception:
        return False
    # try:
    #     img = cv2.imread(file_path)
    #     # Проверяем, успешно ли загружено изображение
    #     if img is None:
    #         return False
    #     return True
    # except:
    #     return False


# Фильтрация данных
valid_rows = []
for index, row in data.iterrows():
    # image_path = os.path.join(image_folder, row["img_name"])
    object_id, image_name = row["object_id"], row["img_name"]
    image_path = f"{image_folder}/{object_id}/{image_name}"
    if check_image(image_path):
        valid_rows.append(row)
    else:
        print(f"Поврежденное изображение обнаружено и удалено: {row['img_name']}")

# Создание DataFrame из валидных строк
clean_data = pd.DataFrame(valid_rows)
clean_data.to_csv(clean_data_path, index=False, sep=";")

print(f"Очищенная разметка сохранена в {clean_data_path}")
