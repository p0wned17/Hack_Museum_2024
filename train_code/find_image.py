import os


def find_image_by_name(directory, target_image_name):
    """
    Функция для поиска изображения по названию в заданной директории и всех поддиректориях.
    :param directory: Путь к директории, в которой нужно искать.
    :param target_image_name: Название файла изображения, включая расширение.
    :return: Полный путь к найденному изображению или None, если изображение не найдено.
    """
    for root, dirs, files in os.walk(directory):
        if target_image_name in files:
            return os.path.join(root, target_image_name)
    return None


# Укажите путь к вашей начальной директории
start_directory = "path"
# Укажите название файла изображения, которое нужно найти
image_name = "image"

found_image_path = find_image_by_name(start_directory, image_name)
if found_image_path:
    print(f"Изображение найдено: {found_image_path}")
else:
    print("Изображение не найдено.")
