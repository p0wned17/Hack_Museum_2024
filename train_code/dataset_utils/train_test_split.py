import pandas as pd

# Загрузите данные
data = pd.read_csv("/mnt/hack_museums/clean_train_data.csv", delimiter=";")

# Количество изображений для выборки на каждую группу
samples_per_group = 2000 // 15

# Выборка данных для каждой группы с сохранением индексов
selected = data.groupby("group").sample(n=samples_per_group, random_state=42)


# Стратифицированное разделение каждой группы на gallery и query
def split_gallery_query(group_data, fraction=0.5):
    # Перемешиваем данные в группе
    group_data = group_data.sample(frac=1, random_state=42).reset_index(drop=True)
    # Делим данные на две части
    split_index = int(len(group_data) * fraction)
    gallery_part = group_data[:split_index]
    query_part = group_data[split_index:]
    return gallery_part, query_part


# Применяем функцию к каждой группе
gallery_list = []
query_list = []

for group_name, group_data in selected.groupby("group"):
    gallery_part, query_part = split_gallery_query(group_data)
    gallery_list.append(gallery_part)
    query_list.append(query_part)

# Объединяем все части в одни DataFrame
gallery = pd.concat(gallery_list).reset_index(drop=True)
query = pd.concat(query_list).reset_index(drop=True)

# Сохранение данных
gallery.to_csv("gallery.csv", sep=";", index=False)
query.to_csv("query.csv", sep=";", index=False)

# Удаление выбранных данных из основного датасета для создания обучающего набора
remaining_data = data.drop(selected.index)
remaining_data.to_csv("training_data.csv", sep=";", index=False)
