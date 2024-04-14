# Проект для Хакатона: Поиск Музейных Предметов

## Описание задачи
Участникам предлагается разработать прототип системы (программного модуля), обеспечивающего возможность поиска музейных предметов, в наибольшей степени соответствующих заданному пользователем изображению, классификации предметов и формирования описания музейных предметов.

## Очистка изображений
Перед тем, как обучать нужно почистить датасет в папке train_code/dataset_utils

Выполнить 
```bash
python cleaning_dataset.py
```

Затем разбиваем на train и validation
Выполнить 
```bash 
python train_test_split
```

## Обучение модели
Для запуска обучения модели обратитесь к инструкции [запуск обучения модели](train_code/README.md). Внутри этой папки находится README файл с дополнительной информацией о процессе обучения.

## Этапы обработки данных
1. Создание эмбеддингов изображений: Используйте скрипт 
```bash 
src/create_embeddings_by_image.py
```
2. Конвертация эмбеддингов из JSON в CSV: Используйте скрипт 
```bash
src/json_embeddings_to_csv.py
```
3. Создание коллекции в Milvus: Используйте скрипт 
```bash
src/create_collection_milvus.py
```

## Загрузка данных в Milvus
Загрузите эмбеддинги в базу данных Milvus, которую можно развернуть с помощью Docker и файла на данном пути  
```bash
docker/docker-compose.yaml
```
Используя команду 
```bash
docker/docker-compose up
```

## Сервисы
После выполнения вышеперечисленных этапов, у нас имеется два сервиса в папке `services`:
- **FastAPI**: Сервер на fastapi.
- **Telegram bot**: Телеграмм бот.
- **Museum app**: Кроссплатформенное приложение


## Запуск сервисов
Для запуска сервисов необходимо следовать инструкциям в соответствующих репозиториях.

---

**Примечание:** Подробная информация о запуске и использовании каждого сервиса будет предоставлена в соответствующих репозиториях проекта.