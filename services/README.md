# Проект для Хакатона: Поиск Музейных Предметов

## Описание задачи
Участникам предлагается разработать прототип системы (программного модуля), обеспечивающего возможность поиска музейных предметов, в наибольшей степени соответствующих заданному пользователем изображению, классификации предметов и формирования описания музейных предметов.

## Запуск Сервера

Запустите сервер FastAPI, используя следующую команду:

```bash
cd services/api_service
python main.py
```
----
* важное примечание, не забудь указать в файлах свои переменные.
## Запуск Телеграмм Бота

Запустите сервер FastAPI, используя следующую команду:

```bash
cd services/telegram_bot_service
python telegram_bot.py
```

----
* важное примечание, не забудь указать в файлах свои переменные.

## Запуск Музейного приложения на андроиде

Для того, чтобы запустить у себя на телефоне приложение

1. Скачать приложение по этому пути `services/museum_app/build/apk/app-release.apk`

2. С помощью пакетного менеджера установить и начать пользоваться.

----
* важное примечание, не забудь указать в файлах свои переменные.

## Использование API 

Отправьте POST-запрос на `http://localhost:3333/predict/` с изображением документа в формате файла. Пример использования с помощью `curl`:

```bash
curl -X 'POST' \
  'http://localhost:3333/predict/' \
  -F 'file=@path_to_your_document.jpg' \
  -H 'accept: application/json'
  ```