# ml_model_toxic_comments_deployment

# Пример того, как построить модель машинного обучения с использованием [Fast API](https://fastapi.tiangolo.com/) и [Docker](https://www.docker.com/)

## Цель тренировки модели
* Добиться метрики точности 0.95 при хорошей полноте, для того что бы вероятность удалить не токсичный комментарий была низкая.

## Как запустить
* Установить и запустить Docker
* Создать образ Docker с помощью `docker build . -t toxic_comments_server`
* Запустить контейнер Docker, используя `docker run --rm -it -p 80:80 toxic_comments_server`
* Перейти на `http://127.0.0.1:80/docs` чтобы увидеть все доступные методы API

## Исходный код
* [server.py](server.py) содержит логику API
* [train.py](train.py) обучает модель с использованием набора данных labeled
* [Dockerfile](Dockerfile) описывает образ Docker, который используется для запуска API
* [requirements.txt](requirements.txt) содержит версии нужных библиотек 
* [labeled.csv](labeled.csv) датасет с размеченными токсичными и не токсичными комментариями
