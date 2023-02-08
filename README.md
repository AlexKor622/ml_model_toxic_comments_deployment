# ml_model_toxic_comments_deployment

# Пример того, как построить модель машинного обучения с использованием [Fast API](https://fastapi.tiangolo.com/) and [Docker](https://www.docker.com/)

## Как запустить
* Установить и запустить Docker
* Создайте образ Docker с помощью `docker build . -t iris_server`
* Запустить контейнер Docker, используя `docker run --rm -it -p 80:80 iris_server`
* Перейти на `http://127.0.0.1:80/docs` чтобы увидеть все доступные методы API

## Исходный код
* [server.py](server.py) содержит логику API
* [train.py](train.py) обучает фиктивную модель с использованием набора данных Iris
* [query_example.py](query_example.py) помогает проверить правильность работы контейнера docker
* [Dockerfile](Dockerfile) описывает образ Docker, который используется для запуска API
