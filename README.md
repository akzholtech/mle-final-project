# Цель: Предсказать, какие товары предложить пользователю интернет-магазина.

- Бакет в S3: s3-student-mle-20251124-4e76e7f4c1

- Основные файли по проекту
    api.py — FastAPI приложение и endpoints
    data_loader.py запрос данных с БД
    schemas.py — Pydantic модели запроса и ответа
    service.py — бизнес-логика рекомендаций
    mlflow_loader.py — загрузка модели из MLflow
    recommender.py — класс-обёртка над ALS

Чтобы запустить проект нужны следующие шаги:

git clone https://github.com/akzholtech/mle-final-project.git

cd mle-final-project

docker compose up --build

В ходе разработки были использованы инструменты такие как:

- Airflow
- Mlflow
- FastAPI
- база данных PostgreSQL


Для задачи рекомендаций была использована модель ALS из библиотеки implicit.
Оценка качества модели проводилась с использованием метрик recall и precision.

Модель будет периодически дообучаться на обновлённых данных — каждую неделю по понедельникам в 03:00

