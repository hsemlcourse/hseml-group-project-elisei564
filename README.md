[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание рыночной стоимости футболиста

**Студент:** Панов Елисей Николаевич

**Группа:** 234


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Данные](#данные)
4. [Запуски](#запуск)
5. [Результаты](#результаты)
6. [Вывод](#выводы)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Регрессия - предсказание зарплаты футболиста (`P_WageEUR`).

**Датасет:** FIFA 21 Players & Teams - Full Database  
https://www.kaggle.com/datasets/cashncarry/fifa-21-players-teams-full-database

**Цель:** построить модель, которая по характеристикам игрока и клуба предсказывает его зарплату.

**Целевая метрика:** RMSE в лог-пространстве


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-elisei564.git
cd hseml-group-project-elisei564

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```
- Перейдите по ссылке на [Kaggle](https://www.kaggle.com/datasets/cashncarry/fifa-21-players-teams-full-database).
- Нажмите кнопку Download.
- Скачайте и распакуйте архив.
- Перенесите файлы players_fifa21.csv и teams_fifa21.csv в директорию проекта:
```
└── data/
    └── raw/
        ├── players_fifa21.csv
        └── teams_fifa21.csv
```
```bash

# 5. Запуск линтера
flake8 src tests app

# 6. Запуск тестов
docker compose up --build test

# 7. Сборка и запуск preprocessing pipeline
docker compose up pipeline

# 8. Обучение модели
docker compose up train

# 9. Запуск веб-интерфейса
docker compose up api streamlit
```


## Результаты

| Модель | RMSE | MAE | R2 | MAPE |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.4827 | 0.3624 | 0.8652 | 39.5157 |
| **Ridge Regression** | 0.4828 | 0.3624 | 0.8651 | 39.5279 |
| **KNN (k=10)** | 0.4877 | 0.3689 | 0.8624 | 41.5176 |
| **Median baseline** | 1.3158 | 1.0849 | -0.0020 | 144.2404 |



## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)