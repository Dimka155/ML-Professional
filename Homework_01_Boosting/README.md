# Домашнее задание №1 — сравнение алгоритмов бустинга

## Задача

Сравнить реализации градиентного бустинга на задаче бинарной классификации оттока клиентов и подобрать гиперпараметры лучшей модели.

## Что выполнено

- проведены EDA и предобработка признаков;
- обучены `GradientBoostingClassifier`, XGBoost, CatBoost и LightGBM;
- модели сравнены по ROC-AUC и accuracy;
- для выбранной модели выполнен подбор гиперпараметров;
- построены ROC-кривые и проведена интерпретация результатов.

## Данные

Используется набор **Telco Customer Churn** (`blastchar/telco-customer-churn`). Ноутбук загружает данные через `kagglehub`; для загрузки может потребоваться настроенный доступ к Kaggle.

## Запуск

Основные зависимости: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `catboost`, `lightgbm`, `kagglehub`.

```bash
jupyter notebook Homework_01.ipynb
```
