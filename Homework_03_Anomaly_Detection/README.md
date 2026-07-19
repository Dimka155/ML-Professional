# Домашнее задание №3 — поиск аномалий в транзакциях

## Задача

Найти мошеннические банковские транзакции несколькими unsupervised-методами и сравнить качество обнаружения аномалий.

## Что выполнено

- реализовано робастное диагональное расстояние Махаланобиса;
- применены HBOS и Isolation Forest;
- метки мошенничества не используются при обучении и выборе порогов;
- модели оценены по precision, recall, F1, balanced accuracy, ROC-AUC и PR-AUC;
- построены матрицы ошибок и t-SNE-визуализация.

## Данные

Набор [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Файл `creditcard.csv` необходимо разместить рядом с ноутбуком.

## Запуск

Основные зависимости: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
jupyter notebook Homework_03.ipynb
```
