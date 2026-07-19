# Домашнее задание №8 — трансформеры для RuCoLA

## Задача

Решить задачу бинарной классификации лингвистической приемлемости русских предложений и сравнить несколько подходов на основе трансформеров.

## Что выполнено

- подготовлены train/validation/test на основе RuCoLA;
- выполнен fine-tuning RuBERT-large как классификатора;
- исследован RuGPT-3 large в zero-shot и few-shot режимах с несколькими промптами;
- выполнен fine-tuning RuT5-base в постановке text-to-text;
- подходы сравнены по Accuracy, F1 и Matthews correlation coefficient;
- проведён анализ ошибок и формируются таблицы с предсказаниями.

## Данные и модели

Датасет: [RussianNLP/RuCoLA](https://github.com/RussianNLP/RuCoLA). Используются модели Hugging Face `ai-forever/ruBert-large`, `ai-forever/rugpt3large_based_on_gpt2` и `ai-forever/ruT5-base`. Полный режим рассчитан на GPU; в ноутбуке также предусмотрен облегчённый режим проверки.

## Запуск

Основные зависимости: `torch`, `transformers`, `datasets`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
jupyter notebook Homework_08.ipynb
```
