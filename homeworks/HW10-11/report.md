# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

**Часть A (классификация изображений):**
- Выбран датасет **CIFAR10** — 60,000 цветных изображений 32×32 пикселей, 10 классов
- Реализованы и обучены 4 модели:
  - **C1**: SimpleCNN без аугментации данных
  - **C2**: SimpleCNN с аугментацией данных (flip, rotation, color jitter)
  - **C3**: MobileNetV2 (transfer learning, замороженный backbone)
  - **C4**: MobileNetV2 (fine-tuning последних слоёв)
- Проведено сравнение подходов, выбрана лучшая модель

**Часть B (детекция объектов):**
- Выбран датасет **Pascal VOC 2012** (validation set)
- Использована предобученная модель **Faster R-CNN ResNet50 FPN**
- Исследовано влияние порога уверенности (score threshold) на метрики:
  - **V1**: score_threshold = 0.3
  - **V2**: score_threshold = 0.7
- Рассчитаны precision, recall, mean IoU для обоих режимов

## 2. Среда и воспроизводимость

**Технические детали:**
- **Python:** 3.12
- **PyTorch / TorchVision:** 2.x
- **Устройство:** CPU (оптимизировано для работы без GPU)
- **Seed:** 42 (зафиксирован для random, numpy, torch)
- **Время выполнения:** ~40-50 минут на CPU

**Как воспроизвести:**
```bash
# 1. Установить зависимости
pip install torch torchvision pandas matplotlib

# 2. Склонировать репозиторий
git clone <repository_url>
cd homeworks/HW10-11

# 3. Запустить ноутбук
jupyter notebook HW10-11.ipynb
# или выполнить построчно в VS Code / PyCharm

# 4. Все артефакты сохранятся в папке artifacts/

3. Данные
3.1. Часть A: классификация

    Датасет: CIFAR10
    Объём: 60,000 изображений (32×32, RGB, 10 классов)
    Разделение:
        Train: 40,000 (80% от original train)
        Validation: 10,000 (20% от original train)
        Test: 10,000 (original test set)
    Классы: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Базовые transforms:
Комментарий: CIFAR10 — классический бенчмарк. Малый размер изображений позволяет быстро обучать модели на CPU. 10 классов обеспечивают достаточное разнообразие для демонстрации преимуществ transfer learning и data augmentation.
3.2. Часть B: детекция объектов

    Датасет: Pascal VOC 2012 (validation set)
    Трек: detection
    Объём: ~1,000 изображений, 20 классов объектов
    Классы: person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, diningtable, pottedplant, sofa, tvmonitor
    Ground truth: bounding boxes из XML-аннотаций VOC
    Предсказания: предобученная Faster R-CNN ResNet50 FPN (COCO weights)

Комментарий: Pascal VOC — стандартный датасет для оценки детекции. 20 разнообразных классов и качественные аннотации делают его подходящим для демонстрации работы детектора.
4. Часть A: модели и обучение (C1-C4)
C1 (SimpleCNN без аугментаций)

    Архитектура: 3 свёрточных блока (32→64→128 фильтров) + BatchNorm + ReLU + MaxPool
    Fully connected: 256 нейронов + dropout 0.5 + выход на 10 классов
    Transform: базовый (без аугментаций)

C2 (SimpleCNN с аугментациями)

    Та же архитектура, что C1
    Transform: с аугментациями (flip, rotation, color jitter)

C3 (MobileNetV2 — head only)

    Предобученный MobileNetV2 с ImageNet
    Backbone заморожен (requires_grad=False)
    Заменён последний слой: Linear(1280 → 10)
    Обучается только head

C4 (MobileNetV2 — fine-tuning)

    Предобученный MobileNetV2 с ImageNet
    Разморожен последний блок features.18 и classifier
    Остальные слои заморожены
    Позволяет адаптировать высокоуровневые признаки под задачу

Общие параметры обучения

    Loss: CrossEntropyLoss
    Optimizer: Adam (lr=1e-3 для C1-C3, lr=1e-4 для C4)
    Batch size: 64 (CPU) / 128 (GPU)
    Epochs: 10 (все эксперименты)
    Критерий выбора лучшей модели: максимальная val_accuracy

5. Часть B: постановка задачи и режимы оценки (V1-V2)
Detection track

    Модель: Faster R-CNN ResNet50 FPN (предобучена на COCO)
    V1: score_threshold = 0.3 (низкий порог — больше детекций, больше false positives)
    V2: score_threshold = 0.7 (высокий порог — меньше детекций, выше точность)
    Precision / Recall

    TP: предсказанный box совпал с GT (IoU ≥ 0.5)
    FP: предсказанный box не совпал ни с одним GT
    FN: GT box не был найден
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

6. Результаты
Ссылки на файлы в репозитории

    Таблица результатов: homeworks/HW10-11/artifacts/runs.csv
    Лучшая модель части A: homeworks/HW10-11/artifacts/best_classifier.pt
    Конфиг лучшей модели: homeworks/HW10-11/artifacts/best_classifier_config.json
    Графики: homeworks/HW10-11/artifacts/figures/

Короткая сводка

    Лучший эксперимент части A: C4 (MobileNetV2 fine-tuning)
    Лучшая val_accuracy: ~90%
    Итоговая test_accuracy: ~89%
    Что дали аугментации (C2 vs C1): +2% val, +2% test
    Что дал transfer learning (C3/C4 vs C1/C2): +11-14% improvement
    Fine-tuning vs Head-only (C4 vs C3): +3% improvement
    V1 (threshold=0.3): Precision ~0.55, Recall ~0.68, mIoU ~0.62
    V2 (threshold=0.7): Precision ~0.78, Recall ~0.45, mIoU ~0.71
    Интерпретация: компромисс precision/recall регулируется порогом уверенности

7. Анализ
Простая CNN (C1, C2)

    Быстрое переобучение: train accuracy ~95%, val ~76-78%
    Разрыв train-val: ~15-20% (сильное переобучение)
    Аугментации снизили переобучение, но недостаточно

Transfer Learning (C3, C4)

    Быстрая сходимость: уже на 1-й эпохе >70%
    Малый разрыв train-val: ~3-5%
    Fine-tuning дал дополнительное улучшение +3%

Precision-Recall Trade-off (Part B)

    При низком threshold: высокий recall, умеренная precision
    При высоком threshold: высокая precision, низкий recall
    Выбор порога зависит от стоимости ошибок (FP vs FN)

Типичные ошибки детектора

    False negatives: маленькие объекты, перекрывающиеся объекты
    False positives: части объектов, похожие текстуры

8. Итоговый вывод

    Базовый конфиг классификации: MobileNetV2 с fine-tuning последних слоёв (C4) — оптимальный баланс между точностью (~90%) и скоростью обучения (~10 мин на CPU).
    Главное про transfer learning: предобученные модели значительно ускоряют сходимость и улучшают качество, особенно на маленьких датасетах. Fine-tuning последних слоёв даёт дополнительное улучшение (+3%) по сравнению с обучением только head, но требует меньшего learning rate (1e-4).
    Главное про detection и метрики: precision и recall — комплементарные метрики, их баланс зависит от порога уверенности. Mean IoU хорошо характеризует точность локализации. Для полноценной оценки нужно использовать PR-кривую и Average Precision (AP).

9. Приложение (опционально)
Оптимизации для CPU

    MobileNetV2 вместо ResNet18 (в 5-10 раз быстрее)
    Размер изображений: 64×64 вместо 224×224 (в 12 раз быстрее)
    Batch size: 64 для CPU, 128 для GPU
    NUM_WORKERS=0 для стабильности на Windows

Дополнительные графики

    ./artifacts/figures/classification_curves_best.png — кривые обучения лучшей модели
    ./artifacts/figures/classification_compare.png — сравнение всех экспериментов
    ./artifacts/figures/augmentations_preview.png — примеры аугментаций
    ./artifacts/figures/detection_examples.png — примеры детекции
    ./artifacts/figures/detection_metrics.png — precision/recall vs threshold