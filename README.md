 Этот код состоит из трех основных файлов, которые решают задачи обучения модели для распознавания лиц, создания базы эмбеддингов лиц и распознавания лиц в реальном времени с использованием камеры. Вот краткое описание каждого файла:

 1. train.py — Обучение модели
Задача: обучение модели для получения эмбеддингов лиц.
Используемая модель: FaceEmbeddingModel, которая используется для извлечения признаков лиц.
Потери: ArcFaceLoss — специализированная функция потерь для задач распознавания лиц.
Процесс: загружается датасет, на основе которого обучается модель, и после завершения обучения сохраняется файл модели face_model.pth.
 2. create_embeddings.py — Создание базы лиц
Задача: для каждого человека в датасете вычисляется среднее значение эмбеддингов всех его изображений, и эти данные сохраняются в файл face_database.npy.
Это позволяет нам создать базу для дальнейшего сопоставления лиц в реальном времени.
 3. real_time_recognition.py — Распознавание лиц в реальном времени
Задача: использование камеры для захвата изображений лиц, преобразование их в эмбеддинги и их сравнение с базой лиц для нахождения наиболее похожего лица.
Для распознавания используется модель, обученная ранее, и база эмбеддингов, созданная в предыдущем шаге.
Используется OpenCV для обработки видеопотока с камеры и отображения результатов распознавания.
# Структура проекта:
Модели и датасеты:
dataset/ — содержит изображения лиц для обучения и создания базы эмбеддингов.
model.py — определение архитектуры модели FaceEmbeddingModel и функции потерь ArcFaceLoss.
Файлы обучения и использования модели:
train.py — обучение модели.
create_embeddings.py — создание базы лиц.
real_time_recognition.py — реальное распознавание лиц с камеры.
# Важные шаги:
Запуск обучения модели:

Для начала обучите модель, запустив train.py. После завершения обучения у вас будет сохраненный файл модели face_model.pth.
Создание базы лиц:

После обучения модели, используйте create_embeddings.py для создания базы эмбеддингов всех лиц в вашем датасете.
Распознавание лиц в реальном времени:

Запустите real_time_recognition.py, чтобы начать распознавание лиц в реальном времени через веб-камеру.
# Зависимости:
torch — для работы с моделями машинного обучения.
opencv-python — для работы с видеопотоками.
numpy — для обработки эмбеддингов.
scikit-learn — для вычисления косинусного сходства между эмбеддингами.
# Настройки:
Убедитесь, что у вас настроена CUDA, если хотите использовать GPU для обучения и распознавания.
Путь к датасету DATASET_PATH в каждом файле нужно установить корректно, чтобы скрипты могли загрузить изображения.
Это решение создает эффективную систему для обучения, создания базы и использования модели для распознавания лиц.
