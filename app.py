import sys
import cv2
import numpy as np
import os
import tensorflow as tf


def load_model(model_path):
    """Загружает модель TensorFlow из указанного пути.

    Args:
        model_path: Путь к модели TensorFlow.

    Returns:
        Загруженная модель TensorFlow.
    """
    return tf.keras.models.load_model(model_path)


def process_video(video_path, output_size, models):
    """Обрабатывает видео, применяя указанные модели, и сохраняет результат.

    Args:
        video_path: Путь к видеофайлу.
        output_size: Размер выходного видео (ширина, высота).
        models: Список моделей TensorFlow для последовательного применения к кадрам видео.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.splitext(video_path)[0] + '_corrected.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 30.0, output_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Изменение размера кадра
        original_size = frame.shape[1], frame.shape[0]
        frame = cv2.resize(frame, output_size)
        frame = frame / 255.0  # Нормализация кадра к диапазону [0,1]

        # Последовательное применение моделей
        for model in models:
            frame = model.predict(np.expand_dims(frame, axis=0))[0]

        frame = (frame * 255).astype(np.uint8)  # Денормализация к диапазону [0,255]
        frame = cv2.resize(frame, original_size)  # Восстановление исходного размера
        out.write(frame)

    cap.release()
    out.release()
    print(f"Обработанное видео сохранено как {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python app.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_size = (160, 90)

    # Загрузка моделей в определенном порядке
    models_path = 'models/'
    denoise_model = load_model(os.path.join(models_path, 'denoise_best.keras'))
    color_model = load_model(os.path.join(models_path, 'color_best.keras'))
    upscale_model = load_model(os.path.join(models_path, 'upscale_best.keras'))
    models = [denoise_model, color_model, upscale_model]

    process_video(video_path, output_size, models)
