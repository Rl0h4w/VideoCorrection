import time
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import optuna
from optuna.integration import TFKerasPruningCallback
from tuner import NoiseRemovalOptunaModel, ColorCorrectionOptunaModel, RCANOptunaModel
from metrics import psnr, ssim, delta_e
from data import write_tfrecords, load_tfrecord_dataset
import gc

# Параметры
train_dir = 'data/train_sharp'
val_dir = 'data/val_sharp'
batch_size = 128
input_size = (160, 90)
upscale_factor = 3
output_size = (480, 270)
epochs = 30
save_path = 'models'


def check_and_create_tfrecords():
    """Проверяет и создает файлы TFRecord, если они не существуют."""
    if not os.path.exists('train_super_resolution.tfrecord'):
        write_tfrecords(train_dir, 'train_super_resolution.tfrecord', input_size, 'super_resolution')
    if not os.path.exists('val_super_resolution.tfrecord'):
        write_tfrecords(val_dir, 'val_super_resolution.tfrecord', input_size, 'super_resolution')


check_and_create_tfrecords()

# Настройка конфигурации TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.MirroredStrategy()

print(f'Количество устройств: {strategy.num_replicas_in_sync}')


def count_files(directory):
    """Подсчитывает количество файлов изображений в каталоге.
    
    Args:
        directory: Директория для подсчета файлов.

    Returns:
        Количество файлов изображений.
    """
    count = 0
    for subdir, _, files in os.walk(directory):
        count += len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    return count


train_steps_per_epoch = count_files(train_dir) // batch_size
val_steps_per_epoch = count_files(val_dir) // batch_size


def get_noise_removal_datasets():
    """Загружает наборы данных для удаления шума из файлов TFRecord.
    
    Returns:
        Кортеж из обучающего и валидационного наборов данных.
    """
    train_dataset_noise = load_tfrecord_dataset('train_noise_removal.tfrecord', batch_size, input_size)
    val_dataset_noise = load_tfrecord_dataset('val_noise_removal.tfrecord', batch_size, input_size)
    return train_dataset_noise, val_dataset_noise


def optimize_noise_removal(trial):
    """Оптимизация модели удаления шума с использованием Optuna.
    
    Args:
        trial: Текущий эксперимент Optuna.

    Returns:
        Значение потерь на валидационном наборе данных.
    """
    train_dataset_noise, val_dataset_noise = get_noise_removal_datasets()
    with strategy.scope():
        noise_removal_model = NoiseRemovalOptunaModel(input_shape=input_size + (3,))
        model = noise_removal_model.build(trial=trial)

        model.fit(
            train_dataset_noise,
            validation_data=val_dataset_noise,
            epochs=3,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            callbacks=[
                TensorBoard(log_dir='./logs/tuning_noise_removal', update_freq='epoch'),
                TFKerasPruningCallback(trial, 'val_loss')
            ],
            verbose=1
        )

        val_loss = model.evaluate(val_dataset_noise, steps=val_steps_per_epoch, verbose=0)
        return val_loss[0]


# Запуск оптимизации модели удаления шума
start_time = time.time()
study = optuna.create_study(direction='minimize')
study.optimize(optimize_noise_removal, n_trials=30)
end_time = time.time()
print(f"Время оптимизации гиперпараметров: {end_time - start_time} секунд")

best_params = study.best_trial.params

# Обучение лучшей модели удаления шума
train_dataset_noise, val_dataset_noise = get_noise_removal_datasets()
with strategy.scope():
    noise_removal_model = NoiseRemovalOptunaModel(input_shape=input_size + (3,))
    best_model = noise_removal_model.build(params=best_params)

    start_time = time.time()
    best_model.fit(
        train_dataset_noise,
        validation_data=val_dataset_noise,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[
            ModelCheckpoint(f"{save_path}/denoise_best.keras", save_best_only=True, save_weights_only=False, monitor="val_loss"),
            TensorBoard(log_dir='./logs/denoise_best', update_freq='epoch')
        ]
    )
    end_time = time.time()
    print(f"Время обучения: {end_time - start_time} секунд")

# Удаление модели удаления шума и набора данных из памяти
del best_model
gc.collect()


def get_color_correction_datasets():
    """Загружает наборы данных для коррекции цвета из файлов TFRecord.
    
    Returns:
        Кортеж из обучающего и валидационного наборов данных.
    """
    train_dataset_color = load_tfrecord_dataset('train_color_correction.tfrecord', batch_size, input_size)
    val_dataset_color = load_tfrecord_dataset('val_color_correction.tfrecord', batch_size, input_size)
    return train_dataset_color, val_dataset_color


def optimize_color_correction(trial):
    """Оптимизация модели коррекции цвета с использованием Optuna.
    
    Args:
        trial: Текущий эксперимент Optuna.

    Returns:
        Значение потерь на валидационном наборе данных.
    """
    train_dataset_color, val_dataset_color = get_color_correction_datasets()
    with strategy.scope():
        color_correction_model = ColorCorrectionOptunaModel(input_shape=input_size + (3,))
        model = color_correction_model.build(trial=trial)

        model.fit(
            train_dataset_color,
            validation_data=val_dataset_color,
            epochs=3,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            callbacks=[
                TensorBoard(log_dir='./logs/tuning_color_correction', update_freq='epoch'),
                TFKerasPruningCallback(trial, 'val_loss')
            ],
            verbose=1
        )

        val_loss = model.evaluate(val_dataset_color, steps=val_steps_per_epoch, verbose=0)
        return val_loss[0]


# Запуск оптимизации модели коррекции цвета
start_time = time.time()
study = optuna.create_study(direction='minimize')
study.optimize(optimize_color_correction, n_trials=30)
end_time = time.time()
print(f"Время оптимизации гиперпараметров: {end_time - start_time} секунд")

best_params = study.best_trial.params

# Обучение лучшей модели коррекции цвета
train_dataset_color, val_dataset_color = get_color_correction_datasets()
with strategy.scope():
    color_correction_model = ColorCorrectionOptunaModel(input_shape=input_size + (3,))
    best_model = color_correction_model.build(params=best_params)
    best_model.fit(
        train_dataset_color,
        validation_data=val_dataset_color,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[
            ModelCheckpoint(f"{save_path}/color_best.keras", save_best_only=True, save_weights_only=False, monitor="val_loss"),
            TensorBoard(log_dir='./logs/color_best', update_freq='epoch')
        ]
    )

# Удаление модели коррекции цвета и набора данных из памяти
del best_model
gc.collect()


def get_super_resolution_datasets():
    """Загружает наборы данных для суперразрешения из файлов TFRecord.
    
    Returns:
        Кортеж из обучающего и валидационного наборов данных.
    """
    train_dataset_super = load_tfrecord_dataset('train_super_resolution.tfrecord', batch_size, input_size, output_size=output_size)
    val_dataset_super = load_tfrecord_dataset('val_super_resolution.tfrecord', batch_size, input_size, output_size=output_size)
    return train_dataset_super, val_dataset_super


def optimize_super_resolution(trial):
    """Оптимизация модели суперразрешения с использованием Optuna.
    
    Args:
        trial: Текущий эксперимент Optuna.

    Returns:
        Значение потерь на валидационном наборе данных.
    """
    train_dataset_super, val_dataset_super = get_super_resolution_datasets()
    with strategy.scope():
        super_resolution_model = RCANOptunaModel(input_shape=input_size + (3,), upscale_factor=upscale_factor)
        model = super_resolution_model.build(trial=trial)

        model.fit(
            train_dataset_super,
            validation_data=val_dataset_super,
            epochs=3,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            callbacks=[
                TensorBoard(log_dir='./logs/tuning_super_resolution', update_freq='epoch'),
                TFKerasPruningCallback(trial, 'val_loss')
            ],
            verbose=1
        )

        val_loss = model.evaluate(val_dataset_super, steps=val_steps_per_epoch, verbose=0)
        return val_loss[0]


# Запуск оптимизации модели суперразрешения
start_time = time.time()
study = optuna.create_study(direction='minimize')
study.optimize(optimize_super_resolution, n_trials=30)
end_time = time.time()
print(f"Время оптимизации гиперпараметров: {end_time - start_time} секунд")

best_params = study.best_trial.params

# Обучение лучшей модели суперразрешения
train_dataset_super, val_dataset_super = get_super_resolution_datasets()
with strategy.scope():
    super_resolution_model = RCANOptunaModel(input_shape=input_size + (3,), upscale_factor=upscale_factor)
    best_model = super_resolution_model.build(params=best_params)
    best_model.fit(
        train_dataset_super,
        validation_data=val_dataset_super,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[
            ModelCheckpoint(f"{save_path}/upscale_best.keras", save_best_only=True, save_weights_only=False, monitor="val_loss"),
            TensorBoard(log_dir='./logs/upscale_best', update_freq='epoch')
        ]
    )

# Удаление модели суперразрешения и набора данных из памяти
del best_model
gc.collect()
