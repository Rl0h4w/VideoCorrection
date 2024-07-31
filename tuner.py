import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Multiply
from tensorflow.keras.models import Model
import optuna
import numpy as np
from metrics import psnr, ssim, delta_e


class NoiseRemovalOptunaModel:
    """Модель для удаления шума с использованием Optuna для подбора гиперпараметров."""

    def __init__(self, input_shape):
        """
        Инициализация модели.

        Args:
            input_shape: Форма входного изображения.
        """
        self.input_shape = input_shape

    def build(self, trial=None, params=None):
        """
        Создает и компилирует модель для удаления шума с учетом предложенных параметров Optuna или вручную заданных параметров.

        Args:
            trial: Опциональный параметр Optuna trial для предложений гиперпараметров.
            params: Опциональный словарь параметров для настройки модели.

        Returns:
            Компилированная модель Keras.
        """
        inputs = layers.Input(shape=self.input_shape)

        if params:
            filters1 = params['filters1']
            filters2 = params['filters2']
            filters3 = params['filters3']
            filters4 = params['filters4']
            filters5 = params['filters5']
            filters6 = params['filters6']
        else:
            filters1 = trial.suggest_int('filters1', 128, 512, step=128)
            filters2 = trial.suggest_int('filters2', 128, 512, step=128)
            filters3 = trial.suggest_int('filters3', 256, 1024, step=256)
            filters4 = trial.suggest_int('filters4', 256, 1024, step=256)
            filters5 = trial.suggest_int('filters5', 128, 512, step=128)
            filters6 = trial.suggest_int('filters6', 128, 512, step=128)

        x = layers.Conv2D(filters=filters1, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(filters=filters2, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(filters=filters3, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=filters4, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(filters=filters5, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=filters6, kernel_size=(3, 3), activation='relu', padding='same')(x)
        outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss='mse', metrics=['mae', psnr, ssim])
        return model


class ColorCorrectionOptunaModel:
    """Модель для коррекции цвета с использованием Optuna для подбора гиперпараметров."""

    def __init__(self, input_shape):
        """
        Инициализация модели.

        Args:
            input_shape: Форма входного изображения.
        """
        self.input_shape = input_shape

    def build(self, trial=None, params=None):
        """
        Создает и компилирует модель для коррекции цвета с учетом предложенных параметров Optuna или вручную заданных параметров.

        Args:
            trial: Опциональный параметр Optuna trial для предложений гиперпараметров.
            params: Опциональный словарь параметров для настройки модели.

        Returns:
            Компилированная модель Keras.
        """
        inputs = layers.Input(shape=self.input_shape)

        if params:
            filters1 = params['filters1']
            filters2 = params['filters2']
            filters3 = params['filters3']
        else:
            filters1 = trial.suggest_int('filters1', 128, 512, step=128)
            filters2 = trial.suggest_int('filters2', 128, 512, step=128)
            filters3 = trial.suggest_int('filters3', 128, 512, step=128)

        x = layers.Conv2D(filters=filters1, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(filters=filters2, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=filters3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
        outputs = layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss='mse', metrics=['mae', 'mse', delta_e])
        return model


class RCANOptunaModel:
    """Модель RCAN для суперразрешения с использованием Optuna для подбора гиперпараметров."""

    def __init__(self, input_shape=(160, 90, 3), upscale_factor=3):
        """
        Инициализация модели.

        Args:
            input_shape: Форма входного изображения.
            upscale_factor: Коэффициент увеличения разрешения.
        """
        self.input_shape = input_shape
        self.upscale_factor = upscale_factor
        self.study = None

    def channel_attention(self, input, ratio=16):
        """
        Реализует механизм внимания к каналам.

        Args:
            input: Входной тензор.
            ratio: Коэффициент уменьшения каналов.

        Returns:
            Тензор после применения внимания к каналам.
        """
        channel = input.shape[-1]
        shared_layer_one = Conv2D(channel // ratio, (1, 1), activation='relu', padding='same')
        shared_layer_two = Conv2D(channel, (1, 1), padding='same')
        avg_pool = tf.reduce_mean(input, axis=[1, 2], keepdims=True)
        avg_out = shared_layer_two(shared_layer_one(avg_pool))
        max_pool = tf.reduce_max(input, axis=[1, 2], keepdims=True)
        max_out = shared_layer_two(shared_layer_one(max_pool))
        out = avg_out + max_out
        return Multiply()([input, Activation('sigmoid')(out)])

    def residual_block(self, input, filters=64, kernel_size=(3, 3), strides=(1, 1)):
        """
        Реализует остаточный блок с механизмом внимания к каналам.

        Args:
            input: Входной тензор.
            filters: Количество фильтров в сверточных слоях.
            kernel_size: Размер ядра свертки.
            strides: Шаг свертки.

        Returns:
            Тензор после применения остаточного блока.
        """
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = self.channel_attention(x)
        return Add()([input, x])

    def build(self, trial=None, params=None):
        """
        Создает и компилирует модель RCAN с учетом предложенных параметров Optuna или вручную заданных параметров.

        Args:
            trial: Опциональный параметр Optuna trial для предложений гиперпараметров.
            params: Опциональный словарь параметров для настройки модели.

        Returns:
            Компилированная модель Keras.
        """
        input = Input(shape=self.input_shape)
        
        if params:
            num_residual_blocks = params['num_residual_blocks']
            filters = params['filters']
        else:
            num_residual_blocks = trial.suggest_int('num_residual_blocks', 5, 20)
            filters = trial.suggest_int('filters', 32, 128)
        
        x = Conv2D(filters, (3, 3), padding='same')(input)
        x = Activation('relu')(x)
        res = x
        for _ in range(num_residual_blocks):
            res = self.residual_block(res, filters=filters)
        x = Add()([x, res])
        x = Conv2D(filters * (self.upscale_factor ** 2), (3, 3), padding='same')(x)
        x = tf.nn.depth_to_space(x, self.upscale_factor)
        output = Conv2D(3, (3, 3), padding='same')(x)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer="adam", loss='mse', metrics=['mae', psnr, ssim])
        return model

    def objective(self, trial):
        """
        Определяет целевую функцию для Optuna, строит и обучает модель, возвращая значение функции потерь на валидации.

        Args:
            trial: Параметр Optuna trial для предложений гиперпараметров.

        Returns:
            Значение функции потерь на валидации.
        """
        model = self.build(trial)
        x_train = np.random.rand(100, *self.input_shape)
        y_train = np.random.rand(100, self.input_shape[0] * self.upscale_factor, self.input_shape[1] * self.upscale_factor, 3)
        history = model.fit(x_train, y_train, epochs=2, validation_split=0.2, verbose=0)
        val_loss = history.history['val_loss'][-1]
        return val_loss

    def optimize(self, n_trials=10):
        """
        Запускает процесс оптимизации гиперпараметров с использованием Optuna.

        Args:
            n_trials: Количество испытаний для Optuna.
        """
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=n_trials)

    def get_best_params(self):
        """
        Возвращает лучшие параметры, найденные Optuna.

        Returns:
            Словарь с лучшими параметрами.

        Raises:
            ValueError: Если оптимизация еще не была выполнена.
        """
        if self.study is None:
            raise ValueError("Optimization has not been run yet.")
        return self.study.best_trial.params
