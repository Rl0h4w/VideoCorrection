import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Multiply
from tensorflow.keras.models import Model


def build_noise_removal_model(input_shape):
    """Создает и компилирует модель для удаления шума.

    Args:
        input_shape: Форма входного изображения.

    Returns:
        Компилированная модель TensorFlow.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Downscaling
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Upscaling
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Output layer
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


def build_color_correction_model(input_shape):
    """Создает и компилирует модель для коррекции цвета.

    Args:
        input_shape: Форма входного изображения.

    Returns:
        Компилированная модель TensorFlow.
    """
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, x)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


def channel_attention(input, ratio=16):
    """Создает механизм внимания для каналов.

    Args:
        input: Входной тензор.
        ratio: Коэффициент уменьшения размерности.

    Returns:
        Тензор с примененным механизмом внимания для каналов.
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


def residual_block(input, filters=64, kernel_size=(3, 3), strides=(1, 1)):
    """Создает остаточный блок с механизмом внимания для каналов.

    Args:
        input: Входной тензор.
        filters: Количество фильтров.
        kernel_size: Размер ядра свертки.
        strides: Шаг свертки.

    Returns:
        Тензор после применения остаточного блока.
    """
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = channel_attention(x)
    return Add()([input, x])


def RCAN(input_shape=(160, 90, 3), num_residual_blocks=10, upscaling_factor=3, filters=64, kernel_size=(3, 3)):
    """Создает модель RCAN для суперразрешения.

    Args:
        input_shape: Форма входного изображения.
        num_residual_blocks: Количество остаточных блоков.
        upscaling_factor: Коэффициент увеличения разрешения.
        filters: Количество фильтров.
        kernel_size: Размер ядра свертки.

    Returns:
        Модель TensorFlow.
    """
    input = Input(shape=input_shape)
    x = Conv2D(filters, kernel_size, padding='same')(input)
    x = Activation('relu')(x)
    
    # Residual in Residual (RIR)
    res = x
    for _ in range(num_residual_blocks):
        res = residual_block(res, filters=filters, kernel_size=kernel_size)
    
    x = Add()([x, res])
    
    # Upscaling
    x = Conv2D(filters * (upscaling_factor ** 2), kernel_size, padding='same')(x)
    x = tf.nn.depth_to_space(x, upscaling_factor)
    
    output = Conv2D(3, kernel_size, padding='same')(x)
    
    model = Model(inputs=input, outputs=output)
    return model


def objective(trial):
    """Целевая функция для оптимизации гиперпараметров модели RCAN с использованием Optuna.

    Args:
        trial: Эксперимент Optuna.

    Returns:
        Значение потерь на валидационном наборе данных.
    """
    input_shape = (160, 90, 3)
    num_residual_blocks = trial.suggest_int('num_residual_blocks', 5, 20)
    upscaling_factor = 3  # Фиксированное значение, так как размер выхода определен
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = (3, 3)  # Фиксированный размер ядра

    model = RCAN(input_shape=input_shape, num_residual_blocks=num_residual_blocks, 
                 upscaling_factor=upscaling_factor, filters=filters, kernel_size=kernel_size)
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    # Демонстрационные данные
    import numpy as np
    x_train = np.random.rand(100, 160, 90, 3)
    y_train = np.random.rand(100, 480, 270, 3)

    # Обучение модели
    history = model.fit(x_train, y_train, epochs=2, validation_split=0.2, verbose=0)

    # Возвращаем значение потерь на валидационном наборе данных для оптимизации
    val_loss = history.history['val_loss'][-1]
    return val_loss
