from tensorflow.keras import layers, models
from keras_tuner import HyperModel
from metrics import psnr, ssim

class NoiseRemovalHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = layers.Input(shape=self.input_shape)

        # Downscaling
        x = layers.Conv2D(
            filters=hp.Int('filters1', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu',
            padding='same')(inputs)
        x = layers.Conv2D(
            filters=hp.Int('filters2', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu',
            padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Bottleneck
        x = layers.Conv2D(
            filters=hp.Int('filters3', min_value=64, max_value=256, step=64),
            kernel_size=(3, 3),
            activation='relu',
            padding='same')(x)
        x = layers.Conv2D(
            filters=hp.Int('filters4', min_value=64, max_value=256, step=64),
            kernel_size=(3, 3),
            activation='relu',
            padding='same')(x)

        # Upscaling
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(
            filters=hp.Int('filters5', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu',
            padding='same')(x)
        x = layers.Conv2D(
            filters=hp.Int('filters6', min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            activation='relu',
            padding='same')(x)

        # Output layer
        outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', psnr, ssim]
        )
        return model
