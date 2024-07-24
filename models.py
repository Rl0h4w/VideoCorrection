from tensorflow.keras import layers, models

def build_noise_removal_model(input_shape):
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
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, x)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def build_super_resolution_model(input_shape, upscale_factor=3):
    inputs = layers.Input(shape=input_shape)
    
    # Initial feature extraction
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    
    # Upscaling blocks
    for _ in range(upscale_factor):
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D((2, 2))(x)
    
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
