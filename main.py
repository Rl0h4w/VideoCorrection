from models import build_color_correction_model, build_super_resolution_model, build_noise_removal_model
from data import create_dataset, count_files
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras_tuner import RandomSearch
from tuner import NoiseRemovalHyperModel
from metrics import psnr, ssim, delta_e
import tensorflow as tf

# Setting up multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Parameters
train_dir = 'data/train_sharp'
val_dir = 'data/val_sharp'
batch_size = 32
input_size = (40, 40)
upscale_factor = 3
output_size = (input_size[0]*upscale_factor, input_size[1]*upscale_factor)
epochs = 100

# Calculate steps per epoch
train_steps_per_epoch = count_files(train_dir) // batch_size
val_steps_per_epoch = count_files(val_dir) // batch_size

# Create datasets
train_dataset_noise, val_dataset_noise = create_dataset(train_dir, val_dir, input_size, batch_size=batch_size, task='noise_removal')

# Hyperparameter tuning for noise removal model
with strategy.scope():
    hypermodel = NoiseRemovalHyperModel(input_shape=input_size + (3,))

    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='tuner_logs',
        project_name='noise_removal'
    )

    tuner.search(
        train_dataset_noise,
        validation_data=val_dataset_noise,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[
            TensorBoard(log_dir='./logs/tuning', update_freq='batch')
        ]
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train the best model
    best_model = hypermodel.build(best_hps)
    best_model.fit(
        train_dataset_noise,
        validation_data=val_dataset_noise,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[
            ModelCheckpoint("denoise_best.keras", save_best_only=True, monitor="val_loss"),
            TensorBoard(log_dir='./logs/denoise_best', update_freq='batch')
        ]
    )

# Color Correction
train_dataset_color, val_dataset_color = create_dataset(train_dir, val_dir, input_size, batch_size=batch_size, task='color_correction')
train_steps_per_epoch_color = count_files(train_dir) // batch_size
val_steps_per_epoch_color = count_files(val_dir) // batch_size

with strategy.scope():
    color_correction_model = build_color_correction_model(input_shape=input_size + (3,))
    color_correction_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse', delta_e])
    color_correction_model.fit(
        train_dataset_color,
        validation_data=val_dataset_color,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch_color,
        validation_steps=val_steps_per_epoch_color,
        callbacks=[
            ModelCheckpoint("color.keras", save_best_only=True, monitor="val_loss"),
            TensorBoard(log_dir='./logs/color', update_freq='batch')
        ]
    )

# Super Resolution
train_dataset_super, val_dataset_super = create_dataset(train_dir, val_dir, input_size, output_size=output_size, batch_size=batch_size, task='super_resolution')
train_steps_per_epoch_super = count_files(train_dir) // batch_size
val_steps_per_epoch_super = count_files(val_dir) // batch_size

with strategy.scope():
    super_resolution_model = build_super_resolution_model(input_shape=input_size + (3,), upscale_factor=upscale_factor)
    super_resolution_model.compile(optimizer='adam', loss='mse', metrics=['mae', psnr, ssim])
    super_resolution_model.fit(
        train_dataset_super,
        validation_data=val_dataset_super,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch_super,
        validation_steps=val_steps_per_epoch_super,
        callbacks=[
            ModelCheckpoint("upscale.keras", save_best_only=True, monitor="val_loss"),
            TensorBoard(log_dir='./logs/upscale', update_freq='batch')
        ]
    )
