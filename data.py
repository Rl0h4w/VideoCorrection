import os
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf

def random_distort(image):
    image = Image.fromarray(np.uint8(image * 255))
    # Random brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    # Random contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    # Random saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    # Random sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    return np.array(image) / 255.0

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = np.random.uniform(0.001, 0.05)  # Random variance
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy

def load_image(img_path, size):
    try:
        img = Image.open(img_path).resize(size)
        img = np.array(img) / 255.0
        return img
    except (IOError, ValueError) as e:
        print(f"Could not process file {img_path}: {e}")
        return None

def image_generator(folder, size, task):
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue
                img_path = os.path.join(subdir_path, filename)
                img = load_image(img_path, size)
                if img is not None:
                    if task == 'noise_removal':
                        distorted_img = random_distort(img)
                        noisy_img = add_noise(distorted_img)
                        yield noisy_img, distorted_img
                    elif task == 'color_correction':
                        distorted_img = random_distort(img)
                        yield distorted_img, img
                    elif task == 'super_resolution':
                        high_res_size = (size[0] * 3, size[1] * 3)  # For example, 40x40 to 120x120
                        high_res_img = load_image(img_path, high_res_size)
                        if high_res_img is not None:
                            yield img, high_res_img

def create_dataset(train_dir, val_dir, input_size, output_size=None, batch_size=32, task='noise_removal'):
    train_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(train_dir, input_size, task),
        output_signature=(tf.TensorSpec(shape=(input_size[0], input_size[1], 3), dtype=tf.float32),
                          tf.TensorSpec(shape=(input_size[0], input_size[1], 3), dtype=tf.float32))
    ).batch(batch_size)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(val_dir, input_size, task),
        output_signature=(tf.TensorSpec(shape=(input_size[0], input_size[1], 3), dtype=tf.float32),
                          tf.TensorSpec(shape=(input_size[0], input_size[1], 3), dtype=tf.float32))
    ).batch(batch_size)
    
    return train_dataset, val_dataset

def count_files(directory):
    count = 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    count += 1
    return count
