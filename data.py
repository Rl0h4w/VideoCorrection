import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor


def image_example(image_string, label):
    """Создание TFExample из изображения и метки.
    
    Args:
        image_string: Сырые байты изображения.
        label: Метка изображения в формате TensorFlow.

    Returns:
        Экземпляр tf.train.Example.
    """
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _bytes_feature(tf.io.serialize_tensor(label).numpy()),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _bytes_feature(value):
    """Возвращает bytes_list из значения.
    
    Args:
        value: Значение для преобразования в bytes_list.

    Returns:
        Экземпляр tf.train.Feature, содержащий bytes_list.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Возвращает int64_list из значения.
    
    Args:
        value: Значение для преобразования в int64_list.

    Returns:
        Экземпляр tf.train.Feature, содержащий int64_list.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def process_image(img_path, size, task):
    """Обрабатывает изображение в зависимости от задачи.
    
    Args:
        img_path: Путь к изображению.
        size: Размер изображения для изменения.
        task: Задача для обработки ('noise_removal', 'color_correction', 'super_resolution').

    Returns:
        Сериализованный строковый пример TF или None, если изображение не удалось обработать.
    """
    img = load_image(img_path, size)
    if img is not None:
        if task == 'noise_removal':
            distorted_img = random_distort(img)
            noisy_img = add_noise(distorted_img)
            label = tf.convert_to_tensor(distorted_img, dtype=tf.float32)
        elif task == 'color_correction':
            distorted_img = random_distort(img)
            label = tf.convert_to_tensor(img, dtype=tf.float32)
        elif task == 'super_resolution':
            high_res_size = (480, 270)
            high_res_img = load_image(img_path, high_res_size)
            label = tf.convert_to_tensor(high_res_img, dtype=tf.float32)
        tf_example = image_example(
            tf.io.encode_jpeg(tf.convert_to_tensor(img * 255, dtype=tf.uint8)).numpy(), label)
        return tf_example.SerializeToString()
    return None


def write_tfrecords(folder, output_path, size, task):
    """Записывает TFRecords из изображений в папке.
    
    Args:
        folder: Путь к папке с изображениями.
        output_path: Путь для сохранения TFRecord файла.
        size: Размер изображения для изменения.
        task: Задача для обработки ('noise_removal', 'color_correction', 'super_resolution').
    """
    with tf.io.TFRecordWriter(output_path) as writer:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, img_path, size, task)
                       for img_path in image_file_generator(folder)]
            for future in futures:
                result = future.result()
                if result is not None:
                    writer.write(result)


def load_image(img_path, size):
    """Загружает и изменяет размер изображения.
    
    Args:
        img_path: Путь к изображению.
        size: Размер изображения для изменения.

    Returns:
        Измененное изображение как массив numpy или None, если не удалось загрузить.
    """
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (size[1], size[0]))
        img = img.astype(np.float32) / 255.0
        return img
    except Exception as e:
        print(f"Не удалось обработать файл {img_path}: {e}")
        return None


def image_file_generator(folder):
    """Генерирует пути к изображениям в папке.
    
    Args:
        folder: Путь к папке с изображениями.

    Yields:
        Полный путь к каждому изображению в папке.
    """
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    yield os.path.join(subdir_path, filename)


def random_distort(image):
    """Применяет случайные искажения к изображению.
    
    Args:
        image: Исходное изображение в формате numpy array.

    Returns:
        Искаженное изображение в формате numpy array.
    """
    image = Image.fromarray(np.uint8(image * 255))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    return np.array(image) / 255.0


def add_noise(image):
    """Добавляет шум к изображению.
    
    Args:
        image: Исходное изображение в формате numpy array.

    Returns:
        Изображение с добавленным шумом в формате numpy array.
    """
    row, col, ch = image.shape
    mean = 0
    var = np.random.uniform(0.001, 0.05)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy


def load_tfrecord_dataset(file_pattern, batch_size, image_size, output_size=None):
    """Загружает датасет из TFRecord файлов.
    
    Args:
        file_pattern: Путь к файлам TFRecord.
        batch_size: Размер батча.
        image_size: Размер изображения для изменения.
        output_size: Размер выходного изображения (если есть).

    Returns:
        Датасет TensorFlow.
    """
    def _parse_image_function(proto):
        keys_to_features = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.string),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        image = tf.image.decode_jpeg(parsed_features['image_raw'])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.io.parse_tensor(parsed_features['label'], out_type=tf.float32)
        label = tf.reshape(label, output_size + (3,))  # Ensure label shape
        return image, label

    raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
    dataset = raw_dataset.map(_parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().batch(batch_size, drop_remainder=True).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

    def ensure_shape_with_fallback(x, y, fallback_shape, output_shape=None):
        """Устанавливает форму тензоров с запасной формой на случай несовпадения.
        
        Args:
            x: Тензор изображения.
            y: Тензор метки.
            fallback_shape: Запасная форма изображения.
            output_shape: Форма выходного изображения (если есть).

        Returns:
            Тензоры x и y с установленной формой.
        """
        desired_shape = (batch_size, *fallback_shape, 3)
        output_shape = output_shape if output_shape else fallback_shape
        desired_output_shape = (batch_size, *output_shape, 3)
        x = tf.ensure_shape(x, desired_shape)
        y = tf.ensure_shape(y, desired_output_shape)
        return x, y

    dataset = dataset.map(
        lambda x, y: ensure_shape_with_fallback(x, y, image_size, output_size), 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset
