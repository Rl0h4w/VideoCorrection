import tensorflow as tf

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def rgb_to_lab(srgb):
    srgb = tf.where(srgb > 0.04045, tf.math.pow((srgb + 0.055) / 1.055, 2.4), srgb / 12.92)
    xyz = tf.tensordot(srgb, [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]], axes=1)
    xyz = xyz / [0.95047, 1.000, 1.08883]
    xyz = tf.where(xyz > 0.008856, tf.math.pow(xyz, 1/3), (7.787 * xyz) + (16 / 116))
    lab = tf.stack([(116 * xyz[..., 1]) - 16, 500 * (xyz[..., 0] - xyz[..., 1]), 200 * (xyz[..., 1] - xyz[..., 2])], axis=-1)
    return lab

def delta_e(y_true, y_pred):
    y_true_lab = rgb_to_lab(y_true)
    y_pred_lab = rgb_to_lab(y_pred)
    delta = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true_lab - y_pred_lab), axis=-1)))
    return delta