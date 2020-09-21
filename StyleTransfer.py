import tensorflow as tf
from keras.applications import vgg19, VGG19
from keras import Model
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools




def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# 画像を読み込み、最大寸法を512ピクセルに制限する関数
def load_img(path_to_img):
    max_dim = 512

    # ファイルの読み込み 
    img = tf.io.read_file(path_to_img)
    # フォーマットに従いデコード
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    
    plt.imshow(image)
    if title:
        plt.title(title)

def vgg_layers(layer_names):
    """隠れ層の値のリストを返すVGGモデルを作る"""
    # imagenetデータで学習されたVGGモデルをロード
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = Model([vgg.input], outputs)
    return model


content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_image = load_img(content_path)
style_image = load_img(style_path)

# 画像前処理用 [0, 1] => [0, 255] 省略可
x = vgg19.preprocess_input(content_image*255)

x = tf.image.resize(x, (244, 244))

# VGG19の読み込み
# include_top : 出力層側の3つの全結合層を含むかどうか
# weights='imagenet' : imagenetで学習した重み
vgg = VGG19(include_top=False, weights='imagenet')
# 各層の出力
for layer in vgg.layers:
    print(layer.name)

# 画像のスタイルとコンテンツを表すために、ネットワークから中間レイヤーを選択
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)



