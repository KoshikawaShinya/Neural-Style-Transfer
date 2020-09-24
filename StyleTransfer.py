import tensorflow as tf
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
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# スタイルの計算
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# ピクセル値を0から1の間に保つ
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# 損失の計算
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    # スタイルの損失
    # コンテンツ画像とスタイル画像のスタイルの部分でMSEを行う
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    
    # コンテンツの損失
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor.call(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)
    
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))



# スタイルとコンテンツを抽出する
class StyleContentModel(tf.keras.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """inputは[0, 1]のfloat入力"""
        # [0, 1] => [0, 255]
        inputs = inputs * 255.0
        # 画像の前処理(画像の正規化やimagenetデータセットのRGB各チャンネルごとの平均値を引く等々)
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        # スタイルとコンテンツに指定した各層の出力を取り出す
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        # style_outputsをグラムマトリックスにする
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name : value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name : value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style' : style_dict}





content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_image = load_img(content_path)
style_image = load_img(style_path)

# 画像前処理用
#x = vgg19.preprocess_input(content_image*255)

#x = tf.image.resize(x, (244, 244))

# VGG19の読み込み
# include_top : 出力層側の3つの全結合層を含むかどうか
# weights='imagenet' : imagenetで学習した重み


# 画像のスタイルとコンテンツを表すために、ネットワークから中間レイヤーを選択
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# モデルの作成
extractor = StyleContentModel(style_layers, content_layers)

results = extractor.call(tf.constant(content_image))

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())

# スタイル画像のスタイルとコンテンツ画像のコンテンツをターゲットとして設定
style_targets = extractor.call(style_image)['style']
content_targets = extractor.call(content_image)['content']

# 最適化する画像を含むtf.Variableを定義。コンテンツ画像で初期化(tf.Variableはコンテンツ画像と同じ形状でなければならない)
image = tf.Variable(content_image)

# オプティマイザ
optimizer = tf.optimizers.Adam()

# スタイルとコンテンツの重み
style_weight = 1e-2
content_weight = 1e4

# 総変動損失の重み
total_variation_weight=30

for i in range(10):
    train_step(image)

plt.imshow(tensor_to_image(image))
plt.show()
