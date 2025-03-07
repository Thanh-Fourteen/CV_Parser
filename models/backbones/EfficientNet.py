# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import models

# Tham số khởi tạo kernel
parameters = {
    "kernel_initializer": "he_normal"
}

# Hàm Swish activation
def swish(x):
    return x * tf.nn.sigmoid(x)

# Lớp BatchNormalization tùy chỉnh với tùy chọn freeze
class BatchNormalization(KL.BatchNormalization):
    def __init__(self, freeze=False, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)
        self.trainable = not self.freeze

    def call(self, inputs, training=None, **kwargs):
        if self.freeze:
            return super(BatchNormalization, self).call(inputs, training=False, **kwargs)
        return super(BatchNormalization, self).call(inputs, training=training, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config

# Khối MBConv (Mobile Inverted Bottleneck Convolution)
def MBConvBlock(inputs, filters, kernel_size, strides, expansion_factor, se_ratio=0.25, drop_rate=0.0, freeze_bn=False):
    channel_axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1
    input_filters = inputs.shape[channel_axis]

    # Expansion phase (nếu expansion_factor > 1)
    x = inputs
    if expansion_factor != 1:
        x = KL.Conv2D(expansion_factor * input_filters, (1, 1), padding="same", use_bias=False, **parameters)(x)
        x = BatchNormalization(freeze=freeze_bn)(x)
        x = KL.Activation(swish)(x)

    # Depthwise Convolution
    x = KL.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False, **parameters)(x)
    x = BatchNormalization(freeze=freeze_bn)(x)
    x = KL.Activation(swish)(x)

    # Squeeze-and-Excitation (SE) block
    if se_ratio:
        se_filters = max(1, int(input_filters * se_ratio))
        se = KL.GlobalAveragePooling2D()(x)
        se = KL.Reshape((1, 1, x.shape[channel_axis]))(se)
        se = KL.Conv2D(se_filters, (1, 1), padding="same", use_bias=True, **parameters)(se)
        se = KL.Activation(swish)(se)
        se = KL.Conv2D(x.shape[channel_axis], (1, 1), padding="same", use_bias=True, **parameters)(se)
        se = KL.Activation("sigmoid")(se)
        x = KL.Multiply()([x, se])

    # Projection phase
    x = KL.Conv2D(filters, (1, 1), padding="same", use_bias=False, **parameters)(x)
    x = BatchNormalization(freeze=freeze_bn)(x)

    # Skip connection nếu input và output có cùng shape và strides=1
    if strides == 1 and input_filters == filters:
        if drop_rate > 0:
            x = KL.Dropout(drop_rate)(x)
        x = KL.Add()([x, inputs])

    return x

# Xây dựng EfficientNetB0 backbone với nhiều đầu ra dưới dạng Model
def EfficientNetB0(inputs, include_top=False, classes=1000, freeze_bn=False):
    # Cấu hình các stage của EfficientNetB0
    # [filters, kernel_size, strides, expansion_factor, num_repeats]
    config = [
        [16,  3, 1, 1, 1],  # Stage 1
        [24,  3, 2, 6, 2],  # Stage 2 (C2)
        [40,  5, 2, 6, 2],  # Stage 3 (C3)
        [80,  3, 2, 6, 3],  # Stage 4 (C4)
        [112, 5, 1, 6, 3],  # Stage 5
        [192, 5, 2, 6, 4],  # Stage 6 (C5)
        [320, 3, 1, 6, 1],  # Stage 7
    ]

    # Stem
    x = KL.Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False, **parameters)(inputs)
    x = BatchNormalization(freeze=freeze_bn)(x)
    x = KL.Activation(swish)(x)

    outputs = []

    # Xây dựng các stage và lấy các đặc trưng đa mức
    for stage_idx, (filters, kernel_size, strides, expansion_factor, repeats) in enumerate(config):
        for block_idx in range(repeats):
            block_strides = strides if block_idx == 0 else 1
            x = MBConvBlock(x, filters, kernel_size, block_strides, expansion_factor, se_ratio=0.25, freeze_bn=freeze_bn)
        
        # Lưu đầu ra của các stage quan trọng (C2, C3, C4, C5)
        if stage_idx in [1, 2, 3, 5]:  # Stage 2, 3, 4, 6 tương ứng với C2, C3, C4, C5
            outputs.append(x)

    if include_top:
        x = KL.Conv2D(1280, (1, 1), padding="same", use_bias=False, **parameters)(x)
        x = BatchNormalization(freeze=freeze_bn)(x)
        x = KL.Activation(swish)(x)
        x = KL.GlobalAveragePooling2D()(x)
        x = KL.Dense(classes, activation="softmax")(x)
        return models.Model(inputs=inputs, outputs=x, name="EfficientNetB0")
    else:
        # Trả về một Model với nhiều đầu ra
        return models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo tensor đầu vào
    input_image = KL.Input(shape=(224, 224, 3))
    # Khởi tạo backbone EfficientNetB0
    backbone = EfficientNetB0(inputs=input_image, include_top=False, freeze_bn=True)
    # Gán các đầu ra
    C2, C3, C4, C5 = backbone.outputs
    # Tạo model
    model = models.Model(inputs=input_image, outputs=[C2, C3, C4, C5], name="EfficientNetB0")
    model.summary()