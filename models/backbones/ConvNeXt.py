import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import backend as K

# BatchNormalization và ConvNeXtBlock giữ nguyên như trước
class BatchNormalization(KL.BatchNormalization):
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        if self.freeze:
            kwargs['training'] = False
        return super(BatchNormalization, self).call(*args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config

class ConvNeXtBlock(tf.keras.layers.Layer):
    def __init__(self, channels, drop_path_rate=0.0, layer_scale_init=1e-6, freeze_bn=False, name=None):
        super(ConvNeXtBlock, self).__init__(name=name)
        self.dwconv = KL.DepthwiseConv2D(
            kernel_size=7,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            name="dwconv"
        )
        self.norm = BatchNormalization(freeze=freeze_bn, epsilon=1e-6, name="norm")
        self.pwconv1 = KL.Conv2D(
            filters=4 * channels,
            kernel_size=1,
            use_bias=False,
            kernel_initializer="he_normal",
            name="pwconv1"
        )
        self.act = KL.Activation("gelu", name="act")
        self.pwconv2 = KL.Conv2D(
            filters=channels,
            kernel_size=1,
            use_bias=False,
            kernel_initializer="he_normal",
            name="pwconv2"
        )
        self.layer_scale_init = layer_scale_init
        self.gamma = tf.Variable(
            layer_scale_init * tf.ones((channels,)),
            trainable=True if layer_scale_init > 0 else False,
            name="gamma"
        )
        self.drop_path = KL.Dropout(rate=drop_path_rate, name="drop_path") if drop_path_rate > 0. else KL.Activation("linear", name="identity")

    def call(self, inputs, training=None):
        x = inputs
        residual = x
        
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        
        x = self.drop_path(x, training=training)
        return residual + x

    def get_config(self):
        config = super(ConvNeXtBlock, self).get_config()
        config.update({
            "channels": self.pwconv2.filters,
            "drop_path_rate": self.drop_path.rate if hasattr(self.drop_path, "rate") else 0.0,
            "layer_scale_init": self.layer_scale_init,
            "freeze_bn": self.norm.freeze
        })
        return config

class ConvNeXt(tf.keras.Model):
    def __init__(self, inputs, depths, channels, include_top=False, classes=1000, freeze_bn=True, drop_path_rate=0.0, name="ConvNeXt"):
        super(ConvNeXt, self).__init__(name=name)
        
        if K.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        self.stem = tf.keras.Sequential([
            KL.Conv2D(
                channels[0], 
                kernel_size=4, 
                strides=4,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                name="stem_conv"
            ),
            BatchNormalization(freeze=freeze_bn, epsilon=1e-6, name="stem_bn")
        ], name="stem")

        total_blocks = sum(depths)
        dpr = tf.linspace(0.0, float(drop_path_rate), total_blocks).numpy().tolist()
        curr_dpr = 0

        self.stages = []
        self.feature_outputs = []
        
        for stage_idx, (depth, channel) in enumerate(zip(depths, channels)):
            stage_blocks = []
            for block_idx in range(depth):
                stage_blocks.append(
                    ConvNeXtBlock(
                        channels=channel,
                        drop_path_rate=dpr[curr_dpr + block_idx],
                        freeze_bn=freeze_bn,
                        name=f"block_{stage_idx}_{block_idx}"
                    )
                )
            curr_dpr += depth
            
            self.stages.append(tf.keras.Sequential(stage_blocks, name=f"stage_{stage_idx}"))
            
            if stage_idx < len(depths) - 1:
                downsample = tf.keras.Sequential([
                    BatchNormalization(freeze=freeze_bn, epsilon=1e-6, name=f"downsample_bn_{stage_idx}"),
                    KL.Conv2D(
                        channels[stage_idx + 1],
                        kernel_size=2,
                        strides=2,
                        use_bias=False,
                        kernel_initializer="he_normal",
                        name=f"downsample_conv_{stage_idx}"
                    )
                ], name=f"downsample_{stage_idx}")
                self.stages.append(downsample)

        # Xử lý đầu ra
        x = self.stem(inputs)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i % 2 == 0:  # Lưu đặc trưng từ mỗi giai đoạn
                self.feature_outputs.append(x)

        if include_top:
            assert classes > 0, "When include_top=True, classes must be specified and greater than 0."
            x = KL.GlobalAveragePooling2D(name="global_pool")(self.feature_outputs[-1])  # Dùng đặc trưng cuối cùng
            x = KL.Dense(classes, activation="softmax", name="classifier")(x)
            super(ConvNeXt, self).__init__(inputs=inputs, outputs=x, name=name)
        else:
            super(ConvNeXt, self).__init__(inputs=inputs, outputs=self.feature_outputs, name=name)

class ConvNeXtB(ConvNeXt):
    def __init__(self, inputs, include_top=False, classes=1000, freeze_bn=True, drop_path_rate=0.1):
        depths = [3, 3, 9, 3]
        channels = [128, 256, 512, 1024]
        
        super(ConvNeXtB, self).__init__(
            inputs=inputs,
            depths=depths,
            channels=channels,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            drop_path_rate=drop_path_rate,
            name="ConvNeXtB"
        )