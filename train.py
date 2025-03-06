import os
import os.path as osp
import tensorflow as tf

print(tf.config.list_physical_devices())

from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.metrics import Precision, Recall
from generate import generate
from models.model import DBNet
from config import DBConfig

cfg = DBConfig()

strategy = tf.distribute.MirroredStrategy()
print(f"Số lượng GPU đang sử dụng: {strategy.num_replicas_in_sync}")

train_generator = generate(cfg, 'train')
val_generator = generate(cfg, 'val')

with strategy.scope():
    model = DBNet(cfg, model='training', backbone= cfg.BACKBONE)
    if cfg.PRETRAINED_MODEL_PATH:
        model.load_weights(cfg.PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        metrics={
            'binarize_map': [Precision(name='precision'), Recall(name='recall')],
            'thresh_binary': [Precision(name='precision_binary'), Recall(name='recall_binary')]
        }
    )

model.summary()

class MetricsCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        precision = logs.get("binarize_map_precision", 0)  # Lấy từ binarize_map
        recall = logs.get("binarize_map_recall", 0)
        f1_score = (2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())) if (precision + recall) > 0 else 0
        print(f"Epoch {epoch + 1} - Loss: {logs.get('loss'):.4f}, Val Loss: {logs.get('val_loss'):.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
        logs['f1_score'] = f1_score

checkpoint_callback = callbacks.ModelCheckpoint(
    osp.join(cfg.CHECKPOINT_DIR, 'db_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5')
)
tensorboard_callback = callbacks.TensorBoard(
    log_dir=cfg.LOG_DIR,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

callbacks_list = [checkpoint_callback, tensorboard_callback, MetricsCallback()]

model.fit(
    x=train_generator,
    steps_per_epoch=len(train_generator) // cfg.BATCH_SIZE,
    initial_epoch=cfg.INITIAL_EPOCH,
    epochs=cfg.EPOCHS,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=val_generator,
    validation_steps=len(val_generator) // cfg.BATCH_SIZE
)