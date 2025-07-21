import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = 'dataset'
MODEL_PATH = 'models/asl_cnn.h5'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
try:
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
except KeyboardInterrupt:
    print("Training interrupted. Saving model so far...")

# Save model (always runs)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
# Always save as 'asl_cnn.h5' (no double extension)
if not MODEL_PATH.endswith('.h5'):
    MODEL_PATH += '.h5'
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
if 'history' in locals():
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}") 