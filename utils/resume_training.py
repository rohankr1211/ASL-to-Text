import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

DATASET_DIR = 'dataset'
MODEL_PATH = 'models/asl_cnn.h5'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
ADDITIONAL_EPOCHS = 5  # How many more epochs to train

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Please train from scratch first.")
    exit()

# Load the existing model
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# Data generators (same as original)
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

# Continue training
print(f"Resuming training for {ADDITIONAL_EPOCHS} more epochs...")
try:
    history = model.fit(train_gen, validation_data=val_gen, epochs=ADDITIONAL_EPOCHS)
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")

# Save the updated model
model.save(MODEL_PATH)
print(f"Updated model saved to {MODEL_PATH}")

if 'history' in locals():
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}") 