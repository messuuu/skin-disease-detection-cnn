"""
train_model.py

Skin Disease Classification using a Convolutional Neural Network (CNN).

Classifies dermoscopic skin lesion images into 9 disease categories using
the ISIC (International Skin Imaging Collaboration) dataset, built and
trained in Google Colab with TensorFlow/Keras.

Pipeline:
    1. Load images from Google Drive (ISIC dataset)
    2. Build train/validation datasets from directory structure
    3. Train a custom CNN (3 conv blocks + dense classifier)
    4. Plot training/validation accuracy and loss curves
    5. Run inference on a sample test image
    6. Save the trained model

Usage (in Google Colab):
    Run cells top to bottom after mounting Google Drive and placing
    SkinCancerDataset.zip in your Drive root.
"""

# ==========================================================
# 1. MOUNT GOOGLE DRIVE
# ==========================================================
from google.colab import drive
drive.mount('/content/gdrive')

# ==========================================================
# 2. UNZIP DATASET
# ==========================================================
!unzip "/content/gdrive/MyDrive/SkinCancerDataset.zip" -d "/content/" > /dev/null

# ==========================================================
# 3. IMPORT LIBRARIES
# ==========================================================
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

# ==========================================================
# 4. SET TRAIN & TEST PATHS
# ==========================================================
data_dir_train = pathlib.Path("/content/SkinCancer ISIC The International Skin Imaging Collaboration/Train/")
data_dir_test  = pathlib.Path("/content/SkinCancer ISIC The International Skin Imaging Collaboration/Test/")

# ==========================================================
# 5. COUNT NUMBER OF IMAGES
# ==========================================================
image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
print("Train Images:", image_count_train)
print("Test Images :", image_count_test)

# ==========================================================
# 6. GET CLASS NAMES
# ==========================================================
class_names = sorted([item.name for item in data_dir_train.glob('*') if item.is_dir()])
print("Classes:", class_names)

# ==========================================================
# 7. SHOW SAMPLE IMAGES
# ==========================================================
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img_path = list(data_dir_train.glob(f"{class_names[0]}/*"))[i]
    img = load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
plt.show()

# ==========================================================
# 8. CREATE TRAIN & VALIDATION DATASET
# ==========================================================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

# ==========================================================
# 9. PERFORMANCE OPTIMIZATION
# ==========================================================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ==========================================================
# 10. BUILD THE CNN MODEL
# ==========================================================
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))  # output layer
])

model.summary()

# ==========================================================
# 11. COMPILE THE MODEL
# ==========================================================
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ==========================================================
# 12. TRAIN THE MODEL
# ==========================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# ==========================================================
# 13. PLOT ACCURACY & LOSS
# ==========================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title("Loss")

plt.show()

# ==========================================================
# 14. TEST ON ONE SAMPLE IMAGE
# ==========================================================
test_image_path = list(data_dir_test.glob('*/*.jpg'))[0]
img = load_img(test_image_path, target_size=(180, 180))
plt.imshow(img)
plt.axis('off')

# convert to array
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("Predicted Class:", class_names[np.argmax(score)])
print("Confidence :", np.max(score))

# ==========================================================
# 15. SAVE THE MODEL
# ==========================================================
model.save("melanoma_skin_cancer_cnn_model.h5")
print("Model Saved Successfully!")
