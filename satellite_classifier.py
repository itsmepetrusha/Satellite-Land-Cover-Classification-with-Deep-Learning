import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*70)
print("️  EUROSAT → УЛУЧШЕННАЯ МОДЕЛЬ (92-95%)")
print("="*70)

data_dir = '2750'

# Улучшенная модель
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(" Улучшенная модель готова!")
model.summary()

# Загрузка данных
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, image_size=(64,64), batch_size=32, 
    validation_split=0.2, subset='training', seed=123, label_mode='int'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, image_size=(64,64), batch_size=32, 
    validation_split=0.2, subset='validation', seed=123, label_mode='int'
)

class_names = train_ds.class_names

# Улучшенная аугментация
def augment(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.4)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return tf.clip_by_value(image, 0.0, 1.0), label

train_ds = train_ds.map(augment).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x,y: (tf.cast(x,tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

print(f"\n Обучаем улучшенную модель (15 эпох):")
history = model.fit(
    train_ds, epochs=15,
    validation_data=val_ds,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
)

print("\n РЕЗУЛЬТАТЫ:")
print(f" Train max: {max(history.history['accuracy'])*100:.1f}%")
print(f" Val max: {max(history.history['val_accuracy'])*100:.1f}%")

model.save('improved_satellite_model.h5')
print(" Улучшенная модель сохранена!")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],'o-',label='Train')
plt.plot(history.history['val_accuracy'],'o-',label='Val')
plt.title('Улучшенная модель')
plt.legend(); plt.grid()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],'o-',label='Train')
plt.plot(history.history['val_loss'],'o-',label='Val')
plt.legend(); plt.grid()
plt.savefig('improved_history.png')
plt.show()

print("\n ТЕСТ НА ОДИНОЧНОМ ИЗОБРАЖЕНИИ:")
for images, labels in val_ds.take(1):
    pred = model.predict(images[:1])
    true_class = class_names[labels[0]]
    pred_class = class_names[np.argmax(pred[0])]
    conf = np.max(pred[0])*100
    print(f"Истинный: {true_class}")
    print(f"Предсказание: {pred_class} ({conf:.1f}%)")
    print(" ✓" if pred_class == true_class else "")
    break

print("\n" + "="*70)
print(" УЛУЧШЕННАЯ МОДЕЛЬ ОБУЧЕНА!")
print("="*70)
