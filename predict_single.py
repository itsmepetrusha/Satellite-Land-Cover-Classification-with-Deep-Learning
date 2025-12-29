import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import os

print("="*60)
print(" ТЕСТ НА ОДИНОЧНОМ ИЗОБРАЖЕНИИ")
print("="*60)

# Загружаем вашу модель
model_path = 'improved_satellite_model.h5'  # Или 'offline_satellite_model.h5'
model = tf.keras.models.load_model(model_path)
print(f" Модель загружена: {model_path}")

class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

print(f"️  Классы: {class_names}")
print("\n НАЙДЕННЫЕ ИЗОБРАЖЕНИЯ В ПАПКЕ:")
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
found_images = []
for ext in image_extensions:
    found_images.extend([f for f in os.listdir('.') if f.endswith(ext)])
    found_images.extend([f for f in os.listdir('test_images/') if f.endswith(ext)] if os.path.exists('test_images/') else [])
    
print(f"   Найдено: {len(found_images)} изображений")
for img in found_images[:3]:
    print(f"    {img}")

def predict_image(image_path):
    """Предсказание на одном изображении"""
    print(f"\n Анализирую: {os.path.basename(image_path)}")
    
    # Загружаем изображение
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, 0) / 255.0
    
    # Предсказание модели
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # Топ-5 предсказаний
    top5_idx = np.argsort(predictions)[-5:][::-1]
    top5 = [(class_names[i], predictions[i]*100) for i in top5_idx]
    
    return predicted_class, confidence, top5, img_array[0], img

# 1. ТЕСТ НАВЫТРЕСКИ НАЙДЕННЫХ ИЗОБРАЖЕНИЙ
print("\n 1. АВТОТЕСТ НАЙДЕННЫХ ИЗОБРАЖЕНИЙ:")
results = []
for img_file in found_images[:3]:
    try:
        pred_class, conf, top5, processed_img, original_img = predict_image(img_file)
        results.append((img_file, pred_class, conf, top5, processed_img, original_img))
        
        print(f"    {pred_class}")
        print(f"    Уверенность: {conf*100:.1f}%")
        print(f"    Топ-3: {top5[0][0]}({top5[0][1]:.1f}%), {top5[1][0]}({top5[1][1]:.1f}%)")
    except Exception as e:
        print(f"    Ошибка: {e}")

# 2. РУЧНОЙ ВВОД ПУТИ
print("\n" + "="*60)
image_path = input(" Введите путь к изображению (или Enter для выхода): ").strip().strip('"\'')
if image_path:
    if os.path.exists(image_path):
        pred_class, conf, top5, processed_img, original_img = predict_image(image_path)
        results.append((image_path, pred_class, conf, top5, processed_img, original_img))
    else:
        print(" Файл не найден!")

# 3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
if results:
    plt.figure(figsize=(15, 5*len(results)))
    for i, (path, pred, conf, top5, proc_img, orig_img) in enumerate(results):
        # Оригинал
        plt.subplot(len(results), 2, i*2+1)
        plt.imshow(orig_img)
        plt.title(f'ОРИГИНАЛ\n{os.path.basename(path)}', fontsize=12)
        plt.axis('off')
        
        # Обработанное + предсказание
        plt.subplot(len(results), 2, i*2+2)
        plt.imshow(proc_img)
        plt.title(f'ПРЕДСКАЗАНИЕ\n{pred}\n{conf*100:.1f}%\n{top5[1][0]}: {top5[1][1]:.1f}%', 
                 fontsize=12, color='green' if conf > 0.8 else 'orange')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n РЕЗУЛЬТАТЫ СОХРАНЕНЫ: predictions_result.png")
else:
    print("\n️  Изображения для теста не найдены.")

print("\n" + "="*60)
print(" ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
print(" СОВЕТЫ:")
print("   • Сохраняйте снимки Google Maps размером 500x500+ пикселей")
print("   • Лучше качество = лучше предсказание")
print("="*60)
