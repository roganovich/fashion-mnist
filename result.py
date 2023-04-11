import numpy as np
import tensorflow as tf

# список тестовых картинок
images = [
    'examples/Screenshot_1.png',
    'examples/Screenshot_2.png',
    'examples/Screenshot_3.png',
    'examples/Screenshot_4.png',
    'examples/Screenshot_5.png',
    'examples/Screenshot_6.png',
    'examples/Screenshot_7.png',
    'examples/shirts_1.jpg',
    'examples/shirts_2.jpg',
    'examples/trouser_1.jpg',
    'examples/shirts_2.jpg'
];
# названия классов обьтектов
class_names = ['Футболка', 'Брюки', 'Пуловер', 'Платье', 'Пальто', 'Сандалий', 'Рубашка', 'Кроссовок', 'Сумка', 'Ботинок']
# размеры изображений в нашей модели
img_height = 28
img_width = 28
# подключаем обученную модель
model = tf.keras.models.load_model('models/shop_model')

for num, path in enumerate(images):
    # формируем изображение в rgb и размером обученной модели
    img = tf.keras.utils.load_img(
        path, color_mode='rgb', target_size=(img_height, img_width)
    )
    # переводим в изображение с 1 слоем
    img = img.convert('L')
    img_array = tf.keras.utils.img_to_array(np.array(img))
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    # расчет вероятностей
    predictions = model.predict(img_array)
    # получить результат
    score = tf.nn.softmax(predictions[0])
    result = np.argmax(predictions[0])
    # выводим ответ
    print(
        "{} - это {} на {:.2f} %."
        .format(path, class_names[result], 100 * np.max(score))
    )
