import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# создаем датасет
filePath_train_label = 'data/train-labels-idx1-ubyte'
filePath_train_set = 'data/train-images-idx3-ubyte'

filePath_test_label = 'data/t10k-labels-idx1-ubyte'
filePath_test_set = 'data/t10k-images-idx3-ubyte'


with open(filePath_train_label, 'rb') as trainLbpath:
  y_train = np.frombuffer(trainLbpath.read(), np.uint8, offset=8)

with open(filePath_train_set, 'rb') as trainSetpath:
  x_train = np.frombuffer(trainSetpath.read(), np.uint8, offset=16).reshape(
            len(y_train), 28, 28
        )

with open(filePath_test_label, 'rb') as testLbpath:
  y_test = np.frombuffer(testLbpath.read(), np.uint8, offset=8)

with open(filePath_test_set, 'rb') as testSetpath:
  x_test = np.frombuffer(testSetpath.read(), np.uint8, offset=16).reshape(
            len(y_test), 28, 28
        )

# нормализация данных
x_train = x_train / 255
x_test = x_test / 255

# классы/назчания обьектов
classes = np.unique(y_train)
#print('Output classes : ', classes)
# классы товаров человеческим языком
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#print('Output classes : ', class_names)

# plt.figure(figsize=(10,10))
# for i in range(25):
#   plt.subplot(5,5,i+1)
#   plt.title("No." + str(i))
#   plt.imshow(x_train[i,:],cmap='Greys')

# создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# компиляция модели
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# вывод параметров модели
model.summary()
# обучение модели
model.fit(x_train, y_train, epochs=10)
# сохраняем модель в файле
model.save('models/shop_model')
# открываем модель из файла
#model = tf.keras.models.load_model('models/shop_model')

# проверка точности предсказания
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test:', test_acc)

check = 12
# предсказываем
predictions = model.predict(x_train)
predictions[check]

# получить результат
result = np.argmax(predictions[12])

#выведем картинку
# plt.figure()
# plt.imshow(x_train[12])
# plt.colorbar()
# plt.grid(False)

# класс обьекта
print(class_names[result])