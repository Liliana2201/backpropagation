import random
import numpy as np

# Функция активации - гиперболический тангенс
def f(x):
    return 2 / (1 + np.exp(-x)) - 1

# Производная от функции активации
def df(x):
    return 0.5*(1 + f(x))*(1 - f(x))

# Задаем матрицы весов слоев random.uniform(-0.5, 0.5)
W1 = [[0] * 4 for i in range(3)]
W2 = [[0] * 4 for k in range(4)]
W3 = [0] * 4
for i in range(4):
    for j in range(4):
        W2[i][j] = random.uniform(-0.5, 0.5)
        if j < 3:
            W1[j][i] = random.uniform(-0.5, 0.5)
        W3[j] = random.uniform(-0.5, 0.5)
W1 = np.array(W1)
W2 = np.array(W2)
W3 = np.array(W3)
print("Матрицы весов:\n")
print("W1:\t", W1, "\n")
print("W2:\t", W2, "\n")
print("W3:\t", W3, "\n")

# обучающие примеры (3 входа - 1 выход)
epoch = [[0] * 4 for b in range(8)]
for i in range(8):
    for j in range(4):
        epoch[i][j] = random.choice([-1, 1])
print("\nОбучающие примеры\n")
print(epoch)

def go_forward(inp):
    # умножаем вектор входных данных на матрицу весов первого скрытого слоя (получаем входы на нейронах первого слоя)
    sum = np.dot(inp, W1)
    # применяем функцию к входам - получаем выходы на нейронах 1 слоя
    out1 = np.array([f(x) for x in sum])
    # умножаем выходы 1 слоя на матрицу весов 2 слоя - получаем входы на нейроны второго слоя
    sum = np.dot(out1, W2)
    # применяем функцию к входам - получаем выходы на нейронах 2 слоя
    out2 = np.array([f(x) for x in sum])
    # умножаем выходы 2 слоя на матрицу весов - получаем вход на реагирующий слой
    sum = np.dot(out2, W3.T)
    # применяем функцию к входу реагирующего слоя - получаем выход (результат)
    y = f(sum)
    return (y, out1, out2)


def train(epoch):
    global W1, W2, W3
    # шаг коррекции весов
    lmd = 0.001
    # эпохи обучения
    N = 100000
    # Количество обучающих примеров
    count = len(epoch)
    # Обучение
    for k in range(N):
        # берем рандомный обучающий пример
        x = epoch[np.random.randint(0, count)]
        # делаем прямой проход
        y, out1, out2 = go_forward(x[0:3])
        # вычисляем отклонение
        e = y - x[-1]
        # вычисляем дельту для последнего слоя
        delta3 = e * df(y)
        # обратный проход
        delta2 = [0 for x in range(4)]
        delta1 = [0 for x in range(4)]
        for i in range(4):
            # меняем веса между вторым скрытым и реагирующим слоями
            W3[i] -= lmd * delta3 * out2[i]
            # вычисляем дельту для второго скрытого слоя
            delta2[i] = delta3 * W3[i] * df(out2[i])
            for j in range(4):
                # вычисляем сумму вхождений
                delta1[i] += delta2[j] * W2[i][j]
            # домножаем на производную (дельта 1 скрытого слоя)
            delta1[i] *= df(out1[i])

        for i in range(4):
            for j in range(4):
                # меняем веса между первым и вторым скрытыми слоями
                W2[i][j] -= lmd * delta2[i] * out1[i]

        for i in range(3):
            for j in range(4):
                # меняем веса между сенсорным и первым скрытыми слоями
                W1[i][j] -= lmd * delta1[j] * x[i]



train(epoch)

print("\nИзмененные матрицы весов:\n")
print("W1:\t", W1, "\n")
print("W2:\t", W2, "\n")
print("W3:\t", W3, "\n")

print("\nРезультат:\n")
for x in epoch:
    y, out1, out2 = go_forward(x[0:3])
    print(f"Выходное значение НС: {y}; ожидалось: {x[-1]}")