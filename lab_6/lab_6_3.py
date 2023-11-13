import numpy as np
import neurolab as nl

# Определение целевых и входных данных
target = [
    [-1, 1, -1, -1, 1, -1, -1, 1, -1],
    [1, 1, 1, 1, -1, 1, 1, -1, 1],
    [1, -1, 1, 1, 1, 1, 1, -1, 1],
    [1, 1, 1, 1, -1, -1, 1, -1, -1],
    [-1, -1, -1, -1, 1, -1, -1, -1, -1]
]

input_data = [
    [-1, -1, 1, 1, 1, 1, 1, -1, 1],
    [-1, -1, 1, -1, 1, -1, -1, -1, -1],
    [-1, -1, -1, -1, 1, -1, -1, 1, -1]
]

# Создание и тренировка нейронной сети
net = nl.net.newhem(target)
output = net.sim(target)

print("Тест на обучающих примерах (ожидаемые значения: [0, 1, 2, 3, 4])")
print(np.argmax(output, axis=0))

output = net.sim([input_data[0]])
print("Выводы на рекуррентном цикле:")
print(np.array(net.layers[1].outs))

output = net.sim(input_data)
print("Выводы на тестовых примерах:")
print(output)
