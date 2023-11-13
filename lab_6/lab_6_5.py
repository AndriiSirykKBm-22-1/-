import numpy as np
import neurolab as nl

target = [
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0,
     1, 0, 1, 0, 1,
     1, 1, 1, 0, 1,
     1, 0, 1, 0, 1,
     1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 0,
     1, 0, 0, 0, 0,
     0, 1, 1, 1, 0]
]

chars = ['С', 'А', 'Ю']

target = np.asfarray(target)
target[target == 0] = -1

# Создание и обучение сети
net = nl.net.newhop(target)
output = net.sim(target)

print("Тест на обучающих данных:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

print("\nТест на искаженном С:")
test = np.asfarray([1, 1, 1, 1, 1,
                     1, 0, 1, 0, 1,
                     1, 0, 1, 0, 1,
                     1, 0, 1, 0, 1,
                     1, 0, 0, 0, 1])
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))

print("\nТест на искаженном А:")
test = np.asfarray([1, 1, 0, 1, 0,
                     1, 0, 1, 0, 1,
                     1, 1, 1, 0, 1,
                     1, 0, 1, 0, 1,
                     1, 1, 1, 1, 0])
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))

print("\nТест на искаженном Ю:")
test = np.asfarray([0, 1, 1, 1, 1,
                     0, 1, 0, 0, 1,
                     0, 1, 0, 1, 0,
                     1, 1, 1, 1, 1,
                     0, 1, 1, 1, 0])
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[2]).all(), 'Sim. steps', len(net.layers[0].outs))
