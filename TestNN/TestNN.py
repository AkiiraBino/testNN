import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
import time

fileinput = open("input.txt", "r")
fileoutput = open("output.txt", "r")

input = []
output = []

with open("input.txt") as file:
    for line in file:
        input.append([int(x) for x in line.split()])

with open("output.txt") as file:
    for line in file:
        output.append([int(x) for x in line.split()])

c = np.array(input)
f = np.array(output)

x_val = [200]
y_val = [200]

model = keras.Sequential()  
model.add(Dense(units = 2, input_shape = (1, ), activation = 'linear'))
model.add(Dense(units = 1, input_shape = (1, ), activation = 'linear'))

model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(0.1))
t0 = time.time()
history = model.fit(c, f, epochs = 100, verbose = 0,  validation_data=(x_val, y_val))
t0 = time.time() - t0
print('\nhistory linear dict:', history.history['loss'])

print('\n Время работы: ', t0)