import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import re


n = 500

MX = 3.9
MY = -0.7

DX = 4.3
DY = 12.96

r = -0.05

SEED = 0

SIGMA_X = np.sqrt(DX)
SIGMA_Y = np.sqrt(DY)


R_XY = r * SIGMA_X * SIGMA_Y
print(f"Корреляционный момент: {R_XY}")


R = (
    (DX, R_XY),
    (R_XY, DY)
)
print(
f"""Корреляционная матрица: \n( {DX}  , {np.round(R_XY, 3)})\n({np.round(R_XY, 3)},  {DY})"""
)


vector_XY: np.ndarray = stats.multivariate_normal.rvs(
    mean=(MX, MY), cov=R, size=n, random_state=SEED)
x: np.ndarray = vector_XY[:, 0]
y: np.ndarray = vector_XY[:, 1]


print(
f"""Первые пара значений вектора (X, Y):\n({str(vector_XY[0])[1:-1]}), ({str(vector_XY[1])[1:-1]})..."""
)


sample_average_X = x.sum() / n

sum_tmp = 0

for i in range(n):
    sum_tmp += (x[i] - sample_average_X) ** 2

sample_variance_X = sum_tmp / n

print(
    f"Выборочная  средняя  X: {sample_average_X}\n" +
    f"Выборочная дисперсия X: {sample_variance_X}\n"
)


sample_average_Y = y.sum() / n

sum_tmp = 0

for i in range(n):
    sum_tmp += (y[i] - sample_average_Y) ** 2

sample_variance_Y = sum_tmp / n

print(
    f"Выборочная  средняя  Y: {sample_average_Y}\n" +
    f"Выборочная дисперсия Y: {sample_variance_Y}\n"
)


sample_r = np.sum(x * y) - n * sample_average_X * sample_average_Y
sample_r /= np.sqrt((np.sum(x**2) - n * sample_average_X**2)
                    * (np.sum(y**2) - n * sample_average_Y**2))

print(f"Выборочный коффицент корреляции: {sample_r}")


var_ser_x = x.copy()
var_ser_x.sort()
print(f"Вариационный ряд X: {re.sub(" +", ", ", str(var_ser_x[:4])[1:-1])}...")


print(f"Минимальное: {min(var_ser_x)}")
print(f"Максимальное: {max(var_ser_x)}")


N = 3


h_X = (max(var_ser_x) - min(var_ser_x)) / N
print(f"Длина интервала: {h_X}")


J_X = [[min(var_ser_x) + k * h_X, min(var_ser_x) + (k + 1) * h_X]
       for k in range(N)]

for (i, interval) in enumerate(J_X):
    print(f"Интервал {i + 1}: {str(interval)[:-1]})")


stat_series_X = [[] for _ in range(len(J_X))]

for xi in var_ser_x:
    for i in range(len(J_X)):
        if (J_X[i][0] <= xi < J_X[i][1]):
            stat_series_X[i].append(xi)
stat_series_X[-1].append(max(var_ser_x))


frequencies_X = []

for interval in stat_series_X:
    frequencies_X.append(len(interval))

relative_frequencies_X = [frequency / n for frequency in frequencies_X]

print(f"Частоты: {frequencies_X}")
print(f"Сумма частот: {sum(frequencies_X)}", end="\n\n")

print(f"Относительные частоты: {relative_frequencies_X}")
print(f"Сумма относительных частот: {sum(relative_frequencies_X)}")


var_ser_y = y.copy()
var_ser_y.sort()
print(f"Вариационный ряд X: {re.sub(" +", ", ", str(var_ser_y[:4])[1:-1])}...")


print(f"Минимальное: {min(var_ser_y)}")
print(f"Максимальное: {max(var_ser_y)}")


h_Y = (max(var_ser_y) - min(var_ser_y)) / N
print(f"Длина интервала: {h_Y}")


J_Y = [[min(var_ser_y) + k * h_Y, min(var_ser_y) + (k + 1) * h_Y]
       for k in range(N)]

for (i, interval) in enumerate(J_Y):
    print(f"Интервал {i + 1}: {str(interval)[:-1]})")


stat_series_Y = [[] for _ in range(len(J_Y))]

for yi in var_ser_y:
    for i in range(len(J_Y)):
        if (J_Y[i][0] <= yi < J_Y[i][1]):
            stat_series_Y[i].append(yi)
stat_series_Y[-1].append(max(var_ser_y))


frequencies_Y = []

for interval in stat_series_Y:
    frequencies_Y.append(len(interval))

relative_frequencies_Y = [frequency / n for frequency in frequencies_Y]

print(f"Частоты: {frequencies_Y}")
print(f"Сумма частот: {sum(frequencies_Y)}", end="\n\n")

print(f"Относительные частоты: {relative_frequencies_Y}")
print(f"Сумма относительных частот: {sum(relative_frequencies_Y)}")


rectangles = []

for side_X in J_X:
    for side_Y in J_Y:
        rectangles.append([side_X, side_Y])

for (i, rectangle) in enumerate(rectangles):
    print(f"Прямоугольник {i + 1}:\n\t" +
          f"Сторона X: [{str(np.round(rectangle[0], 3))[1:-1]})\n\t" +
          f"Сторона Y: [{str(np.round(rectangle[1], 3))[1:-1]})")


stat_series_rect = [[] for _ in range(len(rectangles))]

for (xi, yi) in (vector_XY):
    for (i, (side_X, side_Y)) in enumerate(rectangles):
        if (side_X[0] <= xi < side_X[1]) and (side_Y[0] <= yi < side_Y[1]):
            stat_series_rect[i].append((xi, yi))

stat_series_rect[-1].append(vector_XY[vector_XY[:, 0].argmax()])
stat_series_rect[-1].append(vector_XY[vector_XY[:, 1].argmax()])


frequencies_rect = []

for rect in stat_series_rect:
    frequencies_rect.append(len(rect))

relative_frequencies_rect = [frequency / n for frequency in frequencies_rect]

print(f"Частоты: {frequencies_rect}")
print(f"Сумма частот: {sum(frequencies_rect)}", end="\n\n")

print(f"Относительные частоты: {relative_frequencies_rect}")
print(f"Сумма относительных частот: {sum(relative_frequencies_rect)}")


sum_tmp = 0
ij = 0

for i in range(N):
    for j in range(N):
        sum_tmp += frequencies_rect[ij]**2 / \
            (frequencies_X[i] * frequencies_Y[j])
        ij += 1

chi = n * (sum_tmp - 1)
print(f"Критерий Пирсона: {chi}")


def reg_Y_X(x):
    res = sample_r
    res *= (np.sqrt(sample_variance_Y) / np.sqrt(sample_variance_X))
    res *= (x - sample_variance_X)
    return res + sample_average_Y


def reg_X_Y(y):
    res = sample_r
    res *= (np.sqrt(sample_variance_X) / np.sqrt(sample_variance_Y))
    res *= (y - sample_variance_Y)
    return res + sample_average_X


print(f"При коэффиценте корреляции: {r}")
plt.scatter(x, y, color="pink", linewidth=3)
plt.plot(np.linspace(-3, 10, 1000), list(map(reg_Y_X,
         np.linspace(-3, 10, 1000))), color="blue", linewidth=3)
plt.plot(np.linspace(-3, 10, 1000), list(map(reg_X_Y,
         np.linspace(-3, 10, 1000))), color="red",  linewidth=3)
plt.show()
