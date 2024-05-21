import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import re

n = 300
SEED = 0


def f(x, a=2):
    if abs(x) > a:
        return 0
    else:
        return (1 - abs(x) / a) / a


def F(x, a=2):
    if x < -a:
        return 0
    elif -a <= x < 0:
        return 0.5 + x / a + (x**2 / (2 * a**2))
    elif 0 <= x <= a:
        return 0.5 + x / a - (x**2 / (2 * a**2))
    else:
        return 1


x = np.linspace(-3, 3, 100_000)
y = [f(val) for val in x]
plt.plot(x, y)
plt.show()


x = np.linspace(-3, 3, 100_000)
y = [F(val) for val in x]
plt.plot(x, y)
plt.show()


def invF(y):
    if 0 <= y < 0.5:
        return (8 * y)**0.5 - 2
    elif 0.5 <= y <= 1:
        return 2 - (8 * (1 - y))**0.5


x = np.linspace(0, 1, 1000)
y = [invF(val) for val in x]
plt.plot(x, y)
plt.show()


uniform_selection = stats.uniform.rvs(size=n, random_state=SEED)


x = np.array([invF(val) for val in uniform_selection])
print(f"Первые 6 значений вектора:\n{re.sub(" +", ", ", str(x[:6])[2:-1])}...")
print(f"Размерность выборки: {str(x.shape)[1:-2]}")


_, theoretic_x, _ = plt.hist(x, 30, density=True)

theoretic_distribution = [f(val) for val in theoretic_x]

plt.plot(theoretic_x, theoretic_distribution, linewidth=3, color='r')
plt.show()


sample_average = x.sum() / n

sum_tmp = 0

for i in range(n):
    sum_tmp += (x[i] - sample_average) ** 2

sample_variance = sum_tmp / n

sum_tmp = 0

for i in range(n):
    sum_tmp += (x[i] - sample_average) ** 4

sample_variance_4 = sum_tmp / n

print(
    f"Выборочная средняя: {sample_average}\n" +
    f"Выборочная дисперсия: {sample_variance}\n" +
    f"выборочный центральный момент 4-го порядка: {sample_variance_4}"
)


interval_MX_with_unknown_DX = (
    sample_average - 1.96 * (np.sqrt(sample_variance) / np.sqrt(n)),
    sample_average + 1.96 * (np.sqrt(sample_variance) / np.sqrt(n))
)
print(f"Интервальная оценка MX, с неизвестной DX: {
      interval_MX_with_unknown_DX}")


interval_DX_with_unknown_MX = (
    sample_variance - 1.96 *
    (np.sqrt(sample_variance_4 - sample_variance**2) / np.sqrt(n)),
    sample_variance + 1.96 *
    (np.sqrt(sample_variance_4 - sample_variance**2) / np.sqrt(n))
)
print(f"Интервальная оценка DX, с неизвестным MX: {
      interval_DX_with_unknown_MX}")


variation_series = x.copy()
variation_series.sort()
print(f"Вариационный ряд: {
      re.sub(" +", ", ", str(variation_series[:6])[1:-1])}...")


print(f"Минимальное: {min(variation_series)}")
print(f"Максимальное: {max(variation_series)}")


N = 10


h = (max(variation_series) - min(variation_series)) / N
print(f"Длина интервала: {h}")


J = [[min(variation_series) + k * h, min(variation_series) + (k + 1) * h]
     for k in range(N)]

for (i, interval) in enumerate(J):
    print(f"Интервал {i + 1}: {str(interval)[:-1]})")


stat_series = [[] for _ in range(len(J))]

for xi in variation_series:
    for i in range(len(J)):
        if (J[i][0] <= xi < J[i][1]):
            stat_series[i].append(xi)
stat_series[-1].append(max(variation_series))


frequencies = []

for interval in stat_series:
    frequencies.append(len(interval))

relative_frequencies = [frequency / n for frequency in frequencies]

print(f"Частоты: {frequencies}")
print(f"Сумма частот: {sum(frequencies)}", end="\n\n")

print(f"Относительные частоты: {np.round(relative_frequencies, 2)}")
print(f"Сумма относительных частот: {sum(relative_frequencies)}")


p = [F(val[-1], a=2.01) - F(val[0], a=2.01) for val in J]

for (k, pk) in enumerate(p):
    print(f"p{k + 1}{"" if k == 9 else " "} = {pk}")


chi = 0
for k in range(N):
    chi += (frequencies[k] - n * p[k])**2 / n * p[k]
print(f"Критерий Пирсона: {chi}")
