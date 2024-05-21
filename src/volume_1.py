import numpy as np

from scipy import stats
from scipy import optimize
import scipy.integrate as integrate

import matplotlib.pyplot as plt

import re

n = 500             
MX = 3.9           
DX = 4.3            
SIGMA = np.sqrt(DX)
SEED = 0          

x: np.ndarray = stats.norm.rvs(loc=MX, scale=SIGMA, size=n, random_state=SEED)
print(f"Первые 6 значений вектора:\n{re.sub(" +", ", ", str(x[:6])[1:-1])}...")
print(f"Размерность выборки: {str(x.shape)[1:-2]}")

_, theoretic_x, _ = plt.hist(x, 30, density=True)

theoretic_normal_distribution = 1 / \
    (SIGMA * np.sqrt(2 * np.pi)) * \
    np.exp(- (theoretic_x - MX)**2 / (2 * SIGMA**2))

plt.plot(theoretic_x, theoretic_normal_distribution, linewidth=3, color='r')
plt.show()

sample_average = x.sum() / n

sum_tmp = 0

for i in range(n):
    sum_tmp += (x[i] - sample_average) ** 2

sample_variance = sum_tmp / n

print(
    f"Выборочная средняя:{sample_average}\n" +
    f"Выборочная дисперсия: {sample_variance}"
)


def likelihood(params, x):
    return stats.norm.logpdf(x, loc=params[0], scale=params[1]).sum()


plt.plot(np.linspace(1, 10, 1000), [likelihood(
    [MX, val], x) for val in np.linspace(1, 10, 1000)])
plt.show()


def neglikelihood_MX(MX):
    return -1 * likelihood([MX, SIGMA], x)


def neglikelihood_SIGMA(SIGMA):
    return -1 * likelihood([MX, SIGMA], x)


estimator_MX = optimize.minimize(neglikelihood_MX, 10).x
estimator_SIGMA = optimize.minimize(neglikelihood_SIGMA, 10).x
print(f"Оценка мат. ожидания: {str(estimator_MX)[
      1:-1]}\nОценка дисперсии: {str(estimator_SIGMA ** 2)[1:-1]}")

unbiased_variance = (n * sample_variance) / (n - 1)
print(f"Несмещенная оценка дисперсии: {unbiased_variance}")

interval_MX_with_known_DX = sample_average - 1.645 * \
    (SIGMA / np.sqrt(n)), sample_average + 1.645 * (SIGMA / np.sqrt(n))

print(f"Интервальная оценка MX, с известной DX: {interval_MX_with_known_DX}")

sum_tmp = 0

for i in range(n):
    sum_tmp += (x[i] - MX) ** 2

interval_DX_with_known_MX = sum_tmp / 552.84646, sum_tmp / 448.85956

print(f"Интервальная оценка DX, с известным MX: {interval_DX_with_known_MX}")

interval_MX_with_unknown_DX = (
    sample_average - 1.645 * (np.sqrt(sample_variance) / np.sqrt(n - 1)),
    sample_average + 1.645 * (np.sqrt(sample_variance) / np.sqrt(n - 1))
)
print(f"Интервальная оценка MX, с неизвестной DX: {
      interval_MX_with_unknown_DX}")

interval_DX_with_unknown_MX = (
    (n * sample_variance) / (n - 1 + 1.645 * np.sqrt(2 * (n - 1))),
    (n * sample_variance) / (n - 1 - 1.645 * np.sqrt(2 * (n - 1)))
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

print(f"Относительные частоты: {relative_frequencies}")
print(f"Сумма относительных частот: {sum(relative_frequencies)}")


def erf(x):
    return (2 / np.sqrt(np.pi)) * integrate.quad(lambda t: np.exp(- t**2), 0, x)[0]


def F(x, MX, SIGMA):
    return 0.5 * (1 + erf((x - MX) / (SIGMA * np.sqrt(2))))


p = [F(val[-1], MX, SIGMA) - F(val[0], MX, SIGMA) for val in J]

for (k, pk) in enumerate(p):
    print(f"p{k + 1}{"" if k == 9 else " "} = {pk}")

chi = 0
for k in range(N):
    chi += (frequencies[k] - n * p[k])**2 / n * p[k]
print(f"Критерий Пирсона: {chi}")


p = [F(val[-1], estimator_MX, estimator_SIGMA) -
     F(val[0], estimator_MX, estimator_SIGMA) for val in J]

for (k, pk) in enumerate(p):
    print(f"p{k + 1}{"" if k == 9 else " "} = {pk}")

chi = 0
for k in range(N):
    chi += (frequencies[k] - n * p[k])**2 / n * p[k]
print(f"Критерий Пирсона: {chi}")
