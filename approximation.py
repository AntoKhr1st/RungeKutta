from solution import runge_kutta_solve
import numpy as np
import math
import matplotlib.pyplot as plt

# Зададим те же начальные условия для задачи Коши
x0 = 0
y0 = 0
dy_dx0 = 3
d2y_dx20 = -9
d3y_dx30 = -8
d4y_dx40 = 0

x_end = 5

# Задаем шаг
h = 0.01

# используем результат из solution.py
x, y = runge_kutta_solve(h, x0, y0, dy_dx0, d2y_dx20, d3y_dx30, d4y_dx40, x_end)


# для оценки точности полученного численного решения будем искать разницу между y[i] значением численного решения
# в узле сетки со значением аналитического решения в той же точке i
def comparison(y, h, x0, x_end):
    num_steps = int((x_end - x0) / h)
    x = np.linspace(x0, x_end, num_steps + 1)  # разбиваем отрезок [0,5] равномерно на num_steps кусков, формируем сетку
    z = np.zeros(
        num_steps + 1) # создаем вектор длины num_steps + 1 с нулевыми значениями
    # заполняем вектор z значениями разности между численным и аналитическим решениями
    for i in range(num_steps):
        x_i = x[i] # переобозначил для краткости записи в следующей формуле
        z[i] = y[i] - ((-1 / 12) * math.exp(-3 * x_i) * x_i * (129 * x_i * x_i * x_i + 16 * x_i * x_i - 54 * x_i - 36))
        z[i] = round(z[i], 8) # округлил разность до 8 знака после запятой для наглядности результата
    return z


diff = comparison(y, h, x0, x_end) # вектор позволит оценить точность численного решения

plt.plot(x, diff)
plt.xlabel('x')
plt.ylabel('diff')
plt.title('diff between results')
plt.show()
