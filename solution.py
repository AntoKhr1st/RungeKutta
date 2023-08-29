import numpy as np
import matplotlib.pyplot as plt



def f(x, y, dy_dx, d2y_dx2, d3y_dx3, d4y_dx4):
    return -15 * d4y_dx4 - 90 * d3y_dx3 - 270 * d2y_dx2 - 405 * dy_dx - 243 * y


# Буду использовать схему Рунге-Кутты для численного решения уравнения (шаг сетки h и другие параметры зададим снаружи функции)
def runge_kutta_solve(h, x0, y0, dy_dx0, d2y_dx20, d3y_dx30, d4y_dx40, x_end):
    num_steps = int((x_end - x0) / h)
    x = np.linspace(x0, x_end, num_steps + 1)  # разбиваем отрезок [0,5] равномерно на num_steps кусков, формируем сетку
    y = np.zeros(
        num_steps + 1)  # y будем искать как вектор значений функции в узлах сетки (сетка из предыдущего шага)
    dy_dx = np.zeros(num_steps + 1)  # вектор длины num_steps + 1 для разностной схемы
    d2y_dx2 = np.zeros(num_steps + 1)  # вектор длины num_steps + 1 для разностной схемы
    d3y_dx3 = np.zeros(num_steps + 1)  # вектор длины num_steps + 1 для разностной схемы
    d4y_dx4 = np.zeros(num_steps + 1)  # вектор длины num_steps + 1 для разностной схемы
    y[0] = y0  # начальные условия для задачи Коши зададим вне функции
    dy_dx[0] = dy_dx0  # начальные условия для задачи Коши зададим вне функции
    d2y_dx2[0] = d2y_dx20  # начальные условия для задачи Коши зададим вне функции
    d3y_dx3[0] = d3y_dx30  # начальные условия для задачи Коши зададим вне функции
    d4y_dx4[0] = d4y_dx40  # начальные условия для задачи Коши зададим вне функции

    # используем метод Рунге-Кутты для численного решения дифференциального уравнения на заданной сетке
    for i in range(num_steps):
        k1y = h * dy_dx[i]
        k1dy_dx = h * d2y_dx2[i]
        k1d2y_dx2 = h * d3y_dx3[i]
        k1d3y_dx3 = h * d4y_dx4[i]
        k1d4y_dx4 = h * f(x[i], y[i], dy_dx[i], d2y_dx2[i], d3y_dx3[i], d4y_dx4[i])

        k2y = h * (dy_dx[i] + k1dy_dx / 2)
        k2dy_dx = h * (d2y_dx2[i] + k1d2y_dx2 / 2)
        k2d2y_dx2 = h * (d3y_dx3[i] + k1d3y_dx3 / 2)
        k2d3y_dx3 = h * (d4y_dx4[i] + k1d4y_dx4 / 2)
        k2d4y_dx4 = h * f(x[i] + h / 2, y[i] + k1y / 2, dy_dx[i] + k1dy_dx / 2, d2y_dx2[i] + k1d2y_dx2 / 2,
                          d3y_dx3[i] + k1d3y_dx3 / 2, d4y_dx4[i] + k1d4y_dx4 / 2)

        k3y = h * (dy_dx[i] + k2dy_dx / 2)
        k3dy_dx = h * (d2y_dx2[i] + k2d2y_dx2 / 2)
        k3d2y_dx2 = h * (d3y_dx3[i] + k2d3y_dx3 / 2)
        k3d3y_dx3 = h * (d4y_dx4[i] + k2d4y_dx4 / 2)
        k3d4y_dx4 = h * f(x[i] + h / 2, y[i] + k2y / 2, dy_dx[i] + k2dy_dx / 2, d2y_dx2[i] + k2d2y_dx2 / 2,
                          d3y_dx3[i] + k2d3y_dx3 / 2, d4y_dx4[i] + k2d4y_dx4 / 2)

        k4y = h * (dy_dx[i] + k3dy_dx)
        k4dy_dx = h * (d2y_dx2[i] + k3d2y_dx2)
        k4d2y_dx2 = h * (d3y_dx3[i] + k3d3y_dx3)
        k4d3y_dx3 = h * (d4y_dx4[i] + k3d4y_dx4)
        k4d4y_dx4 = h * f(x[i] + h, y[i] + k3y, dy_dx[i] + k3dy_dx, d2y_dx2[i] + k3d2y_dx2, d3y_dx3[i] + k3d3y_dx3,
                          d4y_dx4[i] + k3d4y_dx4)

        y[i + 1] = y[i] + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        dy_dx[i + 1] = dy_dx[i] + (k1dy_dx + 2 * k2dy_dx + 2 * k3dy_dx + k4dy_dx) / 6
        d2y_dx2[i + 1] = d2y_dx2[i] + (k1d2y_dx2 + 2 * k2d2y_dx2 + 2 * k3d2y_dx2 + k4d2y_dx2) / 6
        d3y_dx3[i + 1] = d3y_dx3[i] + (k1d3y_dx3 + 2 * k2d3y_dx3 + 2 * k3d3y_dx3 + k4d3y_dx3) / 6
        d4y_dx4[i + 1] = d4y_dx4[i] + (k1d4y_dx4 + 2 * k2d4y_dx4 + 2 * k3d4y_dx4 + k4d4y_dx4) / 6

    return x, y


# Задаем начальные условия
x0 = 0
y0 = 0
dy_dx0 = 3
d2y_dx20 = -9
d3y_dx30 = -8
d4y_dx40 = 0

# Задаем конечную точку
x_end = 5

# Задаем шаг
h = 0.01

# Решаем дифференциальное уравнение методом Рунге-Кутта
x, y = runge_kutta_solve(h, x0, y0, dy_dx0, d2y_dx20, d3y_dx30, d4y_dx40, x_end)

# Выводим результаты
plt.plot(x, y,'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical solution of the differential equation')
plt.show()


