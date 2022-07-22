# coding: utf-8

import math
import numpy as np

# ## Метод Рунге-Кутта 2 и 4 порядков точности

def RungeKutt2Method(f, x_start, y_start, h, step_cnt):
    # сетка
    x = np.arange(x_start, x_start + h * (step_cnt + 1), h)
    
    # размер сетки
    n = x.shape[0]
    
    # сеточная функция
    y = np.zeros((n, y_start.shape[0]))
    y[0] = y_start
    
    # вычисление компонент сеточной функции
    for i in range(1, n):
        # "предсказание"
        y[i] = y[i - 1] +  f(x[i - 1], y[i - 1]) * h
        
        # "корректировка"
        y[i] = y[i - 1] + h / 2 * (f(x[i - 1], y[i - 1]) + f(x[i], y[i]))
    
    return (x, y)


def RungeKutt4Method(f, x_start, y_start, h, step_cnt):
    # сетка
    x = np.arange(x_start, x_start + h * (step_cnt + 1), h)
    
    # размер сетки
    n = x.shape[0]
    
    # сеточная функция
    y = np.zeros((n, y_start.shape[0]))
    y[0] = y_start
    
    # вычисление компонент сеточной функции
    for i in range(1, n):
        # вычисляем вспомогательные значения функции f
        k1 = f(x[i-1], y[i-1])
        k2 = f(x[i-1] + h/2, y[i-1] + h/2 * k1)
        k3 = f(x[i-1] + h/2, y[i-1] + h/2 * k2)
        k4 = f(x[i-1] + h, y[i-1] + h * k3)
        
        # вычисляем саму компоненту сеточной функции
        y[i] = y[i-1] + h/6 * (k1 + 2*k2 +2*k3 + k4)
        
    return (x, y)


# ## Функции из варианта задания:

# ### Основные тесты

# Таблица 1, вариант 2
def f(x, y):
    return math.sin(x) - y

# Аналитическое решение
def f_solution(x):
    return -0.5 * np.cos(x) + 0.5 * np.sin(x) + 21/2 * np.exp(-x)

# Таблица 2, вариант 8
def f1(x, u, v):
    return math.cos(x + 1.1*v) + u

def f2(x, u, v):
    return -v**2 + 2.1*u + 1.1

def F(x, y):
    return np.array([f1(x, y[0], y[1]), f2(x, y[0], y[1])])


# ### Дополнительные тесты

# Таблица 1, вариант 4
def g(x, y):
    return y - y*x

def g_solution(x):
    return 5 * np.exp(-0.5 * x * (x - 2))

# Таблица 2, вариант 2
def g1(x, u, v):
    return x * u - v

def g2(x, u, v):
    return u - v

def G(x, y):
    return np.array([g1(x, y[0], y[1]), g2(x, y[0], y[1])])


def main():
    print("Варианты 1, 2 - задача Коши для обыкновенного дифференциального уравнения")
    print("Варианты 3, 4 - задача Коши для системы обыкновенных дифференциальных уравнений")    
    print("Выберите вариант задачи (1, 2, 3, 4)")

    var = int(input())

    print("Введите порядок точности метода Рунге-Кутта:")
    accur_order = int(input())

    if accur_order != 2 and accur_order != 4:
        print("Доступные порядки точности: 2, 4")
        return

    RungeKuttMethod = {2: RungeKutt2Method, 4: RungeKutt4Method}

    if var == 1:
        print("Уравнение: y' = sin(x) - y\nначальное условие y(0) = 10")
        x_start = 0
        y_start = np.array([10])
        func = f
    elif var == 2:
        print("Уравнение: y' = y - yx\nначальное условие: y(0) = 5")
        x_start = 0
        y_start = np.array([5])
        func = g
    elif var == 3:
        print("Система уравнений:")
        print("y1' = cos(x + 1.1 * (y2)) + y1")
        print("y2' = -(y2)^2 + 2.1 * y1 + 1.1")
        print("Начальные условия: y1(0) = 0.25, y2(0) = 1")
        x_start = 0
        y_start = np.array([0.25, 1])
        func = F
    elif var == 4:
        print("Система уравнений:")
        print("y1' = x*y1 + y2")
        print("y2' = y1 - y2")
        print("Начальные условия: y1(0) = 0, y2(0) = 1")
        x_start = 0
        y_start = np.array([0, 1])
        func = G
    else:
        print("Такого варианта нет")
        return

    print("Введите размер шага:")
    h = float(input())
    print("Введите количество шагов сетки:")
    step_cnt = int(input())

    x, y = (RungeKuttMethod[accur_order])(func, x_start, y_start, h, step_cnt)

    if (var < 3):
        print("Первый столбец - сетка, второй столбец - значения сеточной функции")
        for x_coord, y_coord in zip(x, y):
            print(x_coord, y_coord[0])
    else:
        print("Первый столбец - сетка, второй и третий столбцы - значения сеточных функций y1 и y2")
        for x_coord, y_coord in zip(x, y):
            print(x_coord, y_coord[0], y_coord[1])

    return

if __name__ == "__main__":
    main()
