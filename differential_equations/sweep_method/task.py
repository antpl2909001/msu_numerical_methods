# coding: utf-8

import numpy as np
import math

# ## Метод прогонки
# A, B, C, F - одномерные массивы трех главных диагоналей и столбца свободных членов соответственно
def SweepMethod(A, B, C, F):
    m = A.shape[0]
    
    # вектор-решение
    y = np.zeros(m)
    
    # коэффициенты альфа и бета
    alpha = np.zeros(m)
    beta = np.zeros(m)
    
    # вычисляем нулевые коэффициенты
    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]
    
    # прямая прогонка:
    # вычисляем по рекуррентной формуле оставшиеся прогоночные коэффициенты
    for i in range(1, m):
        alpha[i] = -C[i] / (A[i] * alpha[i-1] + B[i])
        beta[i] = (F[i] -  A[i] * beta[i-1]) / (A[i] * alpha[i-1] + B[i])
    
    # вычисляем последнюю компоненту решения    
    y[m-1] = (F[m-1] - A[m-1] * beta[m-2]) / (A[m-1] * alpha[m-2] + B[m-1])
    
    # обратная прогонка
    for i in range(m-1, 0, -1):
        y[i-1] = alpha[i-1] * y[i] + beta[i-1]
    
    return y


# ## Решение краевой задачи

# p(x), q(x), f(x) - функции из дифференциального уравнения
# [a, b] - отрезок, на котором решается краевая задача
# n - количество узлов сеточной функции
# sigma1, sigma2, gamma1, gamma2, delta1, delta2 задают граничные условия
def BoundaryValueSolution(p, q, f, a, b, n,
                          sigma1, sigma2, gamma1, 
                          gamma2, delta1, delta2):
    # вычисляем размер шага, формируем сетку
    h = (b-a) / n
    grid = np.linspace(a, b, n+1)

    # вычисляем значения функций p(x), q(x), f(x) в точках сетки
    grid_p = p(grid)
    grid_q = q(grid)
    grid_f = f(grid)
    
    # главные диагонали матрицы (снизу вверх)
    A = np.zeros(n+1)
    B = np.zeros(n+1)
    C = np.zeros(n+1)
    
    # веткор-столбец свободных членов
    F = np.zeros(n+1)
    
    # учитываем граничные условия
    B[0] = sigma1 * h - gamma1
    C[0] = gamma1
    F[0] = delta1 * h
    A[n] = -gamma2
    B[n] = sigma2 * h + gamma2
    F[n] = delta2 * h
    
    # вычисляем диагональные элементы матрицы системы
    for i in range(1, n):
        A[i] = 1 - grid_p[i] * h/2
        B[i] = grid_q[i] * h**2 - 2
        C[i] = 1 + grid_p[i] * h/2
        F[i] = -grid_f[i] * h**2
    
    # находим решение методом прогонки
    grid_y = SweepMethod(A, B, C, F)
    
    return (grid, grid_y)


# ##  Тестирование (Вариант 14 - основной)

# y'' + 2*x^2*y' + y = x 
# 2*y(0.5) - y'(0.5) = 1 
# y(0.8) = 3

def p1(x):
    return 2 * x**2

def q1(x):
    return np.full(x.shape, 1)

def f1(x):
    return -x

# ## Тестирование (Вариант 5 - дополнительный)

# y'' + 2 y' - x y = x^2 
# y'(0.6) = 0.7
# y(0.9) - 0.5 y'(0.9) = 1

def p2(x):
    return np.full(x.shape, 2)

def q2(x):
    return -x

def f2(x):
    return -x**2

# ## Дополнительный тест 1

# y'' + y = 1
# y(0)=0
# y'(1)=1

def solution_3(x):
    return -np.cos(x) + (1 - math.sin(1)) / math.cos(1) * np.sin(x) + 1

def p3(x):
    return np.full(x.shape, 0)

def q3(x):
    return np.full(x.shape, 1)

def f3(x):
    return np.full(x.shape, -1)


# ## Дополнительный тест 2

# y'' +  2y' = x, 
# y(0)=0,  
# y'(1)=1

# 1/8 (3 e^2 - 3 e^(2 - 2 x) + 2 (-1 + x) x)

def solution_4(x):
    return 1/8 * (2*x*(x-1) - 3 * np.exp(2 - 2*x) + 3 * np.exp(np.full(x.shape, 2)))

def p4(x):
    return np.full(x.shape, 2)

def q4(x):
    return np.full(x.shape, 0)

def f4(x):
    return -x


def main():
    print("Выберите вариант задания (1, 2, 3, 4):")

    var = int(input())

    p = {1: p1, 2: p2, 3: p3, 4: p4}
    q = {1: q1, 2: q2, 3: q3, 4: q4}
    f = {1: f1, 2: f2, 3: f3, 4: f4}

    if var == 1:
        # задаем параметры краевых условий
        a = 0.5; b = 0.8
        sigma1 = 2; sigma2 = 1
        gamma1 = -1; gamma2 = 0
        delta1 = 1; delta2 = 3
    elif var == 2:
        # задаем параметры краевых условий
        a = 0.6; b = 0.9
        sigma1 = 0; sigma2 = 1
        gamma1 = 1; gamma2 = -0.5
        delta1 = 0.7; delta2 = 1
    elif var == 3:
        # задаем параметры краевых условий
        a = 0; b = 1
        sigma1 = 1; sigma2 = 0
        gamma1 = 0; gamma2 = 1
        delta1 = 0; delta2 = 1
    elif var == 4:
        # задаем параметры краевых условий
        a = 0; b = 1
        sigma1 = 1; sigma2 = 0
        gamma1 = 0; gamma2 = 1
        delta1 = 0; delta2 = 1
    else:
        print("Такого варианта нет")
        return

    n = int(input("Введите количество шагов сетки: "))

    res = BoundaryValueSolution(p[var], q[var], f[var], a, b, n,
                              sigma1, sigma2, gamma1, 
                              gamma2, delta1, delta2)
    
    print("Вычисленное решение:")
    for tpl in zip(res[0], res[1]):
        print(f"({tpl[0]}, {tpl[1]})")

    if (var >= 3):
        print("Для выбранного варианта найдено аналитическое решение.")
        print("Для сравнения аналитического и вычисленного решения введите 1, иначе - 0.")

        check = int(input())
        if check == 1:
            solution = {3: solution_3, 4: solution_4}
            print("Аналитическое решение, вычисленное в точках сетки:")
            for tpl in zip(res[0], solution[var](res[0])):
                print(f"({tpl[0]}, {tpl[1]})")

if __name__ == "__main__":
    main()
