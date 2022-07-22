# coding: utf-8

import numpy as np
from numpy import linalg as LA
import sys


# ## Функция создания матрицы
# (приложение 2, пример 2, вариант 4)

# Функция генерирует расишренную матрице СЛАУ, заданной вариантом задания
def create_LAES(x):
    n = 100
    M = 4
    q = 1.001 - 2 * M * 10**(-3)
    
    # матрица системы
    A = np.zeros((n, n + 1))
    
    for i in range(n):
        for j in range(n):
            if (i == j):
                A[i][j] = (q - 1)**(i + j)
            else:
                A[i][j] = q**(i + j) + 0.1 * (j - i)
                
    # вычисляем столбец правой части системы
    vec_x = np.full(n, x)
    vec_x_div_i = np.array([x / i for i in range(1, n+1)])

    A[:, n] = vec_x * np.exp(vec_x_div_i) * np.cos(vec_x_div_i)
    
    return A


# ## Реализация метода верхней релаксации

def UpperRelaxMethod(A, f, omega, eps, print_iters=False):
    # Поскольку все матрицы исходных уравнений не удовлетворяли условиям теоремы Самарского,
    # вместо исходной системы Ax=f расмматривается система (A^T)Ax = (A^T)f, имеющая то же 
    # решение, но уже с симметричной и положительно определенной матрицей коэффициентов
    
    n = A.shape[0]
    B = np.dot(A.transpose(), A) # B = (A^T)A
    g = np.dot(A.transpose(), f) # g = (A^T)f
    
    # вектор решений
    x = np.zeros(n)

    # количество итераций, понадобившихся для решения
    iters = 0
    
    # пока норма невязки больше заданной точности:
    while LA.norm(g - np.dot(B, x)) > eps:
        iters += 1
        
        # вычисляем компоненты решения в соответствии с рекуррентной формулой
        for i in range(n):
            x[i] = x[i] + omega / B[i][i] * (g[i] - np.dot(B[i], x))
        
        if print_iters == True and iters % 10000 == 0:
            print(f"{iters} итерация; норма невязки: {LA.norm(g - np.dot(B, x))}")
    
    print(f"Решение найдено за {iters} итераций")
    
    return x

def main():
    print("Выберите способ задания матрицы:\n"
        "1 - выбор матрицы из варианта задания\n"
        "2 - ввод матрицы из файла\n"
        "3 - ввод матрицы через стандартный поток ввода")

    mode = int(input())

    if mode == 1:
        print("Выберите номер матрицы (от 1 до 4): ")
        num = input()
        if num == '4':
            x = float(input("Введите х: "))
            extend_A = create_LAES(x)
        else:
            file_name = 'matrix_' + num + '.txt'
            extend_A = np.loadtxt(file_name, delimiter=' ')
    elif mode == 2:
        file_name = input('Введите имя файла: ')
        extend_A = np.loadtxt(file_name, delimiter=' ')
    elif mode == 3:
        print("Введите расишренную матрицу СЛАУ:")
        extend_A = np.loadtxt(sys.stdin, delimiter=' ')
    else:
        print("Invalid input")
        return 1

    A = extend_A[:, : -1]
    f = extend_A[:, -1]

    print(A)
    print(f"f: {f}")

    eps = float(input('Введите точность: '))
    omega = float(input('Введите параметр омега: '))

    x = UpperRelaxMethod(A, f, omega=omega, eps=eps, print_iters=False)
    print(f"Решение:\n{x}")

    print("Для проверки корректности решения с помощью библиотеки numpy введите 1, иначе введите 0")
    check = int(input())

    if check == 1:
        print("Решение системы функцией numpy.linalg.solve:")
        print(LA.solve(A, f))


if __name__ == '__main__':
    main()
