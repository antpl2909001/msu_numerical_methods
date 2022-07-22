# coding: utf-8

import numpy as np
from numpy import linalg as LA
import sys

# # Классический метод Гаусса

# В данной функции реализован прямой ход метода Гаусса нахождения 
# решения неоднородной СЛАУ с расширенной матрицей A из R^(n*n+1)
# (Функция вернет расширенную матрицу, получившуюся в результате)
#
# В случае, если for_det=True, функция вернет массив с ведущими элементами 
# матрицы на каждом шаге прямого хода метода Гаусса, причем матрица А 
# будет инетрпретироваться как матрица из R^(n*n) (последним 
# элементом массива ведущих элементов будет число (-1)^k, где k - количество 
# перестановок строк)
# Эта опция предусмотрена для нахожедния определителя матрицы А методом Гаусса

def GaussStraightRun(A, for_det=False):
    n = A.shape[0]
    
    # счетчик перестановок строк
    k = 0
    
    # массив ведущих элементов
    lead_elems = np.zeros((n+1,))
    
    # B - копия матрицы А, будет укорачиваться на каждом шаге прямого хода
    C = A.copy()
    B = C
    
    for i in range(n):
        # находим номера строк с ненулевыми элементами
        nonzero_indexes = (B[:, 0].nonzero())[0]

        # в 
        if nonzero_indexes.shape[0] == 0:
            if for_det == True:
                return lead_elems
            else:
                return None
        lead_el_row_num = nonzero_indexes[0]
        
        # меняем первую строку и строку с ведущим элементом
        if (lead_el_row_num != 0):
            B[0], B[lead_el_row_num] = B[lead_el_row_num], B[0].copy()
            k += 1
        # сохраняем ведущий элемент
        lead_elems[i] = B[0][0]

        # делим строку на ее главный элемент
        B[0] = B[0] / B[0][0]
        
        for j in range(1, B.shape[0]):
            B[j] = B[j] - B[0] * B[j][0]
        
        # "укорачиваем" матрицу (всё равно что рассмотреть убираем верхнюю строку и левый столбец)
        B = B[1:, 1:]
    
    # добавляем в массив ведущих элементов последним элементом коэффициент (-1) ** k
    lead_elems[n] = (-1)**k
    
    if (for_det == True):
        return lead_elems
    else:
        return C

# В функции реализован обратный ход метода Гаусса
def GaussReverseRun(C):
    if type(C) == type(None):
        return None
    n = C.shape[0]
    x = np.zeros((n,))
    
    for i in range(n-1, -1, -1):
        x[i] = C[i][n] - C[i, :-1].dot(x)
    
    return x

# Функция объединяет прямой и обратный ход метода Гаусса
def GaussMethod(A):
    C = GaussStraightRun(A)
    return GaussReverseRun(C)



# # Метод Гаусса с выбором главного элемента

# В данной функции реализован прямой ход модифицированого метода Гаусса нахождения 
# решения неоднородной СЛАУ с расширенной матрицей A из R^(n*n+1)
# (Функция вернет расширенную матрицу, получившуюся в результате)
#
# В случае, если for_det=True, функция вернет массив с ведущими элементами 
# матрицы на каждом шаге прямого хода метода Гаусса, причем матрица А 
# будет инетрпретироваться как матрица из R^(n*n) (последним 
# элементом массива ведущих элементов будет число (-1)^k, где k - количество 
# перестановок строк)
# Эта опция предусмотрена для нахожедния определителя матрицы А методом Гаусса

def ModifiedGaussStraightRun(A, for_det=False):
    n = A.shape[0]
    
    # массив нумерации переменных
    var_indexes = np.array(list(range(n)), dtype='int32')
    
    # количество перестановок столбцов
    k = 0
    
    # массив ведущих элементов
    lead_elems = np.zeros((n+1,))
    
    # C - копия матрицы А
    C = A.copy()
    
    for i in range(n):
        #  находим номер столбца с главным элементом
        main_el_col_num = (np.fabs(C[i,:-1])).argmax()
        
        # меняем первый столбец и столбец с главным элементом
        if (main_el_col_num != i):
            # меняем непосредственно столбцы
            C[:, i], C[:, main_el_col_num] = C[:, main_el_col_num], C[:, i].copy()
            # соответственно меняем номера переменных местами
            var_indexes[i], var_indexes[main_el_col_num] = var_indexes[main_el_col_num], var_indexes[i]
            # увеличиваем счетчик перестановок (нужно для вычисления знака определителя)
            k += 1

        # "запоминаем" ведущий элемент
        lead_elems[i] = C[i][i]
        C[i] = C[i] / C[i][i]
        
        for j in range(i + 1, n):
            C[j] = C[j] - C[i] * C[j][i]
    
    # добавляем в массив ведущих элементов последним элементом коэффициент (-1) ** k
    lead_elems[n] = (-1)**k
    
    # функция вызывалась для решения системы
    if for_det == False:
        return C, var_indexes
    # функция вызывалась для нахождения определителя матрицы
    else:
        return lead_elems

def ModifiedGaussReverseRun(C, var_indexes):
    n = C.shape[0]
    x = np.zeros((n,))
    
    for i in range(n-1, -1, -1):
        x[i] = C[i][n] - C[i, :-1].dot(x)
    
    # возвращаем решение системы в соответствии с нумерацией
    return x[np.argsort(var_indexes)]

# Функция объединяет в себе прямой и обратный ход модифицированного метода Гаусса
def ModifiedGaussMethod(A):
    C, indexes = ModifiedGaussStraightRun(A)
    return ModifiedGaussReverseRun(C, indexes)



# ## Функции нахождения определителя матрицы, обратной матрицы и числа обусловленности


# Функция вычисляет определитель матрицы А
def det(A, modified=False):
    if modified == False:
        return GaussStraightRun(A, for_det=True).prod()
    else:
        return ModifiedGaussStraightRun(A, for_det=True).prod()

# Функция находит матрицу А^(-1), обратную к А, методом Гаусса
# (Применяем прямой ход метода Гаусса к матрице А|E, затем левую подматрицу приводим
# к единичной, и результат есть правая подматрица)
def inverse_matrix(A):
    n = A.shape[0]
    E = np.eye(n)
    AE = np.concatenate((A, E), axis=1)
    C = GaussStraightRun(AE)
    
    for i in range(n - 1, 0, -1):
        for j in range(i):
            C[j] = C[j] - C[i] * C[j][i]

    return C[:, n:]

# Функция вычисляет число обусловленности матрицы А
def calc_condition_number(A):
    inv_A = inverse_matrix(A)
    return LA.norm(A) * LA.norm(inv_A)


# ## Функция создания матрицы
# (приложение 2, пример 2, вариант 4)

# Функция генерирует расишренную матрице СЛАУ, заданной вариантом задания
def create_LAES(x):
    n = 100
    M = 4
    q = 1.001 - 2 * M * 10**(-3)
    
    # матрица коэффициентов системы
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

def main():
    mode = int(input('1 - выбор матрицы из варианта задания\n'
        '2 - ввод матрицы из файла\n'
        '3 - ввод матрицы через стандартный поток ввода\n'))
    
    if mode == 1:
        print("Выберите номер матрицы (1, 2, 3, 4):")
        num = input()
        if num == '4':
            x = float(input("Введите x:"))
            A = create_LAES(x)
        else:
            file_name = 'matrix_' + num + '.txt'
            A = np.loadtxt(file_name, delimiter=' ') 
    elif mode == 2:
        file_name = input('Введите имя файла:  ')
        A = np.loadtxt(file_name, delimiter=' ')
    elif mode == 3:
        A = np.loadtxt(sys.stdin, delimiter=' ')
    else:
        return 1

    if A.shape[0] == A.shape[1]:
        print('Вы ввели квадратную матрицу. Для нее доступны следующие операции:\n'
            '1 - найти определитель матрицы\n'
            '2 - найти обратную матрицу\n'
            '3 - найти число обусловленности\n')
        cmds = input('Для выполнения операций введите строку с соответствующими цифрами\n'
                    '(Пример: при вводе 123 программа найдет определитель матрицы, обратную к ней и ее число обусловленности)')

        for cmd in cmds:
            if cmd == '1':
                print(f"Определитель матрицы: {det(A)}")
            elif cmd == '2':
                print(f"Обратная матрица:\n{inverse_matrix(A)}")
            elif cmd == '3':
                print(f"число обусловленности: {calc_condition_number(A)}")
    else:
        print('Вы ввели расширенную матрицу СЛАУ. Для нее доступны следующие операции:\n'
            '1 - найти определитель матрицы коэффициентов\n'
            '2 - найти обратную матрицу к матрице коэффициентов\n'
            '3 - найти число обусловленности матрицы коэффициентов\n'
            '4 - решить систему классическим методом Гаусса\n'
            '5 - решить систему методом Гаусса с выбором главного элемента\n')
        cmds = input('Для выполнения операций введите строку с соответствующими цифрами\n'
            '(Пример: при вводе 123 программа найдет определитель матрицы, обратную к ней и ее число обусловленности)\n')

        A_coef = A[:, :-1]

        for cmd in cmds:
            if cmd == '1':
                print(f"Определитель матрицы: {det(A_coef)}")
            elif cmd == '2':
                print(f"Обратная матрица:\n{inverse_matrix(A_coef)}")
            elif cmd == '3':
                print(f"число обусловленности: {calc_condition_number(A_coef)}")
            elif cmd == '4':
                print(f"Решение системы классическим методом Гаусса:")
                print(GaussMethod(A))
            elif cmd == '5':
                print(f"Решение системы модифицированным методом Гаусса:")
                print(ModifiedGaussMethod(A))

if __name__ == '__main__':
    # для вывода элементов матрицы с округлением до 5 знака разкомментируйте следующую строку:
    # np.set_printoptions(precision=5, suppress=True)
    main()
