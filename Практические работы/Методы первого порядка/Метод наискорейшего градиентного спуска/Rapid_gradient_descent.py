import math
import sympy as sp
from tabulate import tabulate
from typing import Callable, List


class RapidGradientDescent:
    def __init__(self, fitness_function: Callable[..., float], x0: List[float], epsilon: float):
        '''
        Инициализация метода наискорейшего градиентного спуска для оптимизации функции.

        Параметры:
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
            x0 (List[float]): Начальная точка (вектор).
            epsilon (float): Точность для условия остановки.
        '''
        self.func = fitness_function
        self.n = len(x0)
        self.epsilon = epsilon
        self.arr = [[x0, self.func(*x0)]]
        self._variables = sp.symbols(f"x1:{self.n + 1}")
        self._gradient_expr = [sp.diff(self.func(*self._variables), var) for var in self._variables]
        self._hessian_expr = [[sp.diff(self._gradient_expr[i], var) for var in self._variables] for i in range(self.n)]

    def _print_table(self) -> None:
        '''
        Вывод текущего состояния в виде таблицы.
        '''
        headers = ["Вершина"] + [f"x{i+1}" for i in range(self.n)] + ["Значение функции"]
        table = [[f"X{i}"] + vertex[0] + [vertex[1]] for i, vertex in enumerate(self.arr)]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def _compute_gradient(self, x: List[float]) -> List[float]:
        '''
        Вычисление градиента функции в точке x.

        Параметры:
            x (List[float]): Точка, в которой вычисляется градиент.

        Возвращает:
            List[float]: Градиент функции в точке x.
        '''
        return [g.evalf(subs={self._variables[i]: x[i] for i in range(self.n)}) for g in self._gradient_expr]

    def _compute_hessian(self, x: List[float]) -> List[List[float]]:
        '''
        Вычисление матрицы Гессе функции в точке x.

        Параметры:
            x (List[float]): Точка, в которой вычисляется матрица Гессе.

        Возвращает:
            List[List[float]]: Матрица Гессе функции в точке x.
        '''
        return [[h.evalf(subs={self._variables[i]: x[i] for i in range(self.n)}) for h in row] for row in self._hessian_expr]

    def _check_stopping_condition(self, gradient: List[float]) -> bool:
        '''
        Проверка условия остановки.

        Параметры:
            gradient (List[float]): Градиент функции в текущей точке.

        Возвращает:
            bool: True, если условие остановки выполнено, иначе False.
        '''
        norm = math.sqrt(sum(g ** 2 for g in gradient))
        return norm < self.epsilon

    def _compute_step(self, gradient: List[float], hessian: List[List[float]]) -> float:
        '''
        Вычисление шага h_k.

        Параметры:
            gradient (List[float]): Градиент функции в текущей точке.
            hessian (List[List[float]]): Матрица Гессе функции в текущей точке.

        Возвращает:
            float: Оптимальный шаг h_k.
        '''
        numerator = sum(g ** 2 for g in gradient)

        denominator = 0.0
        for i in range(self.n):
            for j in range(self.n):
                denominator += gradient[i] * hessian[i][j] * gradient[j]

        return numerator / denominator

    def optimize(self, show_info: bool = True):
        '''
        Выполнение оптимизации методом наискорейшего градиентного спуска.

        Параметры:
            show_info (bool): Флаг для отображения информации о каждой итерации.

        Возвращает:
            List[float]: Точка минимума.
        '''
        x = self.arr[0][0]
        iteration = 0

        while True:
            gradient = self._compute_gradient(x)
            if self._check_stopping_condition(gradient):
                break

            hessian = self._compute_hessian(x)
            h_k = self._compute_step(gradient, hessian)

            x_new = [x[i] - h_k * gradient[i] for i in range(self.n)]
            f_new = self.func(*x_new)

            self.arr.append([x_new, f_new])
            x = x_new

            if show_info:
                print('\033[92m' + f'Итерация №{iteration}:' + '\033[0m')
                self._print_table()
                print(f"Градиент функции: {gradient}")
                print(f"Матрица Гессе: {hessian}")
                print(f"Шаг: {h_k}")

            iteration += 1

        return x


def test_function(x1: float, x2: float) -> float:
    return 2.8 * x2**2 + 1.9 * x1 + 2.7 * x1**2 + 1.6 - 1.9 * x2


def fitness_function(x1: float, x2: float) -> float:
    '''
    Пример целевой функции для оптимизации.

    Параметры:
        x1 (float): Первая переменная.
        x2 (float): Вторая переменная.

    Возвращает:
        float: Значение функции в точке (x1, x2).
    '''
    return x1 ** 2 + sp.exp(x1 ** 2 + x2 ** 2) + 4 * x1 + 3 * x2


if __name__ == '__main__':
    optimizer = RapidGradientDescent(fitness_function, x0=[1, 1], epsilon=0.0001)
    minimum = optimizer.optimize()
    print("Найденный минимум:", minimum)
    print("Значение функции в минимуме:", fitness_function(*minimum))
