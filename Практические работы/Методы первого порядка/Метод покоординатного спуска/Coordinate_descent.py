import math
from tabulate import tabulate
from typing import Callable, List


class CoordinateDescent:
    def __init__(self, fitness_function: Callable[..., float], x0: List[float], epsilon: float, delta: float = 1e-8):
        '''
        Инициализация метода покоординатного спуска для оптимизации функции.

        Параметры:
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
            x0 (List[float]): Начальная точка (вектор).
            epsilon (float): Точность для условия остановки.
            delta (float): Шаг для численного дифференцирования.
        '''
        self.func = fitness_function
        self.n = len(x0)
        self.epsilon = epsilon
        self.delta = delta
        self.arr = [[x0, self.func(*x0)]]

    def _print_table(self) -> None:
        '''
        Вывод текущего состояния в виде таблицы.
        '''
        headers = ["Вершина"] + [f"x{i+1}" for i in range(self.n)] + ["Значение функции"]
        table = [[f"X{i}"] + vertex[0] + [vertex[1]] for i, vertex in enumerate(self.arr)]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def _compute_gradient(self, x: List[float]) -> List[float]:
        '''
        Вычисление градиента функции в точке x с использованием разностных формул.

        Параметры:
            x (List[float]): Точка, в которой вычисляется градиент.

        Возвращает:
            List[float]: Градиент функции в точке x.
        '''
        gradient = []
        for i in range(self.n):
            x_plus = [xj + (self.delta if j == i else 0) for j, xj in enumerate(x)]
            x_minus = [xj - (self.delta if j == i else 0) for j, xj in enumerate(x)]
            partial_derivative = (self.func(*x_plus) - self.func(*x_minus)) / (2 * self.delta)
            gradient.append(partial_derivative)
        return gradient

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

    def _minimize_along_coordinate(self, x: List[float], coord_index: int) -> float:
        '''
        Минимизация функции вдоль одной координаты с использованием метода деления отрезка пополам.

        Параметры:
            x (List[float]): Текущая точка.
            coord_index (int): Индекс координаты, вдоль которой производится минимизация.

        Возвращает:
            float: Новое значение координаты.
        '''
        def f_along_coord(alpha: float) -> float:
            x_new = x.copy()
            x_new[coord_index] = alpha
            return self.func(*x_new)

        # Начальные границы для поиска минимума
        a = x[coord_index] - 1.0  # Левая граница
        b = x[coord_index] + 1.0  # Правая граница
        tol = 1e-8  # Точность для остановки

        # Метод деления отрезка пополам
        while abs(b - a) > tol:
            mid = (a + b) / 2
            derivative = (f_along_coord(mid + self.delta) - f_along_coord(mid - self.delta)) / (2 * self.delta)
            
            if derivative > 0:
                b = mid  # Минимум находится в левой половине
            else:
                a = mid  # Минимум находится в правой половине

        return (a + b) / 2

    def optimize(self, show_info: bool = True):
        '''
        Выполнение оптимизации методом покоординатного спуска.

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

            for i in range(self.n):
                x[i] = self._minimize_along_coordinate(x, i)

            f_new = self.func(*x)
            self.arr.append([x.copy(), f_new])

            if show_info:
                print('\033[92m' + f'Итерация №{iteration}:' + '\033[0m')
                self._print_table()
                print(f"Градиент функции: {gradient}")

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
    return x1 ** 2 + math.exp(x1 ** 2 + x2 ** 2) + 4 * x1 + 3 * x2


if __name__ == '__main__':
    optimizer = CoordinateDescent(fitness_function, x0=[1, 1], epsilon=0.0001)
    minimum = optimizer.optimize()
    print("Найденный минимум:", minimum)
    print("Значение функции в минимуме:", fitness_function(*minimum))
