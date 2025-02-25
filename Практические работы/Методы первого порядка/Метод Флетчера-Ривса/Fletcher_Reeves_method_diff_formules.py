import math
from tabulate import tabulate
from typing import Callable, List


class FletcherReeves:
    def __init__(self, fitness_function: Callable[..., float], x0: List[float], epsilon: float, delta: float = 1e-6):
        '''
        Инициализация метода Флетчера-Ривса для оптимизации функции.

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

    def _compute_hessian(self, x: List[float]) -> List[List[float]]:
        '''
        Вычисление матрицы Гессе функции в точке x с использованием разностных формул.

        Параметры:
            x (List[float]): Точка, в которой вычисляется матрица Гессе.

        Возвращает:
            List[List[float]]: Матрица Гессе функции в точке x.
        '''
        hessian = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    # Вторая производная по одной переменной
                    x_plus = [xk + (self.delta if k == i else 0) for k, xk in enumerate(x)]
                    x_minus = [xk - (self.delta if k == i else 0) for k, xk in enumerate(x)]
                    second_derivative = (self.func(*x_plus) - 2 * self.func(*x) + self.func(*x_minus)) / (self.delta ** 2)
                    hessian[i][j] = second_derivative
                else:
                    # Смешанная производная
                    x_plus_plus = [xk + (self.delta if k == i or k == j else 0) for k, xk in enumerate(x)]
                    x_plus_minus = [xk + (self.delta if k == i else 0) - (self.delta if k == j else 0) for k, xk in enumerate(x)]
                    x_minus_plus = [xk - (self.delta if k == i else 0) + (self.delta if k == j else 0) for k, xk in enumerate(x)]
                    x_minus_minus = [xk - (self.delta if k == i or k == j else 0) for k, xk in enumerate(x)]
                    mixed_derivative = (self.func(*x_plus_plus) - self.func(*x_plus_minus) - self.func(*x_minus_plus) + self.func(*x_minus_minus)) / (4 * self.delta ** 2)
                    hessian[i][j] = mixed_derivative
        return hessian

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

    def _compute_step(self, gradient: List[float], direction: List[float], hessian: List[List[float]]) -> float:
        '''
        Вычисление шага h_k с использованием матрицы Гессе.

        Параметры:
            gradient (List[float]): Градиент функции в текущей точке.
            direction (List[float]): Направление спуска.
            hessian (List[List[float]]): Матрица Гессе функции в текущей точке.

        Возвращает:
            float: Оптимальный шаг h_k.
        '''
        numerator = sum(g * d for g, d in zip(gradient, direction))
        denominator = sum(d * sum(h_ij * d_j for h_ij, d_j in zip(h_i, direction)) for h_i, d in zip(hessian, direction))
        return numerator / denominator

    def optimize(self, show_info: bool = True):
        '''
        Выполнение оптимизации методом Флетчера-Ривса.

        Параметры:
            show_info (bool): Флаг для отображения информации о каждой итерации.

        Возвращает:
            List[float]: Точка минимума.
        '''
        x = self.arr[0][0]
        iteration = 0

        gradient = self._compute_gradient(x)
        direction = [-g for g in gradient]

        while True:
            if self._check_stopping_condition(gradient):
                break

            hessian = self._compute_hessian(x)
            h_k = self._compute_step(gradient, direction, hessian)

            x_new = [x[i] - h_k * direction[i] for i in range(self.n)]
            f_new = self.func(*x_new)

            self.arr.append([x_new, f_new])

            gradient_new = self._compute_gradient(x_new)

            beta = sum(g_new * g_new for g_new in gradient_new) / sum(g * g for g in gradient)

            direction = [-g_new + beta * d for g_new, d in zip(gradient_new, direction)]

            x = x_new
            gradient = gradient_new

            if show_info:
                print('\033[92m' + f'Итерация №{iteration}:' + '\033[0m')
                self._print_table()
                print(f"Градиент функции: {gradient}")
                print(f"Направление спуска: {direction}")
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
    return x1 ** 2 + math.exp(x1 ** 2 + x2 ** 2) + 4 * x1 + 3 * x2


if __name__ == '__main__':
    optimizer = FletcherReeves(fitness_function, x0=[1, 1], epsilon=0.0001)
    minimum = optimizer.optimize()
    print("Найденный минимум:", minimum)
    print("Значение функции в минимуме:", fitness_function(*minimum))
