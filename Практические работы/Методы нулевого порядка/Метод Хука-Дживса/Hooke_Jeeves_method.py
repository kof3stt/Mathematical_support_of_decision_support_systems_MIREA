import math
from tabulate import tabulate
from typing import Callable, List


class HookeJeevesMethod:
    def __init__(self, fitness_function: Callable[..., float], x0: List[float], h: float, d: float, m: float, epsilon: float):
        '''
        Инициализация метода Хука-Дживса для оптимизации функции.

        Параметры:
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
            x0 (List[float]): Начальная точка (вектор).
            h (float): Начальный шаг для исследующего поиска.
            d (float): Коэффициент уменьшения шага.
            m (float): Ускоряющий множитель для поиска по образцу.
            epsilon (float): Точность для условия остановки.
        '''
        self.func = fitness_function
        self.n = len(x0)
        self.epsilon = epsilon
        self.h = h
        self.d = d
        self.m = m
        self.basis_vector = x0  # Базисная точка
        self.step = [h] * self.n  # Шаги по каждой координате
        self.arr = [[x0, self.func(*x0)]]  # Массив для хранения истории точек

    def _print_table(self) -> None:
        '''
        Вывод текущего состояния в виде таблицы.
        '''
        headers = ["Вершина"] + [f"x{i+1}" for i in range(self.n)] + ["Значение функции"]
        table = [[f"X{i}"] + vertex[0] + [vertex[1]] for i, vertex in enumerate(self.arr)]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def _exploratory_search(self, x: List[float]) -> List[float]:
        '''
        Исследующий поиск вокруг точки x.

        Параметры:
            x (List[float]): Текущая точка.

        Возвращает:
            List[float]: Новая точка после исследующего поиска.
        '''
        new_x = x.copy()
        for i in range(self.n):
            tmp_x = new_x.copy()
            tmp_x[i] += self.step[i]
            if self.func(*tmp_x) < self.func(*new_x):
                new_x = tmp_x
            else:
                tmp_x[i] -= 2 * self.step[i]
                if self.func(*tmp_x) < self.func(*new_x):
                    new_x = tmp_x
        return new_x

    def _pattern_search(self, x: List[float], basis_vector: List[float]) -> List[float]:
        '''
        Поиск по образцу.

        Параметры:
            x (List[float]): Текущая точка.
            basis_vector (List[float]): Базисная точка.

        Возвращает:
            List[float]: Новая точка после поиска по образцу.
        '''
        return [x[i] + self.m * (x[i] - basis_vector[i]) for i in range(self.n)]

    def optimize(self, show_iterations: bool = True) -> List[float]:
        '''
        Оптимизация целевой функции с использованием метода Хука-Дживса.

        Параметры:
            show_iterations (bool): Флаг, указывающий, нужно ли выводить информацию о каждой итерации.

        Возвращает:
            List[float]: Координаты точки минимума.
        '''
        iteration = 0
        while True:
            if show_iterations:
                print('\033[92m' + f'Итерация №{iteration}:' + '\033[0m')
                self._print_table()

            # Исследующий поиск вокруг базисной точки
            new_x = self._exploratory_search(self.basis_vector)
            new_value = self.func(*new_x)

            if new_value >= self.func(*self.basis_vector):
                self.step = [step / self.d for step in self.step]
                if all(step < self.epsilon for step in self.step):
                    return self.basis_vector
            else:
                # Поиск по образцу
                pattern_x = self._pattern_search(new_x, self.basis_vector)
                pattern_value = self.func(*pattern_x)

                if pattern_value < new_value:
                    self.basis_vector = pattern_x
                else:
                    self.basis_vector = new_x

                self.arr.append([self.basis_vector, self.func(*self.basis_vector)])

            iteration += 1


# def test_function(x1: float, x2: float) -> float:
#     return 2.8 * x2**2 + 1.9 * x1 + 2.7 * x1**2 + 1.6 - 1.9 * x2


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
    optimizer = HookeJeevesMethod(fitness_function, x0 = [1, 1], h = 0.2, d = 2, m = 2, epsilon = 0.1)
    minimum = optimizer.optimize()
    print("Найденный минимум:", minimum)
    print("Значение функции в минимуме:", fitness_function(*minimum))
