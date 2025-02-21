import math
from tabulate import tabulate
from typing import Callable, List, Tuple, Optional


class SimplexMethod:
    def __init__(self, fitness_function: Callable[..., float], x0: List[float], edge_length: float, epsilon: float):
        '''
        Инициализация симплекс-метода для оптимизации функции.

        Параметры:
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
            x0 (List[float]): Начальная точка (вектор) для построения симплекса.
            edge_length (float): Длина ребра симплекса.
            epsilon (float): Точность для условия остановки.
        '''
        self.func = fitness_function
        self.n = len(x0)
        self.epsilon = epsilon
        self._delta1 = (math.sqrt(self.n + 1) - 1) / (self.n * math.sqrt(2)) * edge_length
        self._delta2 = (math.sqrt(self.n + 1) + self.n - 1) / (self.n * math.sqrt(2)) * edge_length
        self.simplex = self._initialize_simplex(x0)

    def _initialize_simplex(self, x0: List[float]) -> List[List[Tuple[List[float], float]]]:
        '''
        Инициализация симплекса на основе начальной точки.

        Параметры:
            x0 (List[float]): Начальная точка (вектор).

        Возвращает:
            List[List[Tuple[List[float], float]]]: Симплекс, представленный как список вершин,
            где каждая вершина — это кортеж из координат и значения функции.
        '''
        simplex = [x0]
        for i in range(self.n):
            vertex = list(x0)
            for j in range(self.n):
                vertex[j] += self._delta1 if i == j else self._delta2
            simplex.append(vertex)
        return [[point, self.func(*point)] for point in simplex]

    def _centroid(self, k: Optional[int] = None) -> List[float]:
        '''
        Вычисление центра тяжести симплекса.

        Параметры:
            k (Optional[int]): Индекс вершины, которую нужно исключить из вычисления.
                              Если None, вычисляется центр тяжести всех вершин.

        Возвращает:
            List[float]: Координаты центра тяжести.
        '''
        centroid = [0] * self.n
        for index, (vertex, _) in enumerate(self.simplex):
            if k is None or index != k:
                for j in range(self.n):
                    centroid[j] += vertex[j]
        return [x / (self.n + (k is None)) for x in centroid]

    def _reduction(self, r: int) -> None:
        '''
        Редукция симплекса относительно вершины с индексом r.

        Параметры:
            r (int): Индекс вершины, относительно которой выполняется редукция.
        '''
        best_vertex = self.simplex[r][0]
        for i in range(self.n + 1):
            if i != r:
                self.simplex[i][0] = [
                    best_vertex[j] + 0.5 * (self.simplex[i][0][j] - best_vertex[j])
                    for j in range(self.n)
                ]
                self.simplex[i][1] = self.func(*self.simplex[i][0])

    def _print_simplex(self) -> None:
        '''
        Вывод текущего состояния симплекса в виде таблицы.
        '''
        headers = ["Вершина"] + [f"x{i+1}" for i in range(self.n)] + ["Значение функции"]
        table = [[f"X{i}"] + vertex[0] + [vertex[1]] for i, vertex in enumerate(self.simplex)]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def optimize(self, show_iterations: bool = True) -> List[float]:
        '''
        Оптимизация целевой функции с использованием симплекс-метода.

        Параметры:
            show_iterations (bool): Флаг, указывающий, нужно ли выводить информацию о каждой итерации.

        Возвращает:
            List[float]: Координаты точки минимума.
        '''
        iteration = 0
        while True:
            max_vertex = max(self.simplex, key = lambda x: x[1])
            k = self.simplex.index(max_vertex)

            centroid = self._centroid(k)

            reflected_vertex = [2 * c - max_vertex for c, max_vertex in zip(centroid, max_vertex[0])]
            reflected_value = self.func(*reflected_vertex)

            if reflected_value < self.simplex[k][1]:
                self.simplex[k] = [reflected_vertex, reflected_value]
            else:
                min_vertex = min(self.simplex, key = lambda x: x[1])
                r = self.simplex.index(min_vertex)
                self._reduction(r)

            centroid = self._centroid()
            centroid_value = self.func(*centroid)

            if show_iterations:
                print('\033[92m' + f'Итерация №{iteration}:' + '\033[0m')
                self._print_simplex()
                print(f'Центр тяжести симплекса: {[round(x, 4) for x in centroid]}')
                print(f'В полученной точке f(xc) = {round(centroid_value, 4)}')
                print(f'Проверка условия окончания процесса вычислений...')
                for index, vertex in enumerate(self.simplex):
                    res = abs(self.func(*vertex[0]) - centroid_value) < self.epsilon
                    print(f'|f(x{index})-f(xc)| = {round(abs(self.func(*vertex[0]) - centroid_value), 4)}', end = ' ')
                    print('<' if res else '>', f'epsilon ({self.epsilon})')
                
            iteration += 1

            if all(abs(self.func(*vertex[0]) - centroid_value) < self.epsilon for vertex in self.simplex):
                min_vertex = min(self.simplex, key = lambda x: x[1])
                return min_vertex[0]


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


# def test_function(x1: float, x2: float):
#     return 2.8 * x2**2 + 1.9 * x1 + 2.7 * x1**2 + 1.6 - 1.9*x2


if __name__ == '__main__':
    optimizer = SimplexMethod(fitness_function, x0=[1, 1], edge_length=0.5, epsilon=0.0001)
    minimum = optimizer.optimize()
    print("Найденный минимум:", minimum)
    print("Значение функции в минимуме:", fitness_function(*minimum))
