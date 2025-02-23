import math
from tabulate import tabulate
from typing import Callable, List, Tuple, Optional


class NelderMeadMethod:
    def __init__(self, fitness_function: Callable[..., float], x0: List[float], edge_length: float, beta: float, gamma: float, epsilon: float):
        '''
        Инициализация метода Нелдера-Мида для оптимизации функции.

        Параметры:
            fitness_function (Callable[..., float]): Целевая функция для оптимизации.
            x0 (List[float]): Начальная точка (вектор) для построения многогранника.
            edge_length (float): Длина ребра многогранника.
            beta (float): Параметр растяжения.
            gamma (float): Параметр сжатия.
            epsilon (float): Точность для условия остановки.
        '''
        self.func = fitness_function
        self.n = len(x0)
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
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
        Оптимизация целевой функции с использованием метода Нелдера-Мида.

        Параметры:
            show_iterations (bool): Флаг, указывающий, нужно ли выводить информацию о каждой итерации.

        Возвращает:
            List[float]: Координаты точки минимума.
        '''
        iteration = 0  # Счетчик итераций
        while True:  # Основной цикл оптимизации
            # Сортировка вершин симплекса по значению функции (от наименьшего к наибольшему)
            self.simplex.sort(key=lambda x: x[1])
            # fl - значение функции в наилучшей вершине (с наименьшим значением)
            # fs - значение функции в следующей за наихудшей вершине
            # fh - значение функции в наихудшей вершине (с наибольшим значением)
            fl, fs, fh = self.simplex[0][1], self.simplex[-2][1], self.simplex[-1][1] # Пункт 3

            # Если включен вывод итераций, выводим текущее состояние симплекса
            if show_iterations:
                print('\033[92m' + f'Итерация №{iteration}:' + '\033[0m')
                self._print_simplex()
            
            # Вычисляем центр тяжести всех вершин, кроме наихудшей (вершины с наибольшим значением функции)
            centroid = self._centroid(len(self.simplex) - 1) # Пункт 4
            
            # Отражение наихудшей вершины относительно центра тяжести
            reflected = [2 * c - x for c, x in zip(centroid, self.simplex[-1][0])] # Пункт 5
            reflected_value = self.func(*reflected)  # Значение функции в отраженной точке
            
            # Проверка условий для отражения, растяжения или сжатия
            if reflected_value < fl:  # Если отраженная точка лучше наилучшей
                # Растяжение: пытаемся улучшить отраженную точку
                stretched = [centroid[j] + self.beta * (reflected[j] - centroid[j]) for j in range(self.n)]
                stretched_value = self.func(*stretched)  # Значение функции в растянутой точке
                if stretched_value < reflected_value:  # Если растяжение улучшило значение
                    self.simplex[-1] = [stretched, stretched_value]  # Заменяем наихудшую вершину на растянутую
                else:
                    # Если растяжение не улучшило значение, заменяем наихудшую вершину на отраженную
                    self.simplex[-1] = [reflected, reflected_value]
                    # Выполняем редукцию (уменьшение симплекса) относительно наилучшей вершины
                    min_index = min(range(len(self.simplex)), key=lambda i: self.simplex[i][1])
                    self._reduction(min_index)
            elif reflected_value < fs:  # Если отраженная точка лучше второй наихудшей
                # Заменяем наихудшую вершину на отраженную
                self.simplex[-1] = [reflected, reflected_value]
            else:  # Если отраженная точка не улучшила значение
                # Сжатие: пытаемся улучшить наихудшую вершину
                compressed = [centroid[j] + self.gamma * (self.simplex[-1][0][j] - centroid[j]) for j in range(self.n)]
                compressed_value = self.func(*compressed)  # Значение функции в сжатой точке
                if compressed_value < self.simplex[-1][1]:  # Если сжатие улучшило значение
                    self.simplex[-1] = [compressed, compressed_value]  # Заменяем наихудшую вершину на сжатую
                else:
                    # Если сжатие не улучшило значение, выполняем редукцию (уменьшение симплекса)
                    self._reduction(0)
            
            # Вычисляем центр тяжести всего симплекса
            total_centroid = self._centroid()
            # Вычисляем критерий остановки (сигма) как среднеквадратичное отклонение значений функции от центра тяжести
            sigma = math.sqrt(sum((vertex[1] - self.func(*total_centroid)) ** 2 for vertex in self.simplex) / (self.n + 1))

            if show_iterations:
                print(f'xc = {total_centroid}')
                print(f'f(xc) = {self.func(*total_centroid)}')
                print(f'sigma = {sigma}')

            if sigma < self.epsilon:  # Если критерий остановки выполнен
                return self.simplex[0][0]  # Возвращаем координаты наилучшей вершины
            
            iteration += 1  # Увеличиваем счетчик итераций


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
    optimizer = NelderMeadMethod(fitness_function, x0=[1, 1], edge_length=0.75, beta=1.85, gamma=0.1, epsilon=0.0001)
    minimum = optimizer.optimize()
    print("Найденный минимум:", minimum)
    print("Значение функции в минимуме:", fitness_function(*minimum))
