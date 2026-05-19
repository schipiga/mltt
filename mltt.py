import numpy as np

class Node:
    def __init__(self, char, ndim):
        self.char = char
        # N-мерные координаты: случайный вектор в диапазоне [0.1, 1.0]
        self.coords = np.random.uniform(0.1, 1.0, ndim)
        # Сумма всех N координат (наша целевая константа для уравнений)
        self.coord_sum = np.sum(self.coords)


class Edge:
    def __init__(self, source_char, target_char, ndim, max_memory=None):
        self.source = source_char
        self.target = target_char
        self.ndim = ndim
        self.max_memory = max_memory if max_memory else ndim
        
        # Начальные координаты ребра (N-мерный вектор)
        self.coords = np.random.uniform(0.1, 1.0, ndim)
        
        # Матрицы для хранения истории (уравнений)
        self.A = None  # Матрица векторов внимания (коэффициенты)
        self.B = None  # Вектор целевых сумм (правая часть уравнения)

    def update_equation(self, a_vec, target_sum):
        """Добавляет новый контекст и пересчитывает координаты ребра."""
        
        if self.A is None:
            # Первое появление связи
            self.A = np.atleast_2d(a_vec)
            self.B = np.array([target_sum])
        else:
            # Если память переполнена, "схлопываем" старый опыт с новым
            if len(self.A) >= self.max_memory:
                self.A[0] = (self.A[0] + a_vec) / 2
                self.B[0] = (self.B[0] + target_sum) / 2
            else:
                # Иначе просто добавляем новое уравнение в систему
                self.A = np.vstack([self.A, a_vec])
                self.B = np.append(self.B, target_sum)
        
        # Находим компромиссные координаты через метод наименьших квадратов
        # np.linalg.lstsq идеально решает систему любой формы (даже переопределенную)
        X, _, _, _ = np.linalg.lstsq(self.A, self.B, rcond=None)
        
        # Не даем координатам уйти в отрицательные значения
        self.coords = np.clip(X, 0.01, None)


class MLTT:
    def __init__(self, alphabet, ndim=10, window_size=7):
        self.ndim = ndim
        self.window_size = window_size
        self.alphabet = list(alphabet)
        
        # Инициализируем объекты Node
        self.nodes = {char: Node(char, ndim) for char in self.alphabet}
        # Словарь объектов Edge: ключ (source_char, target_char)
        self.edges = {}

    def _calculate_attention(self, context_chars):
        """Считает N-мерный вектор внимания."""
        attn = np.zeros(self.ndim)
        ctx_len = len(context_chars)

        for pos, char in enumerate(context_chars, start=1):
            weight = pos / self.window_size
            attn += self.nodes[char].coords * weight

        max_possible_weight = sum((i / self.window_size) for i in range(1, ctx_len + 1))
        if max_possible_weight > 0:
            attn /= max_possible_weight

        return attn

    def train(self, text):
        # Очищаем текст от неизвестных символов
        valid_chars = [c for c in text.lower() if c in self.nodes]

        for i in range(1, len(valid_chars)):
            context = valid_chars[max(0, i - self.window_size) : i]
            target_char = valid_chars[i]
            last_char = context[-1]

            edge_key = (last_char, target_char)

            if edge_key not in self.edges:
                self.edges[edge_key] = Edge(last_char, target_char, self.ndim)

            a_vec = self._calculate_attention(context)
            target_sum = self.nodes[target_char].coord_sum

            # Делегируем обновление самому ребру
            self.edges[edge_key].update_equation(a_vec, target_sum)

    def generate(self, seed_text, length=50):
        result = [c for c in seed_text.lower() if c in self.nodes]
        if not result:
            return ""

        for _ in range(length):
            context = result[-self.window_size:]
            last_char = context[-1]

            a_vec = self._calculate_attention(context)
            
            best_char = ' '
            min_error = float('inf')

            # Сканируем ребра, выходящие из last_char
            for (src, tgt), edge in self.edges.items():
                if src != last_char:
                    continue

                # Предсказанная сумма = Внимание * Координаты ребра
                predicted_sum = np.dot(a_vec, edge.coords)
                actual_sum = self.nodes[tgt].coord_sum

                error = abs(predicted_sum - actual_sum)

                if error < min_error:
                    min_error = error
                    best_char = tgt

            result.append(best_char)

        return "".join(result)
