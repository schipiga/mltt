import numpy as np
from scipy.optimize import lsq_linear


class Node:

    def __init__(self, char, ndim):
        self.char = char
        self.coords = np.random.uniform(0.1, 0.9, ndim)
        self.coord_sum = np.sum(self.coords)


class Attn:

    def __init__(self, source_char, target_char, ndim):
        self.source = source_char
        self.target = target_char
        self.ndim = ndim
        self.coords = np.random.uniform(0.1, 0.9, ndim)


class Edge:

    def __init__(self, source_char, target_char, ndim):
        self.source = source_char
        self.target = target_char
        self.ndim = ndim
        self.coords = np.random.uniform(0.1, 0.9, ndim)
        self.A = np.empty((0, ndim))
        self.B = np.empty((0,))

    def update_equation(self, attention, coord_sum):
        if len(self.A) > 0:
            exact_match = np.all(self.A == attention, axis=1)

            if np.any(exact_match):
                return

        if len(self.A) < self.ndim:
            self.A = np.vstack([self.A, attention])
            self.B = np.append(self.B, coord_sum)
        else:
            norms_A = np.linalg.norm(self.A, axis=1, keepdims=True)
            norm_new = np.linalg.norm(attention)

            sim_with_new = np.dot(self.A, attention) / (norms_A.flatten() * norm_new)
            closest_idx = np.argmax(sim_with_new)

            self.A[closest_idx] = (self.A[closest_idx] + attention) / 2

        self.coords = lsq_linear(self.A, self.B).x


class MLTT:

    def __init__(self, alphabet, ndim=16, window_size=16):
        self.ndim = ndim
        self.window_size = window_size
        self.alphabet = list(alphabet)
        self.pos_matrix = np.random.uniform(0.1, 0.9, (window_size, ndim))
        self.nodes = {char: Node(char, ndim) for char in self.alphabet}
        self.edges = {}
        self.attns = {}

        for src in self.alphabet:
            for tgt in self.alphabet:
                self.attns[(src, tgt)] = Attn(src, tgt, ndim)

    def _matrix_attention(self, context_chars, focus_char):
        """Works."""
        ctx_len = len(context_chars)

        context_coords = np.array([self.nodes[c].coords for c in context_chars])
        attn_coords = np.array([self.attns[(c, focus_char)].coords for c in context_chars])

        current_pos_matrix = self.pos_matrix[:ctx_len]

        return np.sum(attn_coords * current_pos_matrix, axis=0)

    def train(self, text):
        """Works."""
        for i in range(1, len(text)):
            start_ctx = max(0, i - (self.window_size - 1))
            context = text[start_ctx:i]

            curr_char = context[-1]
            next_char = text[i]

            attention = self._matrix_attention(context, curr_char)

            edge_key = (curr_char, next_char)
            if edge_key not in self.edges:
                self.edges[edge_key] = Edge(curr_char, next_char, self.ndim)

            coord_sum = self.nodes[next_char].coord_sum
            self.edges[edge_key].update_equation(attention, coord_sum)

    def generate(self, seed_text, length=30):
        result = [c for c in seed_text]

        for _ in range(length):
            start_ctx = max(0, len(result) - (self.window_size - 1))
            context = result[start_ctx:]

            last_char = context[-1]
            attention = self._matrix_attention(context, last_char)

            valid_edges = [edge for (src, tgt), edge in self.edges.items() if src == last_char]

            if not valid_edges:
                return "".join(result)

            edges_A = np.array([edge.coords for edge in valid_edges])
            targets_B = np.array([self.nodes[edge.target].coord_sum for edge in valid_edges])

            predicted_sums = np.dot(edges_A, attention)
            errors = np.abs(predicted_sums - targets_B)

            best_idx = np.argmin(errors)
            best_char = valid_edges[best_idx].target

            result.append(best_char)

        return "".join(result)


data = 'deep learning - it is architecture' # need debug

data = "deep learning architectures are amazing. machine language text transformer expands dimensions. in higher dimensions we can cross multiple hyperplanes to find perfect edge coordinates. the memory of this model scales with numpy matrix operations. hello world, welcome to n-dimensional geometry. hello zlo, what do you t hink about twenty dimensions?"

i = 0
for line in lines:
    model.train(line)
    i = i + 1
    if i % 100 == 0:
        print(f"Trained on {i} lines...")
