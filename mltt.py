import numpy as np


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
        self.S_AA = np.zeros((ndim, ndim))
        self.S_AB = np.zeros(ndim)
        self.S_AA += np.eye(ndim) * 1e-4

    def add_case(self, attention, coord_sum, alpha=1.0):
        if alpha < 1.0:
            self.S_AA *= alpha
            self.S_AB *= alpha

        self.S_AA += np.outer(attention, attention)
        self.S_AB += attention * coord_sum

    def solve(self):
        self.coords = np.linalg.solve(self.S_AA, self.S_AB)


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
        ctx_len = len(context_chars)

        context_coords = np.array([self.nodes[c].coords for c in context_chars])
        attn_coords = np.array([self.attns[(c, focus_char)].coords for c in context_chars])

        current_pos_matrix = self.pos_matrix[:ctx_len]

        return np.sum(attn_coords * current_pos_matrix, axis=0)

    def train(self, text, alpha=1.0):
        for i in range(1, len(text)):
            start_ctx = max(0, i - (self.window_size - 1))
            context = text[start_ctx:i]

            curr_char = context[-1]
            next_char = text[i]

            self.train_step(context, next_char, alpha)

    def train_step(self, context, next_char, alpha=1.0):
        if not context:
            return

        curr_char = context[-1]
        attention = self._matrix_attention(context, curr_char)

        edge_key = (curr_char, next_char)
        if edge_key not in self.edges:
            self.edges[edge_key] = Edge(curr_char, next_char, self.ndim)

        coord_sum = self.nodes[next_char].coord_sum

        self.edges[edge_key].add_case(attention, coord_sum, alpha)
        self.edges[edge_key].solve()

    def generate(self, seed_text, length=None):
        length = length or self.window_size
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
