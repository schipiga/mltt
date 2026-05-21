import numpy as np
import tiktoken


enc = tiktoken.get_encoding('gpt2')


class Node:

    def __init__(self, char, window_size):
        self.char = char
        self.coords = np.random.uniform(0.1, 0.9, window_size)


class Attn:

    def __init__(self, source_char, target_char, ndim):
        self.source = source_char
        self.target = target_char
        self.coords = np.random.uniform(0.1, 0.9, ndim)


class Edge:

    def __init__(self, source_char, target_char, window_size, ndim):
        self.source = source_char
        self.target = target_char
        self.window_size = window_size
        self.ndim = ndim
        self.flatten_dim = window_size * ndim
        self.coords = np.zeros(self.flatten_dim)
        self.S_AA = np.eye(self.flatten_dim) * 1e-4 # NOTE: Ridge regression (Tikhonov regularization)
        self.S_AB = np.zeros((self.flatten_dim, self.window_size))

    def add_case(self, attention, target_coords, alpha=1.0):
        if alpha < 1.0: # NOTE: Forgetting factor (exponential decay)
            self.S_AA *= alpha
            self.S_AB *= alpha

        self.S_AA += np.outer(attention, attention)
        self.S_AB += np.outer(attention, target_coords)

    def solve(self):
        self.coords = np.linalg.solve(self.S_AA, self.S_AB) # NOTE: Linear matrix "magic" to get optimal weights


class MLTT:

    def __init__(self, ndim=16, window_size=16):
        self.ndim = ndim
        self.window_size = window_size
        self.pos_matrix = np.random.uniform(0.1, 0.9, (window_size, ndim))

        self.nodes = {}
        self.edges = {}
        self.attns = {}

        self._attn_buffer = np.zeros((window_size, ndim))

    def _get_node(self, token_id):
        if token_id not in self.nodes:
            self.nodes[token_id] = Node(token_id, self.window_size)
        return self.nodes[token_id]

    def _get_attn(self, src_token, tgt_token):
        key = (src_token, tgt_token)
        if key not in self.attns:
            self.attns[key] = Attn(src_token, tgt_token, self.ndim)
        return self.attns[key]

    def _matrix_attention(self, context_tokens, focus_token):
        ctx_len = len(context_tokens)
        self._attn_buffer[:ctx_len] = 0.0 

        for idx, t in enumerate(context_tokens):
            self._attn_buffer[idx] = self._get_attn(t, focus_token).coords

        self._attn_buffer[:ctx_len] *= self.pos_matrix[:ctx_len]

        if ctx_len < self.window_size:
            self._attn_buffer[ctx_len:] = 0.0

        return self._attn_buffer.flatten()

    def train(self, text, alpha=1.0):
        tokens = enc.encode(text)
        
        for i in range(1, len(tokens)):
            start_ctx = max(0, i - (self.window_size - 1))
            context = tokens[start_ctx:i]

            curr_token = context[-1]
            next_token = tokens[i]

            self.train_step(context, next_token, alpha)

    def train_step(self, context, next_token, alpha=1.0):
        if not context:
            return

        curr_token = context[-1]
        attention = self._matrix_attention(context, curr_token)

        edge_key = (curr_token, next_token)
        if edge_key not in self.edges:
            self.edges[edge_key] = Edge(curr_token, next_token, self.window_size, self.ndim)

        coords = self._get_node(next_token).coords

        self.edges[edge_key].add_case(attention, coords, alpha)

    def solve(self):
        i = 0
        for edge in self.edges.values():
            edge.solve()
            i += 1
            if i % 100 == 0:
                print(f"Solved {i} edges...")

    def generate(self, seed_text, length=None):
        length = length or self.window_size

        result_tokens = enc.encode(seed_text)

        for _ in range(length):
            start_ctx = max(0, len(result_tokens) - (self.window_size - 1))
            context = result_tokens[start_ctx:]

            last_token = context[-1]
            attention = self._matrix_attention(context, last_token)

            valid_edges = [edge for (src, tgt), edge in self.edges.items() if src == last_token]

            if not valid_edges:
                break

            edges_matrices = np.array([edge.coords for edge in valid_edges])

            targets_vectors = np.array([self._get_node(edge.target).coords for edge in valid_edges])

            predicted_vectors = np.einsum('i,kij->kj', attention, edges_matrices)
            errors = np.linalg.norm(predicted_vectors - targets_vectors, axis=1)

            best_idx = np.argmin(errors)
            best_token = valid_edges[best_idx].target

            result_tokens.append(best_token)

        return enc.decode(result_tokens)