import numpy as np
import tiktoken


enc = tiktoken.get_encoding('gpt2')
END_PUNCTUATION = ('.', '!', '?', ".'", "!'","?'", '."', '!"', '?"')


class Node:

    def __init__(self, char):
        self.char = char


class Attn:

    def __init__(self, source_char, target_char, ndim, dtype):
        self.source = source_char
        self.target = target_char
        self.coords = np.random.uniform(0.1, 0.9, ndim).astype(dtype)


class Edge:
    def __init__(self, source_char, target_char, window_size, ndim, eps):
        self.source = source_char
        self.target = target_char
        self.window_size = window_size
        self.ndim = ndim
        self.flatten_dim = window_size * ndim
        self.eps = eps

        self.coords_list = []
        self.coords_matrix = None
        self.normalized_matrix = None

    def add_case(self, attention):
        if self.coords_matrix is None:
            self.coords_list.append(attention)

            if len(self.coords_list) >= self.ndim:
                self.coords_matrix = np.array(self.coords_list).T
                self.coords_list = None
        else:
            current_slots = self.coords_matrix
            temp_matrix = np.column_stack((current_slots, attention))

            U, S, _ = np.linalg.svd(temp_matrix, full_matrices=False)

            U_reduced = U[:, :self.ndim]
            S_reduced = S[:self.ndim]

            self.coords_matrix = (U_reduced * np.sqrt(S_reduced))

    def release(self):
        if self.coords_matrix is None:
            coords = np.array(self.coords_list).T
        else:
            coords = self.coords_matrix

        slot_norms = np.linalg.norm(coords, axis=0)
        slot_norms[slot_norms == 0] = self.eps

        self.normalized_matrix = (coords / slot_norms)
        self.coords_list = None
        self.coords_matrix = None

class MLTT:

    def __init__(self, ndim=32, window_size=16, is_32bit=False):
        self.ndim = ndim
        self.window_size = window_size

        if is_32bit:
            self.dtype = np.float32
            self.eps = 1e-8
        else:
            self.dtype = np.float64
            self.eps = 1e-12

        self.pos_matrix = np.random.uniform(0.1, 0.9, (window_size, ndim)).astype(self.dtype)

        self.nodes = {}
        self.attns = {}
        self.edges = {}
        self.edges_by_src = {}

        self._attn_buffer = np.zeros((window_size, ndim), dtype=self.dtype)

    def _get_node(self, token_id):
        if token_id not in self.nodes:
            self.nodes[token_id] = Node(token_id)
        return self.nodes[token_id]

    def _get_attn(self, src_token, tgt_token):
        key = (src_token, tgt_token)
        if key not in self.attns:
            self.attns[key] = Attn(src_token, tgt_token, self.ndim, self.dtype)
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

    def train(self, text, skip_end=False, log=False):
        tokens = enc.encode(text)

        if not skip_end and any(text.endswith(punct) for punct in END_PUNCTUATION):
            tokens += [enc.eot_token]

        if log:
            print(f"Training on {len(tokens)} tokens...")

        for i in range(self.window_size, len(tokens)):
            start_ctx = i - self.window_size
            context = tokens[start_ctx:i]
            next_token = tokens[i]

            self.train_step(context, next_token)

            if log and i % 100 == 0:
                print(f"Trained on {i} tokens")

    def train_step(self, context, next_token):
        if not context:
            return

        curr_token = context[-1]
        attention = self._matrix_attention(context, curr_token)

        edge_key = (curr_token, next_token)
        if edge_key not in self.edges:
            edge = Edge(curr_token, next_token, self.window_size, self.ndim, self.eps)
            self.edges[edge_key] = edge

            if curr_token not in self.edges_by_src:
                self.edges_by_src[curr_token] = []
            self.edges_by_src[curr_token].append(edge)

        self.edges[edge_key].add_case(attention)

    def release(self, log=False):
        self.train = None
        self.train_step = None

        if log:
            print(f"Releasing {len(self.edges.values())} edges...")
            i = 0

        for edge in self.edges.values():
            edge.release()
            if log:
                i += 1
                if i % 100 == 0:
                    print(f"Released {i} edges")

    def generate(self, seed_text, length=100, temperature=1.0):
        length = length or self.window_size
        result_tokens = enc.encode(seed_text)

        for _ in range(length):
            start_ctx = max(0, len(result_tokens) - (self.window_size - 1))
            context = result_tokens[start_ctx:]
            last_token = context[-1]

            valid_edges = self.edges_by_src.get(last_token, [])
            if not valid_edges:
                break

            attention = self._matrix_attention(context, last_token)
            attn_norm = attention / (np.linalg.norm(attention) + self.eps)

            edge_scores = []

            for edge in valid_edges:
                if edge.normalized_matrix is None:
                    if edge.coords_matrix is None:
                        coords = np.array(edge.coords_list).T
                    else:
                        coords = edge.coords_matrix

                    slot_norms = np.linalg.norm(coords, axis=0)
                    slot_norms[slot_norms == 0] = self.eps

                    normalized_coords = coords / slot_norms
                else:
                    normalized_coords = edge.normalized_matrix

                cos_sims = attn_norm @ normalized_coords

                edge_scores.append(np.linalg.norm(cos_sims))

            edge_scores = np.array(edge_scores)

            if temperature <= 1e-3:
                best_edge_idx = np.argmax(edge_scores)
            else:
                shifted_scores = edge_scores - np.max(edge_scores)
                scaled_scores = shifted_scores / temperature
                exp_scores = np.exp(scaled_scores)
                probs = exp_scores / np.sum(exp_scores)
                
                best_edge_idx = np.random.choice(len(valid_edges), p=probs)

            next_token = valid_edges[best_edge_idx].target

            if next_token == enc.eot_token:
                break

            result_tokens.append(next_token)

        return enc.decode(result_tokens)
