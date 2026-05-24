import numpy as np
import tiktoken


enc = tiktoken.get_encoding('gpt2')
stype = np.float16
ctype = np.float32
END_PUNCTUATION = ('.', '!', '?', ".'", "!'","?'", '."', '!"', '?"')


class Node:

    def __init__(self, char):
        self.char = char


class Attn:

    def __init__(self, source_char, target_char, ndim):
        self.source = source_char
        self.target = target_char
        self.coords = np.random.uniform(0.1, 0.9, ndim).astype(stype)


class Edge:
    def __init__(self, source_char, target_char, window_size, ndim):
        self.source = source_char
        self.target = target_char
        self.window_size = window_size
        self.ndim = ndim
        self.flatten_dim = window_size * ndim

        self.coords_list = []
        self.coords_matrix = None
        self.is_matrix = False

    def add_case(self, attention):
        if not self.is_matrix:
            self.coords_list.append(attention)

            if len(self.coords_list) >= self.ndim:
                self.coords_matrix = np.column_stack(self.coords_list)
                self.coords_list = None
                self.is_matrix = True
        else:
            attn = attention.astype(ctype)
            current_slots = self.coords_matrix.astype(ctype)
            temp_matrix = np.column_stack((current_slots, attn))

            U, S, _ = np.linalg.svd(temp_matrix, full_matrices=False)

            U_reduced = U[:, :self.ndim]
            S_reduced = S[:self.ndim]

            self.coords_matrix = (U_reduced @ np.diag(np.sqrt(S_reduced))).astype(stype)


class MLTT:

    def __init__(self, ndim=32, window_size=16):
        self.ndim = ndim
        self.window_size = window_size
        self.pos_matrix = np.random.uniform(0.1, 0.9, (window_size, ndim)).astype(stype)

        self.nodes = {}
        self.attns = {}
        self.edges = {}
        self.edges_by_src = {}

        self._attn_buffer = np.zeros((window_size, ndim), dtype=stype)

    def _get_node(self, token_id):
        if token_id not in self.nodes:
            self.nodes[token_id] = Node(token_id)
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

        return self._attn_buffer.ravel()

    def train(self, text, skip_end=False):
        tokens = enc.encode(text)

        if not skip_end and any(text.endswith(punct) for punct in END_PUNCTUATION):
            tokens += [enc.eot_token]
        
        for i in range(1, len(tokens)):
            start_ctx = max(0, i - (self.window_size - 1))
            context = tokens[start_ctx:i]

            curr_token = context[-1]
            next_token = tokens[i]

            self.train_step(context, next_token)

    def train_step(self, context, next_token):
        if not context:
            return

        curr_token = context[-1]
        attention = self._matrix_attention(context, curr_token)

        edge_key = (curr_token, next_token)
        if edge_key not in self.edges:
            edge = Edge(curr_token, next_token, self.window_size, self.ndim)
            self.edges[edge_key] = edge

            if curr_token not in self.edges_by_src:
                self.edges_by_src[curr_token] = []
            self.edges_by_src[curr_token].append(edge)

        self.edges[edge_key].add_case(attention)

    def release(self):
        self.train = None
        self.train_step = None

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

            attention = self._matrix_attention(context, last_token).astype(ctype)
            attn_norm = attention / (np.linalg.norm(attention) + 1e-8)

            edge_scores = []

            for edge in valid_edges:
                if not edge.is_compressed:
                    coords = np.column_stack(edge.coords_list).astype(ctype)
                else:
                    coords = edge.coords_matrix.astype(ctype)

                slot_norms = np.linalg.norm(coords, axis=0)
                slot_norms[slot_norms == 0] = 1e-8

                normalized_coords = coords / slot_norms
                cos_sims = attn_norm @ normalized_coords 

                edge_scores.append(np.max(cos_sims))

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
