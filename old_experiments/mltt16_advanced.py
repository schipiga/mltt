import gc
import numpy as np
import scipy.linalg as la
import tiktoken


enc = tiktoken.get_encoding('gpt2')
stype = np.float16
ctype = np.float32
END_PUNCTUATION = ('.', '!', '?', ".'", "!'","?'", '."', '!"', '?"')


class Node:

    def __init__(self, char, window_size):
        self.char = char
        self.coords = np.random.uniform(0.1, 0.9, window_size).astype(stype)


class Attn:

    def __init__(self, source_char, target_char, ndim):
        self.source = source_char
        self.target = target_char
        self.coords = np.random.uniform(0.1, 0.9, ndim).astype(stype)


class Edge:

    def __init__(self, source_char, target_char, window_size, ndim, diag_idx, tril_idx):
        self.is_wip = None
        self.source = source_char
        self.target = target_char
        self.window_size = window_size
        self.ndim = ndim
        self.flatten_dim = window_size * ndim
        self.diag_idx = diag_idx
        self.tril_idx = tril_idx
        
        self.coords = np.empty(0, dtype=stype)

        self.threshold = self.flatten_dim // 2 
        self.is_matrix = False
        self.count = 0
        self.capacity = 4

        self.raw_A = np.zeros((self.capacity, self.flatten_dim), dtype=stype)
        self.raw_B = np.zeros((self.capacity, self.window_size), dtype=stype)

        self.packed_dim = (self.flatten_dim * (self.flatten_dim + 1)) // 2
        self.S_AA = np.empty(0, dtype=stype)
        self.S_AB = np.empty((0, 0), dtype=stype)

    def add_case(self, attention, target_coords, alpha=1.0):
        attn = attention.astype(ctype)
        tgt = target_coords.astype(ctype)

        if not self.is_matrix:
            if self.count >= self.capacity:
                new_capacity = min(self.capacity * 2, self.threshold)
                
                new_A = np.zeros((new_capacity, self.flatten_dim), dtype=stype)
                new_A[:self.capacity] = self.raw_A
                self.raw_A = new_A
                
                new_B = np.zeros((new_capacity, self.window_size), dtype=stype)
                new_B[:self.capacity] = self.raw_B
                self.raw_B = new_B
                
                self.capacity = new_capacity

            if alpha < 1.0:
                decay = ctype(np.sqrt(alpha))
                self.raw_A[:self.count] = (self.raw_A[:self.count].astype(ctype) * decay).astype(stype)
                self.raw_B[:self.count] = (self.raw_B[:self.count].astype(ctype) * decay).astype(stype)

            self.raw_A[self.count] = attention.astype(stype)
            self.raw_B[self.count] = target_coords.astype(stype)
            self.count += 1

            if self.count >= self.threshold:
                self._evolve()
        else:
            S_AA_comp = self.S_AA.astype(ctype)
            S_AB_comp = self.S_AB.astype(ctype)

            if alpha < 1.0:
                S_AA_comp *= ctype(alpha)
                S_AB_comp *= ctype(alpha)

            S_AA_comp += attn[self.tril_idx[0]] * attn[self.tril_idx[1]]
            S_AB_comp += np.outer(attn, tgt)

            self.S_AA = S_AA_comp.astype(stype)
            self.S_AB = S_AB_comp.astype(stype)

        self.is_wip = True

    def _evolve(self):
        A = self.raw_A.astype(ctype)
        B = self.raw_B.astype(ctype)

        S_AA_dense = A.T @ A
        self.S_AA = S_AA_dense[self.tril_idx].astype(stype)
        self.S_AB = (A.T @ B).astype(stype)

        self.raw_A = np.empty(0, dtype=stype)
        self.raw_B = np.empty((0, 0), dtype=stype)
        self.is_matrix = True

    def solve(self):
        if self.is_wip:
            if self.is_matrix:
                S_AA_dense = np.zeros((self.flatten_dim, self.flatten_dim), dtype=ctype)
                S_AA_dense[self.tril_idx] = self.S_AA.astype(ctype)
                S_AB_comp = self.S_AB.astype(ctype)
            else:
                A = self.raw_A[:self.count].astype(ctype)
                B = self.raw_B[:self.count].astype(ctype)
                S_AA_dense = A.T @ A
                S_AB_comp = A.T @ B

            S_AA_dense[self.diag_idx] += 1e-4
            
            try:
                raw_coords = la.solve(S_AA_dense, S_AB_comp, assume_a='pos', lower=True)
            except:
                import ipdb; ipdb.set_trace()
            self.coords = raw_coords.astype(stype)
            self.is_wip = False
            
    def clear_buffers(self):
        self.raw_A = np.empty((0, 0), dtype=stype)
        self.raw_B = np.empty((0, 0), dtype=stype)
        self.S_AA = np.empty(0, dtype=stype)
        self.S_AB = np.empty((0, 0), dtype=stype)


class MLTT:

    def __init__(self, ndim=16, window_size=16):
        self.ndim = ndim
        self.window_size = window_size
        self.tril_idx = np.tril_indices(window_size * ndim)
        self.diag_idx = np.diag_indices(window_size * ndim)
        self.pos_matrix = np.random.uniform(0.1, 0.9, (window_size, ndim)).astype(stype)

        self.nodes = {}
        self.attns = {}
        self.edges = {}
        self.edges_by_src = {}

        self._attn_buffer = np.zeros((window_size, ndim), dtype=stype)

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

        return self._attn_buffer.ravel()

    def train(self, text, alpha=1.0, skip_end=False):
        tokens = enc.encode(text)

        if not skip_end and any(text.endswith(punct) for punct in END_PUNCTUATION):
            tokens += [enc.eot_token]
        
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
            edge = Edge(curr_token, next_token, self.window_size, self.ndim, self.diag_idx, self.tril_idx)
            self.edges[edge_key] = edge

            if curr_token not in self.edges_by_src:
                self.edges_by_src[curr_token] = []
            self.edges_by_src[curr_token].append(edge)

        coords = self._get_node(next_token).coords
        edge = self.edges[edge_key]

        edge.add_case(attention, coords, alpha)

    def solve_debug(self):
        i = 0
        for edge in self.edges.values():
            edge.solve()
            edge.clear_buffers()

            i += 1
            if i % 100 == 0:
                gc.collect()
                print(f"Solved {i} edges...")

    def solve(self):
        for edge in self.edges.values():
            edge.solve()

    def release(self):
        for edge in self.edges.values():
            edge.clear_buffers()

        self.train = None
        self.train_step = None

        gc.collect()

    def generate(self, seed_text, length=None, temperature=0.0):
        length = length or self.window_size

        result_tokens = enc.encode(seed_text)

        for _ in range(length):
            start_ctx = max(0, len(result_tokens) - (self.window_size - 1))
            context = result_tokens[start_ctx:]

            last_token = context[-1]
            attention = self._matrix_attention(context, last_token)

            valid_edges = self.edges_by_src.get(last_token, [])

            if not valid_edges:
                break

            edges_matrices = np.array([edge.coords for edge in valid_edges], dtype=ctype)
            targets_vectors = np.array([self._get_node(edge.target).coords for edge in valid_edges], dtype=ctype)
            attn_comp = attention.astype(ctype)

            predicted_vectors = np.einsum('i,kij->kj', attn_comp, edges_matrices)
            errors = np.linalg.norm(predicted_vectors - targets_vectors, axis=1)

            if temperature <= 1e-5: # NOTE: "Safe zero" threshold to prevent numerical issues
                best_idx = np.argmin(errors)
            else:
                logits = -errors / temperature
                logits -= np.max(logits)
                
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits)

                best_idx = np.random.choice(len(valid_edges), p=probs)

            best_token = valid_edges[best_idx].target

            if best_token == enc.eot_token:
                break

            result_tokens.append(best_token)

        return enc.decode(result_tokens)

    def learn(self, text, length=512):
        self.train(text)
        self.solve()

        return self.generate(text, length=length)
