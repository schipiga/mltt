# MLTT: Matrix Language Text Transformer

**R&D** project exploring alternative paradigms for training text generators on unstructured corpora (similar to datasets used for ChatGPT). The core mission is to bypass the engineering complexities of backpropagation and explore a fundamentally different architectural foundation.

---

### Foundations

*   **Next-Token Prediction** is **good**: The objective of anticipating the subsequent token remains the most effective way to capture text structure.
*   **Attention Mechanism** is **good**: Relational weighting allows the model to mix history into a single vector.
*   **Markovian Chain** is **good**: Graph-based state transitions provide a clean, trace-able trajectory for text flow.

---

### How It Works

Instead of deep neural layers, weight matrices, and iterative gradient descent (backpropagation), MLTT relies entirely on an explicit **dependency graph** and analytical **SLAE (System of Linear Algebraic Equations)** optimization.

#### 1. Context Routing via SLAE
Suppose we have multiple distinct context paths ($X$ variants) that pass through a node transition $M \to N$. To ensure that each specific context path leads from $M$ straight to the correct target node $N$, we construct a System of Linear Equations:

*   **Matrix $A$**: Attentions computed for each unique history context leading into node $M$.
*   **Vector $B$**: Unique coordinate identifiers representing target node $N$.

The model sets up $X$ equations with $X$ parameters to find the exact geometric hyperplane configuration required to route each trajectory cleanly. What contemporary AI calls "attention" is treated here simply as a spatial encoder designed to pack information into a high-dimensional continuous field. Because we use floating-point numbers, sequences map to unique spatial coordinates.

#### 2. Fixed Resolution & Data Collection
*   **When Contexts $\le$ `ndim`:** The edge acts as a flawless associative data collector. It stores the exact coordinates required to map histories to targets without interference.
*   **When Contexts $>$ `ndim`:** In open-world text, sequences are infinite. The system cannot hold infinite equations, so it compresses the distribution into the fixed `ndim` parameters of the edge using an incremental Ordinary Least Squares (OLS / МНК) solver via Normal Equations:

$$S_{AA} \leftarrow S_{AA} + A \cdot A^T$$
$$S_{AB} \leftarrow S_{AB} + A \cdot B$$

Using matrix "magic" (`np.linalg.solve`), the global mathematical compromise for context routing is recalculated **incrementally and instantly** in a single clock cycle, achieving true online learning with $O(1)$ memory growth per edge.

---

### How To Use

```python
from mltt import MLTT

data = """Deep learning is a subset of machine learning that utilizes multilayered artificial neural networks, 
inspired by the human brain, to autonomously process unstructured data, recognize complex patterns, 
and make predictions. It is the driving force behind modern artificial intelligence, including 
computer vision and language models."""

alphabet = set(data)

model = MLTT(alphabet, ndim=64, window_size=32)

model.train(data)

output = model.generate("Deep")

print(output)
```

### Tiny Shakespeare Example

```ipython
model = MLTT(set(txt), ndim=256, window_size=64) # 256 - max performance in tested laptop

for line in txt.split('\n')
    model.train(line)

In [104]: model.generate('Who')
Out[104]: "Who? Varmo-bbja; YORate. b Ve.'a$lyllc':-n.'Waje,-Lell;-m? u;-mpau."

In [105]: model.generate('How')
Out[105]: "How! aye; Wod yckeximf wk:-avek,' Flpy'Sdi.-d, ocy,-m Whtm; Th; Bua"

In [106]: model.generate('What')
Out[106]: "What! Frt-f? if;'Terod;'d! oy:'bngtluf;-voqu.-ncoequf; ja tartn:'Ho,"

In [107]: model.generate('Why')
Out[107]: "Why? Fomakum; fe! Frd-ci, My?'p-ajohuabhi. fu!'b: onco;'dspiapiom; "

In [108]: model.generate('Where')
Out[108]: "Wherebopa St'II;  out.'; dnkf?-g'Vaviavegus b:'eg'tn'sto?-m; Lumpy:'t"

In [118]: model.generate('How are you?')
Out[118]: "How are you? ys ARE Bau.'ssitcta$ly'dr,': qum?-sp, Yoqungite?-tr.'de?--tr.-d"

In [119]: model.generate('What are you doing?')
Out[119]: "What are you doing?'ln &cirlgrlg-ia;-O?'dg-n ve Yo. Ditap Qu?-why.'tau?----m;----da"

In [120]: model.generate('What is your name?')
Out[120]: "What is your name?'lf! Bosughm r-ogni. Dica!-Lo, ju:-dltw'Bu; akscaug:'cct w'ds;'d"
```

#### Key Observations -- Seems:

- **Contextual Marks:** It understands special marks after question words (`Who?`, `How!`).
- **Capitalization:** It uses capital letters normally and opens sub-contexts for character names (`YORate`, `Vaviavegus`).
- **Word Length:** It creates words of a normal, human-like length (2-6 characters) rather than endless strings.
- **Whitespace:** It distributes spaces and punctuation naturally, preserving standard text rhythm.
