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

model = MLTT()

model.train(data)
model.solve()

output = model.generate("Deep")

print(output)
```

### Tiny Shakespeare Example

```ipython
model = MLTT(set(txt), ndim=12, window_size=8)
lines = [line for line in txt.split("\n") if line]

for line in lines:
    model.train(line)

In [9]: model.generate('How are you?', length=16)
Out[9]: "How are you? In his presence must hear your daughter is gone till you'll straight leg and deep"

In [10]: model.generate('What is your name?', length=16)
Out[10]: "What is your name? a consents but severe. Has he knows whoso empties' pleasure."

In [11]: model.generate('ROMEO:', length=16)
Out[11]: "ROMEO: let it concerns to ask me rather glister enter'd when your sheep-work"

In [12]: model.generate('KING', length=16)
Out[12]: 'KING RICHARD III: good prayers for England. My father so wise, lay'

In [13]: model.generate('I want to', length=16)
Out[13]: 'I want to hope good supporters are certainly whipped out: mark me pains to great; a paper'

In [14]: model.generate("Let's go to", length=16)
Out[14]: "Let's go to one so old as easy matter which you no title of green! Well, together"

In [15]: model.generate("The time will come", length=16)
Out[15]: 'The time will come when thou shalt wish for me yet his face to dissemble deeply their fortunes both'

In [16]: model.generate("Time comes", length=16)
Out[16]: 'Time comes with tender patience here was Henry from Burgundy, rice, learn, embrace but'
```

The result is a funny statistical tidbit, some of whose meaningful phrases are missing from the original text, for example, "My father so wise," -- I checked. I think the results should be better with higher dimensions.
