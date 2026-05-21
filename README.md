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
model = MLTT(set(txt), ndim=32, window_size=32) # 256 - max performance in tested laptop

for line in txt.split('\n')
    model.train(line)

In [19]: model.generate('Who', length=128)
Out[19]: "Whoe, mugrsw rubs, ck'd I: roqunuftssqu!-Meg.  kf-og?-y-wff Gr! ry g: LY:'DO:'dwmle;'d'g;'ghhhy!'d c!-HEObds?'bt'b; s CLAMPrkys'Roy"

In [20]: model.generate('How', length=128)
Out[20]: "How! Waren! irdf s l kw lengyn,'TAEctf?'STyvui, zyf?'CEYOrquzynmiisib.-ax, vu!'d;-izl:'dy!-pt-Mikm? Wa. UGORTIA:-oja?--MPrvoh; NG Y"

In [21]: model.generate('What', length=128)
Out[21]: "What! D: L: foy? a Sms. r: Ve On:-P:-N:-ora.'MIf?'dce?-i,'dc!-equn'g HATub'YOvyme'CKn!-y Sw p!'RWow;-u-BRRGALE:-g:-nnzyhy!'JUpyw-b,-"

In [22]: model.generate('Why', length=128)
Out[22]: "Why? Fod odm lmy lntnc tilielod td:-nmlufsque yf.-Hy'dgfus, u!-w?'i-lc.'LEublng?-ynd?'why!'g:'r; Sivud;-fax,'mp:-l;'Y FOMu? '?--iii"

In [23]: model.generate('Where', length=128)
Out[23]: "Whereory vicoverjod-jovy imm? s;'hyfaudgrgnnveodntt, ourpfs! A: ru:-iebja-ft?-eynzer-juk; h!'p!'rrdjomsft.'df? f  's; QUGSldd. kfy' z"

In [24]: model.generate('How are you?', length=128)
Out[24]: "How are you? TRLy: a; l w h'ft'se Wasgfldc,'b;-Gunorb;-hyhtmp!-att?'ep, rjuib!'d!-e?'sc. tw;-'d? 'Y u? Mer-P sgieffa: s, u-p?'dncibdp'w UFFO"

In [25]: model.generate('What are you doing?', length=128)
Out[25]: "What are you doing?-injer inju:-kojajaw:'!'dpftfu-M:-ore,-yva sl; krgmy: hy insy w: dgr YO: qu w rbi,-pm; lcek?-qug; d-b,-IEdc; mt'rh: ds OP:-vaja:"

In [26]: model.generate('What is your name?', length=128)
Out[26]: "What is your name? IEONLEO! mb, LIs?-f-y!'STyvyw! p?-Crwnvy b!-Muajau!-ymw BI: ib.-jas. rwstg? Je.'R: f Niz,-n; zz, QUMIrs!-mn, vittgsdja?-afta t."
```

While the raw text outputs are expectedly chaotic at this scale, fascinating statistical emergence can already be observed -- working with capital letters, spaces, apostrophes, etc. Perhaps this is a sign of emerging order.
