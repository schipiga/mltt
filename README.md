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
model = MLTT(ndim=16, window_size=8)
lines = [line for line in txt.split("\n") if line]

for line in lines:
    model.train(line)
model.solve()
```

#### Results:

**In [7]:** `model.generate('Oh, my lord', length=128, temperature=0.0)`
**Out[7]:** _"Oh, my lord? or with't, than for myself? it lawful king thy hands, naked, dry your enemies? we are you thanks. LEWIS: although thy bones, well that live to say we now may soon perform it meekly will we know your cousin Buckingham. What say! that Is this over in Yorkshire are those eyes; look about you understand me? Camillo was never may brawling fed: how dastards. Worthy Mariana. Why should become him down for these bitter jades, arm of bleeding slaughter, sailmaker in hand in grief? See, true king had then laid on highness! though"_

**In [8]:** `model.generate('Oh, my lord', length=128, temperature=0.8)`
**Out[8]:** _"Oh, my lord the tub. where they come so it strange dower of Henry is bankrupt of shadows by garment, now come those branches by right or one at evening mass? Think with consent to conceal what oar: thank my infant. Blossom, now begin with Volscian lords, heard it all myself to ride's from head; fourteen. Light to win our reasons urged it stint thou sprang's at bold conspiracy, men hate myself into some chat aside their souls--would take from schoolboys from Port le Blanc, Susan Grindstone in parliament pledge: foolish knave in virtue go away: attorneys to, asked him safely ordered: dear friends but"_

**In [9]:** `model.generate('Oh, my lord', length=128, temperature=0.1)`
**Out[9]:** _"Oh, my lord's not fly from Oxfordshire shalt neuter. Dostulate the vows to hang yourselves. Camillo, Archbishop late thou object the tavern climely owes two men hate thee than taunts himself: repeal, Lord Stanley. Pray to stoics, temperance. Dighton, bodes. Provost? Dido slew to dive into your suits of many servants: it light wings: fresh water. Now, godded my weeping toads; lave that means that flatter. Me, mutton and husband: besides I should it more strength to sleep will in theirs. This has made an eight to assault thy bent"_

**In [10]:** `model.generate('Oh, my Queen', length=128, temperature=0.0)`
**Out[10]:** _"Oh, my Queen Margaret's sudden, milder you show'd thee mine earlorn, seal'd your request? call my most sovereign lie thou shalt not answer his eyes the unsway homely! all alone. 'The unreasonable fury to lose the Sunday comes foremost: to silence: with smoking swords of such business, curst been a leader. Is this; goest in blood upon, seeming, these hands your walls? whence are made of lead me how far gone. Camillo, Braken his neck: learn'd death is noble father! wreat of wicked bescue it mustering; so many, unto these my boy; this"_

**In [11]:** `model.generate('Oh, my Queen', length=128, temperature=0.1)`
**Out[11]:** _"Oh, my Queen Margaret, quick in presence might pardon'd swift business sound. Still, speaking fails. Poor Anne, senators and conjures she says he'That brought stone walls. Grumny Kate! dead bones: the clamour's set footing. Here's set before thee! Can thy coming. Neighbour, display'd home in quiet life is our kitchens"_

**In [12]:** `model.generate('Oh, my Queen', length=128, temperature=0.8)`
**Out[12]:** _"Oh, my Queen Hermione as far surm, too rash remiss, respecting you dear dear importuned drums: when you friends. Yet welcome.--In base declension and bleed's ransom of Agamourner. Friend hast it farther. Friar-pot, Vaughan, strengths do confess the order how like unscour lift them accordingly. Ratcloutrunning more worth such we beat water though with nett taste prove him best thou happy gilded lopp'd throat of worms' tongues. Twice have broken points? Away, gods bless you teach my walks, language. And now be whilst thou rag, quick convey my Katharums? dispute with"_

**In [13]:** `model.generate('Dear King', length=128, temperature=0.0)`
**Out[13]:** _"Dear King Polixenes, see another's in parliament shall. They are come before your gates. Even now she did know her dastly here protest against thy general, king my metal in right Vincentio. They say to kiss. Lady Grey, right; when he holds his lastly of blood! they had one and your pricks him home thy foul deed. Adieu! so wide gap into his very late. Thou wast a prisoner? my inheritance of harm but sick man but himself will thereto frame of wotting-beards dead that may soon recover: and grant; God--I would increase, horn. Nay, stew"_

**In [14]:** `model.generate('Dear King', length=128, temperature=0.1)`
**Out[14]:** _"Dear King Polixenes, cousin should bear with another ballad is more proof against that sparing makes a pale and mother come? speak again by night, adultery, deaf as half by Edward come coffin'd run. Had we, reigns the stage, protesting oath on; says your garden-tree tops of nineteen and mistress is frank'd number, instruments shall bring away to shape, measure to heal their chairs again when first merriment of battle came I surmounts shrived in man speak a Capulet will do? his concord, slain by joint-men, beautiful Binaca full two are broke their hands do another outrun the"_

**In [15]:** `model.generate('Dear King', length=128, temperature=0.8)`
**Out[15]:** _"Dear King Boling, forget that taught his castle: buy any man live chides'clock, sorrow; acquaint his offer me.' Sweet Montagues, should you back? Whom hence in sour annoy'd waters, fleer better person: between your approbeshrew, fly their sugar on them well resembles it fits, down so putting the balms, midnight? comfort lives or I follow'd Richard. Come and known evils, James Soundpost? Grace! Lay hands? thou rag, Setebling rogue, Is my choice love have most contrarious quests"_

**In [16]:** `model.generate('The time is coming, brothers', length=128, temperature=0.0)`
**Out[16]:** _"The time is coming, brothers unto some new made his: he come at leisure for Ireland. Thus with those whom here thou think the tenth of banishment. Then all our streets, nail, blessed was ware-shearing, although thy state to pass. Hear me father and your vantage. Hermione, stars, seven thousand deaths. Are they live: that man and how she survive me fair queen! fard. Can you company. Lords of mine. Wot ye. Our pre-morrow morning: they account's bosom, gentlemen: they themselves as well the justice: fresh water, even thus; where have in earth be. Away with their sufferance"_

**In [17]:** `model.generate('The time is coming, brothers', length=128, temperature=0.1)`
**Out[17]:** _"The time is coming, brothers unto. Lay hands: do their affects with bright. Walk before with love or end, great King of better wear upon their grief must talk; follow us and why weaved and leisure to commit your bade thy stable, Camillo; take note, reconcile them again cry, sworn my royal hands received. Thou art thou still should sprinkle me is worse: Yet give her match. Thy best success, time would give 'scape. Some say King Harry true. 'ay' eyes: shall then dreams; swearing both panting dance. Lady Bess those whose heavy looks foretell of wrath or other forfeits hither now prosperity! Trust"_

**In [18]:** `model.generate('The time is coming, brothers', length=128, temperature=0.8)`
**Out[18]:** _"The time is coming, brothers? use of VAUGHANIO: silence, looking on fees, boys, Minola? Thy bestridays all straining on now is green and call nature with blood was affable chance may never. O Jupiter! Wife, sentenced himself over many an aspect with achesus found his great Hercules, sent not dukedom and goes worse issued, arise, by age, descending now reprobriously and full quit his castle, faintly borne your tribunes i'erbear thy priesthood saves thy brazen gates made in wild, vizard hide his full year. Friar Laurence' ho! Fly, buy some more our"_


The result is looking like funny statistical babbler, but seems it started to feel the language semantic.
