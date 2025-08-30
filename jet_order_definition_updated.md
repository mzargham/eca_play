# 🧠 Boolean Jet Spaces and Jet-Order in ECA

## Overview

We introduce a refined framework for analyzing Elementary Cellular Automata (ECA) via **Boolean Jet spaces**. Two complementary jet constructions are defined: the **XOR Jet** and the **Boolean Taylor Jet**. These tools help reveal structure, complexity, and cyclicity within ECA rules.

---

## ✴️ Two Jet Constructions

### 🔁 XOR Jet

Given an ECA rule \( f \), define the **XOR Jet** as:

\[
J^k_{	ext{XOR}}(x) = \left[ x,\; x \oplus f(x),\; x \oplus f^2(x),\; \dots,\; x \oplus f^k(x) ight]
\]

Each derivative measures the divergence from the **initial state**. This highlights how much the state has changed cumulatively, useful for:

- **Cycle detection** (back to origin)
- **Measuring orbit divergence**
- **Change accumulation tracking**

However, it does **not reflect local rule structure**, and can become noisy due to compounding changes.

---

### 📐 Boolean Taylor Jet

The **Boolean Taylor Jet** approximates a Boolean analog of a derivative chain:

\[
J^k_{	ext{Taylor}}(x) = \left[ x,\; f(x) \oplus x,\; f^2(x) \oplus f(x),\; \dots,\; f^k(x) \oplus f^{k-1}(x) ight]
\]

Each derivative compares successive applications of \( f \), acting as a **local difference operator**. This jet type reveals:

- **Latent structure**
- **Pattern stabilization**
- **Repetitive motifs**
- **Smoothness/chaoticity over time**

Taylor jets are ideal for **visual and structural analysis** of nonlinear ECA rules.

---

## 🚀 Strategy: Combine Both!

We propose a hybrid method:

1. **Start with Taylor Jet**:
   - Compute deeper orders to reveal latent regularity.
   - Patterns often stabilize, compress, or repeat.

2. **Then apply XOR Jet**:
   - Use to detect **when** orbits return.
   - Helps determine cycle period after Taylor structure emerges.

This two-step analysis combines **symbolic smoothness detection** with **cyclicity measurement**.

---

## 🧩 Jet-Order Definition (Revised)

We define:

> A rule \( f \) is **Jet-Order** \( k \) if there exists a Boolean tensor expression \( \hat{f}_k \) such that:
>
> \[
> f(x) = \hat{f}_k(J^k_{	ext{Taylor}}(x)) \quad 	ext{for all } x \in \mathbb{B}^n
> \]

Thus, the Taylor jet space of order \( k \) contains all necessary information to reconstruct the rule.

---

## 📊 Use Case Summary

| Jet Type       | Tracks          | Good For                             |
|----------------|------------------|--------------------------------------|
| XOR Jet        | Divergence from start | Cycle detection, change magnitude   |
| Taylor Jet     | Local differences | Latent rule structure, smoothness   |

---

## 🔢 Example (Rule 110)

- Taylor jet reveals structure begins to **repeat** at higher orders.
- XOR jet shows the **return to origin** clearly once a cycle completes.
- Combining both provides a full characterization of the dynamics.

---

## 🧮 Next Steps

- Compute and visualize both jets for arbitrary rules.
- Detect cycle periods.
- Use jets to identify minimal Boolean tensor expressions for reconstruction.