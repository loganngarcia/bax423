# BAX 423 Homework 2 — Part 1 (Bloom filters)

Logan Garcia, Bonnie Hines

## StreamEdge Bloom filter setup

Given parameters: n = 2,000,000 items, bit array size m = 19,170,117, and k = 6 independent hash functions. The standard approximate false positive probability when the filter holds n distinct items is

p ≈ (1 − exp(−kn/m))^k.

### (a) False positive probability at capacity

First compute kn/m = 6 · 2,000,000 / 19,170,117 ≈ 0.62587.

Then exp(−kn/m) ≈ 0.53474, so 1 − exp(−kn/m) ≈ 0.46526, and

p ≈ (0.46526)^6 ≈ 0.010143.

So the approximate false positive rate is about **1.01%** (rounded to two decimal places).

### (b) Halving p without changing n

To lower the false positive rate you can **increase m** (more bits) and/or **increase k** (more hash functions). Raising k also increases work per lookup, so teams often grow **m** first to hit a target p while keeping k fixed.

**Trade-off.** Bits are memory. A larger m drives exp(−kn/m) closer to 1, which shrinks (1 − exp(−kn/m)) and drops p quickly, but RAM at every edge server scales linearly with m.

**New m with k = 6 fixed.** We want p_new ≈ p/2 ≈ 0.0050716. Set (1 − exp(−6n/m_new))^6 = p_new, so 1 − exp(−6n/m_new) ≈ p_new^(1/6) ≈ 0.3400, hence exp(−6n/m_new) ≈ 0.6600. Then 6n/m_new ≈ −ln(0.6600) ≈ 0.4155, so m_new ≈ 6n/0.4155 ≈ 2.89 × 10^7 bits. In other words, about **22.4 million bits** is in the right range to cut the false positive rate roughly in half for this n and k.

### (c) Why clearing bits on eviction is wrong

A Bloom filter is a **lossy fingerprint**. Many keys share the same bits. If you flip a bit to 0 because one key was deleted, you may remove evidence for other keys that still map to that bit, which creates **false negatives** (the filter says “definitely not in set” when the item is still cached elsewhere). You cannot safely delete without extra per-cell information.

**Structure that supports deletion.** A **Counting Bloom filter** stores a small counter per cell instead of a single bit, so inserts increment and deletes decrement (until zero). Alternatives include **cuckoo filters** or other structures designed for approximate membership with safer delete semantics.

### (d) Scalable Bloom Filter (SBF)

An SBF is a **sequence of Bloom filters** B1, B2, …. New inserts go into the active filter. When the estimated load would push the false positive rate of the active stage past a budget, you **add a new, larger** Bloom stage rather than rebuilding everything.

**Query** checks **all** stages (membership is true if any stage returns true). **False positive guarantee:** if each stage i is tuned so its FP probability is at most ε_i and the series Σ ε_i is bounded (often geometric), the overall FP probability stays below a chosen cap even as new stages are appended.
