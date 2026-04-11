# BAX 423 Homework 2 Part 1

Logan Garcia, Bonnie Hines

## Part 1 setup (StreamEdge)

They track n = 2,000,000 cached video segments per edge box, use m = 19,170,117 bits, and k = 6 hash functions. The usual Bloom approximation at capacity is p ≈ (1 − e^(−kn/m))^k.

## a. False positive probability when the filter is full

Work in one chain so a grader can follow it.

- kn/m = 6 × 2,000,000 / 19,170,117 ≈ 0.62587
- e^(−kn/m) ≈ 0.53474
- 1 − e^(−kn/m) ≈ 0.46526
- p ≈ (0.46526)^6 ≈ 0.010143

So p ≈ 1.01% rounded to two decimals.

## b. Cut that false positive rate in half without changing n

You mainly have two knobs: more bits m, or more hash functions k. Bigger m spreads keys across more buckets and pulls the false positive down. Higher k means more hashes per insert and lookup, so people often bump m first if k is already fixed.

The tradeoff is memory. Every extra bit is another bit on every edge server, and this filter is supposed to sit in RAM to avoid disk.

Keep k = 6 and target p_new ≈ p/2 ≈ 0.0050716. From p ≈ (1 − e^(−6n/m_new))^6 you get 1 − e^(−6n/m_new) ≈ p_new^(1/6) ≈ 0.3400, so e^(−6n/m_new) ≈ 0.6600, so 6n/m_new ≈ −ln(0.6600) ≈ 0.4155, and m_new ≈ 6n / 0.4155 ≈ 2.89 × 10^7 bits. Call it about 22.4 million bits for a clean halving ballpark with the same k.

## c. “Just clear the bits when a segment is evicted”

That breaks the filter. Many keys OR into the same bits. You zero one bit because one segment left, but another segment might still need that bit set. You start answering “not in cache” when something is still there, which is a false negative on membership. Bloom filters are not built for delete unless you add structure.

One fix people actually use is a counting Bloom filter: small counters per cell, increment on insert, decrement on delete, clamp at zero. Cuckoo filters are another option when you care about deletes and space.

## d. Scalable Bloom filter

Picture a list of normal Bloom filters, smallest first. Inserts go into the active one. When that stage gets too full for your false positive budget, you add a new filter with more space instead of rebuilding the world from scratch.

On lookup you query every stage and say yes if any stage says yes. If stage i has false positive rate ε_i and you design the series so the ε_i shrink fast enough, the total false positive rate stays below whatever cap you promised across all stages.

## AI use

We used AI in Cursor to help with grammar and to turn our spoken thoughts into written text using Cursor voice mode. The math and reasoning are ours.
