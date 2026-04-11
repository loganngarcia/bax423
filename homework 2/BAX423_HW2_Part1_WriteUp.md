# BAX 423 Homework 2 Part 1

Logan Garcia, Bonnie Hines

## Part 1 setup for StreamEdge

They track n = 2e6 cached video segments per edge box, use m ≈ 1.92e7 bits, and k = 6 hash functions. The usual Bloom approximation at capacity is p ≈ [1 − e^(−kn/m)]^k.

## a. False positive probability when the filter is full

Work in one chain so a grader can follow it.

- kn/m = 6 × 2e6 / 1.92e7 ≈ 0.626
- e^(−kn/m) ≈ 0.535
- 1 − e^(−kn/m) ≈ 0.465
- p ≈ 0.465^6 ≈ 0.0101

So p is about 1.01% if you round to two decimals.

## b. Cut that false positive rate in half without changing n

You mainly have two knobs, more bits m or more hash functions k. Bigger m spreads keys across more buckets and pulls the false positive down. Higher k means more hashes per insert and lookup, so people often bump m first if k is already fixed.

The tradeoff is memory. Every extra bit is another bit on every edge server, and this filter is supposed to sit in RAM to avoid disk.

Keep k = 6 and target p_new ≈ p/2 ≈ 0.00507. Solving p ≈ [1 − e^(−6n/m_new)]^6 gives 1 − e^(−6n/m_new) ≈ p_new^(1/6) ≈ 0.34, so e^(−6n/m_new) ≈ 0.66, so 6n/m_new ≈ −ln 0.66 ≈ 0.416, and m_new ≈ 6n / 0.416 ≈ 2.9e7 bits. Call it on the order of 29M bits for a halving ballpark with the same k.

## c. Clearing bits when a segment is evicted

That breaks the filter. Many keys OR into the same bits. You zero one bit because one segment left, but another segment might still need that bit set. You start answering not in cache when something is still there, which is a false negative on membership. Bloom filters are not built for delete unless you add structure.

One fix people actually use is a counting Bloom filter, small counters per cell, increment on insert, decrement on delete, clamp at zero. Cuckoo filters are another option when you care about deletes and space.

## d. Scalable Bloom filter

Picture a list of normal Bloom filters, smallest first. Inserts go into the active one. When that stage gets too full for your false positive budget, you add a new filter with more space instead of rebuilding the world from scratch.

On lookup you query every stage and say yes if any stage says yes. If stage i has false positive rate ε_i and you design the series so the ε_i shrink fast enough, the total false positive rate stays below whatever cap you promised across all stages.

## AI use

We used AI to help with grammar and to convert our spoken thoughts into writing on Cursor voice mode.
