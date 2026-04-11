# BAX 423 Homework 2 Part 1

Logan Garcia, Bonnie Hines

## Part 1 setup (StreamEdge)

They track n = 2,000,000 cached video segments per edge box, use m = 19,170,117 bits, and k = 6 hash functions. The usual Bloom approximation at capacity is p ≈ (1 − e^(−kn/m))^k.

## a. False positive probability when the filter is full

Numbers in one place:

- kn/m = 6 × 2,000,000 / 19,170,117 ≈ 0.62587
- e^(−kn/m) ≈ 0.53474
- 1 − e^(−kn/m) ≈ 0.46526
- p ≈ (0.46526)^6 ≈ 0.010143

So p ≈ 1.01% rounded to two decimals.

## b. Cut that false positive rate in half without changing n

Two levers: add bits (m) or add hash functions (k). More bits spreads keys out and drops the FP rate. Raising k adds work on every insert and lookup, so in practice teams often grow m first if k is already set.

The catch is memory—more bits per edge box, and this thing is meant to live in RAM, not on disk.

Keep k = 6 and target p_new ≈ p/2 ≈ 0.0050716. From p ≈ (1 − e^(−6n/m_new))^6 you get 1 − e^(−6n/m_new) ≈ p_new^(1/6) ≈ 0.3400, so e^(−6n/m_new) ≈ 0.6600, so 6n/m_new ≈ −ln(0.6600) ≈ 0.4155, and m_new ≈ 6n / 0.4155 ≈ 2.89 × 10^7 bits. Call it about 22.4 million bits for a clean halving ballpark with the same k.

## c. “Just clear the bits when a segment is evicted”

That doesn’t work. Lots of keys share the same bits. You clear a bit because one item left, but another key might still need it. You’ll start saying “not cached” when the data is still there—a false negative. Standard Bloom filters don’t support deletes without extra machinery.

In production you’d see counting Bloom filters (counters per cell, increment/decrement, clamp at zero) or sometimes cuckoo filters if deletes and space matter.

## d. Scalable Bloom filter

Think of a chain of regular Bloom filters, smallest stage first. New inserts go into the active stage. When that stage is too full for your FP budget, you add another stage with more room instead of rebuilding everything.

Lookup checks every stage; if any stage says “maybe,” you say “maybe.” If stage *i* has false positive rate ε_i and you pick the ε_i so they shrink fast enough across stages, the overall FP rate stays under the cap you committed to.

## AI use

We used AI to help with grammar and to convert our spoken thoughts into writing on Cursor voice mode.