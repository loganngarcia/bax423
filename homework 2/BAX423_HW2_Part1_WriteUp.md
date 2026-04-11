# BAX 423 Homework 2 Part 1

Logan Garcia, Bonnie Hines

## Part 1 setup for StreamEdge

They track about two million cached video segments per edge box, use on the order of nineteen million bits of bitmap, and six hash functions. At capacity the usual Bloom filter approximation is that p is roughly one minus e to the negative kn over m, then raise that whole thing to the k.

## a. False positive probability when the filter is full

Here is one straight chain of the math so a grader can follow it.

- kn over m equals six times two million over about nineteen point two million, which is about point six two six
- e to the negative kn over m is about point five three five
- one minus that is about point four six five
- p is about point four six five to the sixth, which is about point zero one zero one

So p is about one percent if you round to two decimal places.

## b. Cut that false positive rate in half without changing n

You really have two knobs. Add more bits m, or use more hash functions k. A bigger m spreads keys across more buckets and pulls false positives down. A higher k means more hashes on every insert and lookup, so in practice teams often grow m first if k is already where they want it.

The tradeoff is memory. Every extra bit is another bit on every edge box, and this structure is meant to live in RAM so you are not hitting disk for membership checks.

Keep k at six and aim for a new p that is about half of the old one, so near point zero zero five. From p roughly equal to one minus e to the negative six n over m new, to the sixth, you work backward to one minus e to the negative six n over m new near point three four, then e to the negative six n over m new near point six six, then six n over m new near point four one six, and m new lands near thirty million bits for a halving ballpark with the same k.

## c. Just clear the bits when a segment gets evicted

That idea breaks the filter. Lots of keys OR into the same bits. You clear one bit because one segment left, but another segment might still need that bit set. You start saying not in cache when the item is still there, which is a false negative on membership. Plain Bloom filters are not built for delete unless you add structure.

A fix people ship in the real world is a counting Bloom filter, small counters per cell, bump on insert, drop on delete, clamp at zero. Cuckoo filters are another path when deletes and space both matter.

## d. Scalable Bloom filter

Think of a list of normal Bloom filters, smallest stage first. Inserts go into the active stage. When that stage gets too full for your false positive budget, you add a new filter with more space instead of rebuilding everything from scratch.

On lookup you ask every stage and say yes if any stage says yes. If stage i has its own false positive rate and you design the series so those rates shrink fast enough, the overall false positive rate stays under the cap you promised across all stages.

## AI use

We used AI to help with grammar and to turn our spoken thoughts into writing in Cursor voice mode. The math and reasoning are ours.
