# BAX 423 Homework 2 Written responses for Parts 2 through 4

Logan Garcia, Bonnie Hines

## Part 2 Count-Min Sketch and the histogram

When you threshold sketch counts to call traffic an attack, collisions matter more than the lecture line about the minimum across rows being unbiased. A skinny table collides a lot, so counts get inflated. In our setup that mostly shows up as false positives, where a quiet source looks hot. You can get false negatives too, but CMS is not really known for systematically undercounting every row, so we cared about width and depth before treating the histogram like ground truth.

## Part 3 LightGBM runs

On this tabular pipeline LightGBM ended up in about the same accuracy band as XGBoost, often with less wall time, which lines up with what you expect on wide one-hot data where histogram splits help. Exact train and infer seconds and accuracy are in unsw_nb15_lightgbm_measurements.json next to the notebook.

GOSS samples harder examples more often so each boosting round spends gradient work where the loss is actually moving. You still pay for sorting and bookkeeping, so speedup is not automatic at every dataset size.

EFB bundles sparse columns that almost never fire together so the tree code touches fewer effective columns per split. That is mostly a training-time win when your feature space is mostly indicators.

## Part 4 Range queries on text

If you built one sketch per fixed 4000-word chunk, an 8000-word window that does not line up on those boundaries stitches together partial chunks. You add more sketch answers, so error stacks differently than one clean 8000-word block.

If you want a 99% style guarantee on total error, you tighten each sketch, usually with a smaller epsilon per sketch and enough rows, and you stay honest about how many sketches you sum. More segments in the sum usually means you budget epsilon per segment more conservatively.

Dyadic intervals are the usual trick so any range breaks into a handful of power-of-two aligned pieces. You query one sketch per piece and add. That way you are not forcing every user question to match your original chunk boundaries.

## AI use

We used AI to help with grammar and to convert our spoken thoughts into writing on Cursor voice mode.
