# BAX 423 Homework 2 Written responses (Parts 2 through 4)

Logan Garcia, Bonnie Hines

## Part 2 Count-Min Sketch and the histogram

When you threshold sketch counts to call something an attack, collisions matter more than the “minimum across rows is unbiased” story from lecture. A skinny table collides a lot, so counts get pushed up. In our setup that usually shows up as false positives, meaning a quiet source can look hot. False negatives can happen, but undercounting every row is not what CMS is famous for, so you sweat width and depth before you trust the histogram as ground truth.

## Part 3 LightGBM runs

On this tabular pipeline, LightGBM landed in the same accuracy ballpark as XGBoost, often with less wall time, which matches what you expect on wide one-hot data where histogram splits help. Exact train and infer seconds and accuracy are in `unsw_nb15_lightgbm_measurements.json` next to the notebook.

GOSS samples harder examples more often so each boosting round spends gradient work where the loss is moving. You can still pay for sorting and bookkeeping, so speedup is not automatic on every dataset size.

EFB is the idea that sparse columns that almost never fire together can be bundled so the tree code touches fewer effective columns per split. That is a training-time win when your feature space is mostly indicators.

## Part 4 Range queries on text

If you built one sketch per fixed 4000-word chunk, an 8000-word window that does not line up on those boundaries stitches together partial chunks. You add more sketch answers, so error adds up differently than one clean 8000-word block.

If you want a 99% style guarantee on total error, you tighten each sketch, usually with a smaller epsilon per sketch and enough rows, and you stay honest about how many sketches you sum. More segments in the sum usually means you budget epsilon per segment more conservatively.

Dyadic intervals are the usual trick so any range breaks into a handful of power-of-two aligned pieces. You query one sketch per piece and add. That is how you stop forcing every user question to match your original chunk boundaries.
