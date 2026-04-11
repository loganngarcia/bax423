# BAX 423 Homework 2 Written responses for Parts 2 through 4

Logan Garcia, Bonnie Hines

## Part 2 Count-Min Sketch and the histogram

When you threshold sketch counts to call something an attack, collisions matter more than the lecture line that the minimum across rows is unbiased. A skinny table collides a lot, so counts get pushed up. In our run that usually shows up as false positives, meaning a quiet source can look hot. False negatives can happen, but undercounting every row is not what CMS is known for, so you care about width and depth before you treat the histogram like ground truth.

## Part 3 LightGBM runs

On this tabular pipeline LightGBM ended up in the same accuracy ballpark as XGBoost, often with less wall time, which lines up with what you see on wide one-hot style data where histogram splits help. Exact train and infer seconds plus accuracy are in the small JSON file that sits next to the notebook.

GOSS samples harder examples more often so each boosting round spends gradient work where the loss is actually moving. You still pay for sorting and bookkeeping, so speedup is not automatic on every dataset size.

EFB bundles sparse columns that almost never fire together so the tree code touches fewer effective columns per split. That is mostly a training-time win when your feature space is mostly indicators.

## Part 4 Range queries on text

If you built one sketch per fixed four thousand word chunk, an eight thousand word window that does not line up on those boundaries stitches together partial chunks. You add more sketch answers, so error stacks differently than one clean eight thousand word block.

If you want a ninety-nine percent style guarantee on total error, you tighten each sketch, usually a smaller epsilon per sketch and enough rows, and you are upfront about how many sketches you sum. More segments in the sum usually means you budget epsilon per segment more conservatively.

Dyadic intervals are the usual move so any range breaks into a handful of power-of-two aligned pieces. You query one sketch per piece and add. That way you stop forcing every user question to match your original chunk boundaries.

## AI use

We used AI to help with grammar and to turn our spoken thoughts into writing in Cursor voice mode. The technical content is ours.
