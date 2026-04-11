# BAX 423 Homework 2 Written responses (Parts 2 through 4)

Logan Garcia, Bonnie Hines

## Part 2 Count-Min Sketch and the histogram

Once you pick a threshold and treat sketch counts like “this is an attack,” collisions are what bite you—not the fact that the minimum across rows is unbiased in theory. If the table is too narrow, hashes collide a lot and counts get bumped up. In our setup that mostly showed up as false positives: a quiet source can look busy. You can get misses too, but CMS is usually talked about for overcounts, so we’d widen the sketch (more columns / more rows) before we’d treat the histogram as ground truth for decisions.

## Part 3 LightGBM runs

On this tabular pipeline, LightGBM ended up roughly in the same accuracy range as XGBoost, often with shorter wall-clock time. That lines up with what you’d expect on wide one-hot features where histogram splits help. Exact train/infer seconds and accuracy are in `unsw_nb15_lightgbm_measurements.json` next to the notebook.

GOSS samples the harder examples more often each round so gradient work goes where the loss is still moving. There’s still sorting and bookkeeping, so you don’t automatically get a speedup on every dataset size.

EFB bundles sparse columns that rarely fire together so the tree code can touch fewer effective columns per split. It’s mainly a training-time win when most of your features are indicators.

## Part 4 Range queries on text

If each sketch is for a fixed 4000-word chunk, an 8000-word window that doesn’t line up on those edges is really mixing partial chunks. You sum more sketch answers, so error stacks differently than if you had one sketch built for a single 8000-word block.

If you want something like a 99% style bound on total error, you tighten each sketch—usually smaller epsilon per sketch and enough rows—and you’re clear about how many sketches you’re adding together. More pieces in the sum usually means budgeting epsilon per piece more conservatively.

Dyadic intervals are the usual approach: break the range into a few power-of-two aligned segments, query one sketch per segment, add the results. That way you’re not forcing every user question to match the chunk boundaries you chose when you built the sketches.

## AI use

We used AI to help with grammar and to convert our spoken thoughts into writing on Cursor voice mode.
