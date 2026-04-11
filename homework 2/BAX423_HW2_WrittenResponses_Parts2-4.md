# BAX 423 Homework 2 — Written responses (Parts 2–4)

Logan Garcia, Bonnie Hines

## Part 2 (Count–Min Sketch, CMS)

**False negatives vs false positives when thresholding CMS estimates.** Count–Min Sketch estimates are **one-sided** in the usual sense that each estimate is **at least** the true count in expectation for the min across rows, but **collisions** can **inflate** counts. When we flag “attack” using a fixed threshold on estimated flows, **false positives** (benign sources pushed over the threshold) are usually the main worry with a **small width**, because hash collisions add noise upward. **False negatives** (true high-volume sources reported below the threshold) can still happen, but they are less typical than inflation when the table is narrow. Tuning **width and depth** trades memory for collision noise.

## Part 3 (LightGBM vs XGBoost, GOSS, EFB)

**Accuracy and speed.** On this UNSW-NB15 pipeline (one-hot categoricals, log-scaled nonnegative numerics), LightGBM is **competitive or better on accuracy** than the XGBoost baseline while often **training and inferring faster** on wide sparse tabular data, because it exploits histogram-based splits and efficient data layouts. Exact numbers appear in the comparison table in the notebook export JSON.

**GOSS (Gradient-based One-Side Sampling).** GOSS **undersamples a large fraction of low-gradient examples** and keeps high-gradient examples, so each iteration spends work where the loss surface is changing fastest. It tries to **preserve accuracy** by reweighting so the overall gradient direction stays representative. **Speed gains** depend on whether gradient sorting and sampling overhead dominate; on smaller or already fast rounds, GOSS may not look dramatically faster.

**Exclusive Feature Bundling (EFB).** In LightGBM, **mutually exclusive** sparse features (rarely nonzero together) can be **packed into bundles** to reduce the effective feature dimension for histogram building. That **reduces the cost per split** and speeds training when many sparse indicators exist.

## Part 4 (Range queries and dyadic intervals)

**8000-word range vs 4000-word intervals.** If fixed sketches are aligned to **4000-word chunks**, an 8000-word range that **does not align** to those chunk boundaries mixes **partial intervals**, so error accumulates from **multiple CMS estimates** and boundary effects. Accuracy is not identical to a single aligned 8000-word block unless the query decomposes cleanly.

**Bounding error with 99% probability.** To keep total range error small with high probability, you typically **shrink the per-sketch epsilon** (wider/deeper sketches), **align interval construction** so the query is a **sum of independent or weakly dependent** sub-sketches, and accept that union bounds may require **more conservative epsilon** per block when you sum many blocks.

**Dyadic intervals.** Any interval can be covered by **O(log L)** power-of-two aligned blocks (standard dyadic decomposition). That controls how many sketch estimates you sum when answering an arbitrary range query.
