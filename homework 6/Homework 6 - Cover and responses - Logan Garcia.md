# BAX 423 Homework 6: Recommendation Systems

Student: Logan Garcia  
Student ID: `925678478`

## Section 1: Recommender Problem Framing

**1a.** Training only on saved/rated events creates a positive-only, exposure-blind sample. The model cannot distinguish recipes a user saw and rejected from recipes never shown, so the labels are missing-not-at-random and biased toward motivated reviewers.

**1b.** Zero-filling unobserved matrix entries treats unknown user-recipe pairs as explicit dislikes. This overwhelms the observed signal, distorts means and dot products, and violates the assumption that missing entries are unknown or unexposed rather than true zeros.

**1c.** The weather/IP/ingredient system is a context-aware content-based recommender because it uses context and item attributes without historical collaborative ratings.

**1d.** Food.com eventually needs a hybrid system because pure content/context rules cannot learn latent taste, social proof, substitutes, or behavioral preference patterns. Historical interactions add collaborative signals that improve personalization and ranking.

EDA conclusion: the rating distribution is heavily negatively skewed toward 5-star reviews, illustrating selection bias in explicit feedback. The most popular and best-rated lists do not fully match, so popularity and quality are different ranking objectives.

## Section 2: Collaborative Filtering

**2a.** An item-item neighborhood model requires pairwise similarities over recipes, so memory is `O(N^2)` and the comparison/update computation is also roughly `O(N^2)` before considering sparse co-rating scans. Moving from 100,000 to 10,000,000 recipes increases the matrix from 10^10 to 10^14 entries, creating prohibitive RAM/storage pressure and very large CPU/GPU compute costs.

**2b.** Pearson correlation is undefined when either vector has zero variance, such as a user who gives the same rating to every co-rated recipe, or when two users have too few/no overlapping rated recipes. Jaccard still computes on the sets of interacted recipes because it only needs intersection and union counts.

Model conclusion from the executed notebook: the SVD and 3x-factor SVD results are printed in the notebook and screenshots. More factors only help if the added latent dimensions capture repeatable preference structure; if the data are sparse or noisy, extra factors can increase variance and hurt RMSE.

## Section 3: Two-Tower Candidate Generation

Passing a raw interacted Recipe ID into the User Tower violates the two-tower decoupling rule: the user embedding must be computable and cacheable independently of candidate item IDs. To keep serving latency under 50ms, historical recipe interactions should be summarized into user-side features or pooled historical embeddings offline; at request time the cached user vector is compared against a precomputed item-vector index.

The executed notebook implements the required PyTorch towers, prints 10 training losses, reports Hit@10 and Hit@50, and builds both exact FAISS `IndexFlatIP` and approximate `IndexIVFFlat` indexes. FAISS recall increased as `nprobe` increased, with higher recall requiring higher latency.

## Section 4: DCN v2 Ranking

A ranker should optimize a calibrated multi-objective score such as `L = w1 * BCE(CTR) + w2 * BCE(cart_conversion)` with weights tuned on validation revenue or long-term user value. At serving time, the platform can also calibrate the two predicted probabilities into `score = alpha * P(click) + (1-alpha) * P(cart_conversion)` so clickbait that earns clicks but fails conversion is demoted.

The executed notebook compared DCN v2 and Plain DNN, then ran the requested 1/2/4 cross-layer ablation. In the saved run, the 1-layer DCN had the best AUC among ablations, while deeper cross networks did not monotonically improve performance.

## Section 5: YouTube Recommendations

**5a.** Covington et al. handled freshness by adding an example-age feature during training. The model learned the bias associated with how old an example was, and at serving time that feature was set to zero, which corrected the learned tendency to over-favor older videos with more accumulated watch history.

**5b.** YouTube optimized expected watch time rather than raw CTR. They trained with weighted logistic regression where positive impressions were weighted by observed watch time and negative impressions received unit weight; the resulting odds approximate expected watch time per impression instead of only click probability.
