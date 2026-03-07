# Architecture Notes

## Objective
Recommend SKUs that are both likely to be purchased and cheaper alternatives.

## Two-stage flow

1. Candidate generation
- Filter by eclass overlap with user history.
- Prefer same/similar manufacturers.
- Optionally include substitution graph edges.

2. Ranking
- Train a global ranking model on `(user, item)` features.
- Output `p_buy`.

3. Price-aware reranking
- Compute `price_advantage` against user reference price.
- Final score: `p_buy * (1 + price_advantage)`.

## Why one global model
- Scales across many categories and SKUs.
- Avoids maintaining thousands of per-product models.
- Learns shared structure from sparse purchase behavior.
