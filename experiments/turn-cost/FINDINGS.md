# Where an agent's AUA calls go

This note keeps the app-agnostic conclusions from an internal journal analysis. The source
journals, package attribution, app-map cardinalities, exact goals, timings, and per-app results are
intentionally not published in this public repository.

## Conclusions

- Perception calls dominate interactive use, and consecutive identical reads are a meaningful
  source of avoidable work.
- Agents often read a map and then pass an exact internal destination to `goto`; sentence-shaped
  search is useful, but it is not enough on its own to reduce turns.
- `goto` is most valuable for multi-step jumps. For an adjacent visible destination, acting on the
  current observation is naturally cheaper than loading and querying a map first.
- Cache invalidation should follow screen-changing actions. Pure repeated reads can reuse a prior
  result unless the caller explicitly disables caching.

## Reproducible public method

To reproduce the analysis without private application data:

1. Generate journals from fictional fixtures or stock Android Settings.
2. Classify calls into perception, manual action, navigation, and other operations.
3. Count consecutive calls with identical command, arguments, and device identity, stopping at any
   operation that can change the screen.
4. Replay navigation goals only against the synthetic map that produced them.
5. Report aggregate ratios without package names, screen names, raw prompts, selectors, routes, or
   per-application cardinalities.

Synthetic input is required for any committed result. Analyses of a user's local journals and
maps belong in local artifacts, not in this repository.
