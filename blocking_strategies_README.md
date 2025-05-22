# Deduplication Blocking Strategy Testing

This set of scripts allows you to test different blocking strategies for the deduplication API to identify which strategy might be causing the issue of returning 0 duplicates.

## Background

The deduplication API was not finding any duplicates in the sample data, even though duplicates exist. We've created a simplified implementation that successfully finds duplicates, but we need to identify which specific blocking strategy in the original implementation might be causing the issue.

## Available Scripts

### Implementation Scripts

These scripts implement different blocking strategies:

1. `blocking_prefix_only.py`: Uses only prefix blocking (first 4 chars of name + first char of city)
2. `blocking_prefix_metaphone.py`: Uses prefix and metaphone blocking
3. `blocking_prefix_metaphone_soundex.py`: Uses prefix, metaphone, and soundex blocking
4. `blocking_all_strategies.py`: Uses all blocking strategies (prefix, metaphone, soundex, ngram)

### Testing Scripts

1. `test_blocking_strategies.py`: Runs all four implementations and reports the results
2. `modify_app_blocking.py`: Modifies app.py to use a specific blocking strategy

## How to Use

### Testing Local Implementations

To test all blocking strategies locally and compare their results:

```bash
python test_blocking_strategies.py
```

This will:
- Run each implementation with the same parameters
- Test the API endpoint (if available)
- Run the simplified implementation from test_fixed_api.py
- Report the results in a clear, tabular format

### Modifying the API

To modify the API to use a specific blocking strategy:

```bash
python modify_app_blocking.py <strategy>
```

Where `<strategy>` is one of:
- `prefix`: Only use prefix blocking
- `metaphone`: Use prefix and metaphone blocking
- `soundex`: Use prefix, metaphone, and soundex blocking
- `all`: Use all blocking strategies

After modifying the API, you need to rebuild the Docker container:

```bash
.\rebuild_docker.ps1
```

Then you can test the API with the new blocking strategy:

```bash
python test_fixed_api.py
```

## Workflow for Identifying the Issue

1. Run `test_blocking_strategies.py` to see which local implementations find duplicates
2. If any implementation returns 0 duplicates while others find duplicates, that strategy is likely causing the issue
3. Use `modify_app_blocking.py` to modify the API to use a strategy that works
4. Rebuild the Docker container with `.\rebuild_docker.ps1`
5. Test the API with `python test_fixed_api.py` to confirm it now finds duplicates

## Expected Results

If all implementations find duplicates except one, that strategy is likely causing the issue. If all implementations find duplicates but the API still doesn't, there might be another issue with the API implementation or Docker container.

## Notes

- The `prefix` strategy is the simplest and most likely to work
- The `metaphone` and `soundex` strategies add phonetic matching, which can find more duplicates but might also cause issues
- The `ngram` strategy can find even more duplicates but might be more prone to false positives
- The `all` strategy uses all blocking strategies and should find the most duplicates, but might also be the most likely to have issues