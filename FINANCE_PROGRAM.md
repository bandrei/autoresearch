# Finance Autoresearch

This is a specialized version of the autoresearch experiment focused on **Financial Strategies and Quantitative Research**.

## Data Source
The model is trained on a curated corpus of **arXiv papers** from Quantitative Finance (`q-fin`) and Economics (`econ`). The goal is to discover architectures and training regimes that excel at modeling financial research text and numerical patterns found in strategies.

## Setup
Follow the standard setup in `program.md`, but ensure you are using the `tokenizer_arxiv` and `data_arxiv` shards.

## Objectives
1. **Low val_bpb**: The primary metric remains bits-per-byte on the validation set of financial papers.
2. **Strategy Discovery**: Beyond loss, we are looking for models that can effectively represent complex temporal and causal relationships found in financial strategies.
3. **High Sharpe Ratio**: Check `backtest_results.tsv` after each run. Your goal is to improve the model architecture so that it produces better predictive signals (leading to higher Sharpe Ratios) while maintaining or improving `val_bpb`.

## The Feedback Loop
After each `train.py` run, a `run_backtest.py` script is triggered asynchronously. 
- **Read `backtest_results.tsv`**: This file tracks how your model's "internal understanding" of finance translates into strategy performance.
- **Analyze Correlations**: If `val_bpb` goes down but `Sharpe Ratio` also goes down, you may be overfitting to the language but losing the "signal". Adjust your architecture (e.g., more attention heads, different sequence lengths) to capture better temporal signals.

## Experimentation Ideas for Finance
- **Longer Context**: Financial papers often have long-range dependencies between "Abstract" and "Conclusion" or "Data" and "Results". Experiment with `sequence_len`.
- **Numerical Precision**: Financial text is dense with numbers. Consider if specific tokenization or architectural tweaks (like different activation functions) help model numerical data better.
- **Architectural Pruning**: Financial data is often noisier than general web text. Simpler, more robust models might generalize better.

## The Loop
Follow the "LOOP FOREVER" protocol in `program.md`. You are now a Quantitative AI Researcher.
