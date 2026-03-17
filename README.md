# Supply Allocation Optimizer 

LP-based tool for agri-food consortia to optimally allocate sales flows across producers, buyers, and products.

## Problem

Given expected production `b[i,r]` and inventory targets `a[j,r]`, find sales flows `x[i,j,r] ≥ 0` that minimize total deviation from both targets, with a sparsity penalty to avoid marginal logistics flows:

$$\min_{x \geq 0} \sum_{r} \left( \sum_j \left| \sum_i x_{ijr} - a_{jr} \right| + \sum_i \left| \sum_j x_{ijr} - b_{ir} \right| \right) + \lambda \sum_{ijr} x_{ijr}$$

Solved as a Linear Program via ℓ1 linearization.

## Usage

```bash
pip install streamlit scipy numpy pandas
streamlit run app.py
```

Upload `a.csv` and `b.csv`, adjust the sparsity parameter λ, click **Ottimizza**.

## CSV Format

```
# a.csv — expected inventory per (buyer, product)
j,r,value
0,0,12
...

# b.csv — expected production per (producer, product)
i,r,value
0,0,14
...
```

Sample files included in `/data`.

## Output

- Allocation matrix `x[i,j,r]` per product as interactive heatmap
- Full table of non-zero flows

## Stack

- Solver: `scipy.optimize.linprog` (HiGHS)
- Frontend: Streamlit

## License

MIT
