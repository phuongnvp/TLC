# TLC Studio

Streamlit prediction of thin-layer chromatography Rf from molecular SMILES and solvent proportions. Select solvents by name and enter relative parts. The app normalizes the mixture, runs both CPU checkpoints, and displays each prediction and their weighted mean in a session-local conversation.

## Deployment

Use the main branch, entry point `app.py`, and Python 3.12 on Streamlit Community Cloud. Locally, install requirements with `pip install -r requirements.txt` and run `streamlit run app.py`.

Both trained checkpoints are bundled. The app assembles the five AttentiveFP parts at startup, verifies both SHA-256 checksums and caches the loaded models.

## Natural products

The separate Natural products page searches only n-Butanol, Acetic acid, Acetone, Chloroform, Diethyl ether, Ethyl acetate, Formic acid, n-Hexane, NH3, Toluene, Triethylamine and Water, when supported by both checkpoints. Acetic acid and formic acid cannot appear together; NH3 and triethylamine cannot appear together; acids cannot appear with bases. These restrictions apply before candidate mixtures are scored, including all exported results. Checkpoint solvent columns are preserved; 1-Butanol is displayed as n-Butanol. The page retains top-10 results and the full evaluated-mixture CSV download.

## Find molecules

Place the complete `coconut_embeddings` folder beside `app.py`, or set
`TLC_EMBEDDINGS_DIR` to its absolute path. The **Find molecules** page accepts
mobile-phase solvents, relative parts, and a target Rf. It searches every embedded
COCONUT structure and returns predictions within target Rf ±0.10, ranked by
absolute difference. The table shows up to 1,000 rows; the CSV includes all matches,
individual model predictions, and the query conditions.

The database must have `summary.json` with `complete: true`. On first use, the app
verifies chunk checksums, row alignment, encoder/model compatibility, and direct
predictions for three stored molecules. Validation is cached. Queries map one
embedding chunk at a time; they do not load the full 2.11 GiB into memory. Query
results and CSV exports still use memory proportional to the number of matches.
The ±0.10 selection window is not an experimental confidence interval.

Run a standalone CPU query and measure actual runtime with:

```bash
python molecule_search.py --solvents n-Hexane "Ethyl acetate" --ratios 7:3 --target 0.5 --threads 4 --output coconut_matches.csv
```

Run the search tests with `python -m unittest discover -s tests`.

## Evaluation

Five-fold pooled out-of-fold MSE on 21,432 training rows selected 0.5774426289184875 for AttentiveFP and the remaining weight for MPNN. On 5,358 held-out rows, ensemble RMSE was 0.118822 versus 0.126481 for AttentiveFP, 0.130431 for MPNN and 0.118914 for equal weighting. This is the project's original random split.

[Evaluation notebook](https://colab.research.google.com/drive/1fwnP4SJQGWoywN08d1jySOdt_jIg_JMO) · [Inference notebook](https://colab.research.google.com/drive/1n0orAwMN7i8ruWvAdBz9CAmYVkUGCfu5)

