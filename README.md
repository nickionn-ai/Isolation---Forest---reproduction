# Isolation Forest – Annthyroid Experiment

This repository contains the Jupyter notebook for **Task 2: Reproducing the Results** of the Data Wrangling individual assignment at Utrecht University.

The goal is to reproduce the Isolation Forest experiment on the Annthyroid dataset from Liu, Ting and Zhou (2008), and compare our results with those reported in the paper.

## Method

We use the `IsolationForest` implementation from scikit-learn rather than implementing the algorithm from scratch. The model is trained without class labels, and its anomaly scores are evaluated using ROC AUC.

The experiment uses:
- 100 isolation trees (`n_estimators=100`)
- 256 observations per tree (`max_samples=256`)
- Six continuous features from Annthyroid
- `random_state=0` for the initial run, followed by seeds 0–9 for ten runs
- scikit-learn version 1.7.2 for the reported results

Other `IsolationForest` options are left at their scikit-learn defaults. We do not apply additional feature scaling.

## Dataset

The data comes from the **UCI Machine Learning Repository – Thyroid Disease** dataset. Download the Annthyroid files:

- `ann-train.data`
- `ann-test.data`

Place both files in the **same folder as the notebook** before running it. The notebook loads them by filename and combines them.

The combined files contain **7,200 observations**. We use the six continuous input features (columns 16–21 in one-based numbering) and the final column as the class label. Classes 1 and 2 are treated as anomalies; class 3 is treated as normal. This gives **534 anomalies** and **6,666 normal observations**.

The original paper reports **6,832 observations** for Annthyroid. We keep all 7,200 records from the two UCI files rather than removing records without a documented reason. This difference is discussed in the report.

## How to run

1. Install Python and Jupyter Notebook (or use an existing Jupyter environment).
2. Install the required packages:

   ```bash
   pip install numpy scikit-learn==1.7.2 notebook
   ```

3. Put `Isolation_Forest_Task2.ipynb`, `ann-train.data`, and `ann-test.data` in the same folder.
4. Open the notebook:

   ```bash
   jupyter notebook Isolation_Forest_Task2.ipynb
   ```

5. Select **Run All** to execute the cells in order.

The notebook loads and prepares the data, trains the Isolation Forest, calculates ROC AUC, and repeats the experiment with ten random seeds. No class labels are used during training. The notebook fits and evaluates the model on the same combined dataset.

## Results

| Experiment | ROC AUC |
| --- | ---: |
| Original paper (Annthyroid) | 0.8200 |
| Our initial run (seed 0) | 0.7991 |
| Mean of 10 runs (seeds 0–9) | 0.8000 |
| Standard deviation across runs | 0.0142 |

The AUC across the ten runs ranges from **0.7778 to 0.8262**. Differences from the paper may be related to the number of observations, implementation details, and random sampling and splitting. The reported AUC describes anomaly ranking on the data used for fitting; it is not a held-out test-set result.

## Reference

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. IEEE International Conference on Data Mining (ICDM).
