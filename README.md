# Portfolio-default-rate

Mining actual bank data from Italy to calculate the portfolio default rate. The model intends to provide better predictions of the one-year probability of default (PD) for prospective borrowers.

## Running

To run the model as a local application, git clone this repository, and run the `app.py` file. To run the model in a Python environment to predict the probability of default, you can run the `harness.py` file.

## Code

The Jupyter notebook `val.ipynb` contains all the steps for cleaning, Exploratory Data Analysis (EDA), and training of the model. The XGB model checkpoint is also uploaded as `turquoise_model.bin`.

## PitchDeck

This project was part of the Applied ML in Finance class for CDS NYU and NYU Stern, and the presentation file explaining all the steps in the project is in `Turquoise_PitchDeck.pdf`.

## Methodology

The methodology includes the following key steps:

- Cleaned and imputed data using different financial ratios. More details are provided in the write-up.
- Defined the firm year from July to June to avoid future-peeking issues.
- Performed sanity checks to account for erroneous data due to accounting factors, etc.
- Analyzed key financial ratios by dividing them into buckets of liquidity, profitability, debt coverage, and asset management.
- Performed univariate and multivariate analysis using Variance Inflation Factor (ViF) and Correlation analysis to avoid multicollinearity in the independent variables.
- Developed a logit model and XGB model by training through a walk-forward analysis.
- Benchmarked the results based on the Altman Z-Score.

## Results

The results are as follows:

- [Insert figure here]
- AUC Score table for Logit, XGB, and Z-Score.
- Performed calibration to match real-world probabilities.
- Focused on the explainability of the model using stats models for Logit and weight, gain, and cover importance for XGB model.
