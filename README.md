# ChurnLens: customer churn analytics

A Streamlit dashboard that turns a tuned churn model into decisions: who to call, what it is worth, and how far to trust the score.

## What's inside
| Page | What it answers |
|---|---|
| Command center | Where is revenue leaking? Headline KPIs, tenure and contract effects, risk distribution |
| Risk scorer | How likely is this customer to leave, what drives it, and what to offer |
| Action list | Which active customers should the retention team call first (CSV export) |
| Retention ROI | What threshold maximizes net value given offer cost and save rate |
| Segments | Churn by group, plus the contract x internet-service heatmap |
| Model performance | ROC, cumulative gains, precision/recall by threshold, confusion matrix, an honest read of the trade-offs |
| Methodology | Data, preparation, model and limits |

## Model
Logistic regression (balanced class weights, C tuned by 5-fold stratified CV on ROC-AUC), flagged at a 0.40 probability threshold.
Hold-out ROC-AUC 0.838, recall 0.86, precision 0.47. The training code in `model.py` and `train.py` is unchanged; `app.py` only presents it.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The model trains on first launch (a few seconds) from the bundled `data/telco_churn.csv`, so no network access or pickle files are needed.
`python train.py` still works if you want to save artifacts.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Create a new app on Streamlit Community Cloud and choose `app.py` as the entry point.
3. Deploy. No secrets are required.

## Notes
- The dataset is the public Telco Customer Churn sample. Retrain on your own history before using scores for real decisions, and add monitoring and calibration.
- The Retention ROI page uses adjustable assumptions (offer cost, save rate, months of revenue protected). Defaults are illustrative.
