# Customer Churn Analytics & Decision Support Dashboard

**ChurnLens** is an end-to-end customer churn analytics and machine learning dashboard built with **Python, Scikit-learn, Plotly, and Streamlit**.

It transforms customer-level data into churn probabilities, risk segments, model diagnostics, potential retention actions, and an interactive scenario-based ROI analysis.

> **Project type:** Machine Learning + Business Analytics + Decision Support
> **Dataset:** Public Telco Customer Churn dataset
> **Model:** Tuned Logistic Regression
> **Deployment:** Streamlit Community Cloud

---

## 🚀 Live Demo

**[Open ChurnLens →](https://customer-churn-analytics-dashboard-3hgptklf7u78l7kqtcdktt.streamlit.app/)**

**[GitHub Repository →](https://github.com/vedant5964sharma-jpg/customer-churn-analytics-dashboard)**

---

## 📌 Project Overview

Customer churn is a major business problem for subscription-based companies.

The goal of this project is not simply to predict whether a customer will churn. Instead, the project explores how a churn model can be turned into a practical analytics workflow:

```text
Customer Data
      ↓
Data Cleaning & Feature Preparation
      ↓
Exploratory Churn Analysis
      ↓
Machine Learning Model
      ↓
Churn Probability
      ↓
Customer Risk Segmentation
      ↓
Potential Retention Actions
      ↓
Scenario-Based ROI Analysis
      ↓
Interactive Decision-Support Dashboard
```

The result is **ChurnLens**, a Streamlit application designed to help users explore customer risk, understand model behavior, and evaluate potential business scenarios.

---

# 🎯 Business Problem

For a subscription business, losing customers can reduce recurring revenue and increase the cost of acquiring replacements.

A useful churn analytics system should help answer questions such as:

* Which customers have higher estimated churn probability?
* Which customer characteristics are associated with observed churn?
* How does churn vary across contracts, tenure, services, and payment methods?
* What happens when the probability threshold is changed?
* How many customers would be flagged at a particular threshold?
* What could the financial impact look like under different retention assumptions?
* How should model performance and trade-offs be interpreted?

ChurnLens brings these questions together in one dashboard.

---

# 📊 Dataset

The project uses the public **Telco Customer Churn** dataset.

The dataset contains customer information including:

* Customer demographics
* Tenure
* Contract type
* Internet service
* Payment method
* Monthly charges
* Phone and additional services
* Customer churn status

The dataset is used for educational and portfolio purposes.

For real-world deployment, the model would need to be retrained and validated using the organization's own historical customer data.

---

# 🧹 Data Preparation

The modeling workflow includes:

* Converting `TotalCharges` to numeric values
* Handling missing values
* Removing the customer identifier
* Encoding the churn target
* Converting binary categorical variables
* One-hot encoding multi-category variables
* Standardizing numerical features
* Maintaining stratified train/test splits

The final modeling dataset intentionally follows the feature-selection choices implemented in the project.

---

# 🤖 Machine Learning Approach

The project uses **Logistic Regression** as the primary predictive model.

### Why Logistic Regression?

Logistic Regression is particularly useful for this project because it:

* Produces probability estimates
* Is relatively interpretable
* Provides feature coefficients
* Works well as a baseline for binary classification
* Allows threshold-based decision analysis

### Model configuration

The training workflow uses:

* **80/20 stratified train-test split**
* **5-fold Stratified Cross-Validation**
* **ROC-AUC** as the model-selection metric
* **Balanced class weights**
* Hyperparameter tuning for Logistic Regression's `C`
* A decision threshold of **0.40** for the dashboard's risk flagging workflow

The threshold is deliberately separated from the default 0.50 classification boundary so that the dashboard can explore the trade-off between identifying more potential churners and generating more false alarms.

---

# 📈 Model Performance

The recorded hold-out evaluation for the current project is:

| Metric             |    Result |
| ------------------ | --------: |
| ROC-AUC            | **0.838** |
| Recall             |  **0.86** |
| Precision          |  **0.47** |
| Decision Threshold |  **0.40** |

### How to interpret this

The model is designed with relatively high recall at the selected threshold.

That means the system attempts to identify a larger proportion of customers who actually churned, while accepting that some customers who would not churn may also be flagged.

This is why **accuracy alone is not used as the primary measure of the system**.

The dashboard also provides threshold analysis, ROC information, precision/recall analysis, and a confusion matrix so the trade-offs can be inspected rather than hidden.

---

# 🖥️ Dashboard Features

## 1. Command Center

Provides a high-level view of:

* Customer population
* Churn distribution
* Risk distribution
* Tenure patterns
* Contract behavior
* Key dashboard KPIs

---

## 2. Risk Scorer

Allows individual customer information to be evaluated through the trained model.

The page displays:

* Estimated churn probability
* Risk classification
* Important contributing factors
* Potential retention action context

The score should be interpreted as a **model estimate**, not a certainty that a customer will churn.

---

## 3. Action List

Creates a prioritized view of customers based on estimated churn risk.

The page is designed to answer:

> "Which customers could be investigated first?"

The table can also be exported for further analysis.

---

## 4. Retention ROI

The dashboard includes an interactive scenario simulator.

Users can change assumptions such as:

* Offer cost
* Estimated save rate
* Months of revenue protected
* Risk threshold

The tool then estimates the potential net value under those assumptions.

### Important

These ROI calculations are **scenario assumptions**, not causal estimates.

The dashboard does not claim that a particular retention intervention will actually prevent churn.

---

## 5. Customer Segments

Explore observed churn patterns across groups such as:

* Contract type
* Internet service
* Tenure
* Payment method
* Customer characteristics

The dashboard also provides a contract × internet-service view for deeper segmentation analysis.

---

## 6. Model Performance

Provides model diagnostics including:

* ROC analysis
* Cumulative gains
* Precision/recall by threshold
* Confusion matrix
* Threshold trade-offs

This section is designed to make the model's limitations visible instead of treating the prediction as a black box.

---

## 7. Methodology

Documents:

* Dataset
* Feature preparation
* Model selection
* Evaluation approach
* Threshold choice
* Business assumptions
* Project limitations

---

# 💡 Example Business Observations

The dataset allows several useful observations to be explored.

For example, churn can be compared across:

* Short-term vs. long-term contracts
* Different internet-service groups
* Customer tenure
* Monthly charge levels
* Payment methods
* Additional support services

These are **observed relationships within the dataset** and should not automatically be interpreted as causal relationships.

For example:

> Customers on month-to-month contracts show higher observed churn in this dataset.

This does not by itself prove that changing a customer's contract would prevent churn.

---

# 🏗️ Project Structure

```text
customer-churn-analytics-dashboard/
│
├── app.py                  # Streamlit dashboard
├── model.py                # Data preparation and ML pipeline
├── train.py                # Model training / artifact utilities
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore
│
├── data/
│   └── telco_churn.csv     # Public dataset
│
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

---

# 🛠️ Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Stratified Cross-Validation
* Hyperparameter Tuning
* ROC-AUC
* Precision / Recall analysis

### Visualization

* Plotly

### Application

* Streamlit

### Version Control

* Git
* GitHub

---

# ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/vedant5964sharma-jpg/customer-churn-analytics-dashboard.git
```

Enter the project:

```bash
cd customer-churn-analytics-dashboard
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

The application trains the model when it starts and uses the bundled dataset.

---

# ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud** directly from the GitHub repository.

The repository contains:

* `app.py` as the application entry point
* `requirements.txt` for Python dependencies
* `.streamlit/config.toml` for application configuration
* `data/telco_churn.csv` for the bundled dataset

Streamlit Community Cloud supports deploying directly from GitHub repositories and automatically updates the deployed application when committed changes are detected.

---

# ⚠️ Limitations

This project is a **portfolio and analytical prototype**, not a production churn-management system.

Important limitations include:

* The dataset is publicly available rather than proprietary business data.
* Model performance may differ substantially on another company's customer base.
* The model identifies statistical patterns; it does not establish causality.
* Retention actions are potential actions for analysis, not guaranteed interventions.
* ROI calculations depend on user-defined assumptions.
* Production deployment would require monitoring, calibration, retraining, and validation over time.
* Customer-level scores should be interpreted alongside business context rather than treated as definitive outcomes.

---

# 🔮 Future Improvements

Potential extensions include:

* Probability calibration
* Automated model monitoring
* Model drift detection
* SHAP-based explainability
* Additional model benchmarking
* Temporal validation
* Automated retraining
* Database-backed customer data
* Authentication and role-based access
* API-based scoring
* A/B testing of retention interventions
* Integration with CRM systems

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

* End-to-end machine learning workflows
* Binary classification
* Feature preprocessing
* Class imbalance
* Cross-validation
* Hyperparameter tuning
* Probability thresholds
* Model evaluation
* Customer segmentation
* Business-oriented analytics
* Interactive data visualization
* Streamlit application development
* Git and GitHub
* Cloud deployment

---

## 👨‍💻 Author

**Vedant Sharma**

B.Tech Computer Science & Engineering
Symbiosis Institute of Technology, Pune

---

## ⭐ Project

If you found this project useful, consider giving the repository a star.
