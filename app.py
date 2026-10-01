"""ChurnLens - customer churn analytics dashboard.

Presentation layer only. All modelling lives in model.py / train.py and is used as-is.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from model import (
    CATEGORICAL_COLS,
    align_for_model,
    coefficient_drivers,
    load_data,
    prepare_features,
    train_model,
)

st.set_page_config(
    page_title="ChurnLens | Customer churn analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────── design tokens ─────────────────────────────
INK, SLATE, LINE = "#12202F", "#5B6B7C", "#E3E8EE"
BLUE, RISK, AMBER = "#1F6F9F", "#D64550", "#E9A23B"
BLUE_SOFT, RISK_SOFT = "#E6F0F7", "#FBE7E9"
HEAD = "Bricolage Grotesque, Instrument Sans, system-ui, sans-serif"
BODY = "Instrument Sans, system-ui, sans-serif"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
.stApp, .stMarkdown, .stApp p, .stApp label {{ font-family: 'Instrument Sans', system-ui, sans-serif; }}
.stApp h1, .stApp h2, .stApp h3 {{ font-family: 'Bricolage Grotesque', 'Instrument Sans', sans-serif; letter-spacing: -0.015em; color: {INK}; }}
.block-container {{ max-width: 1280px; padding-top: 2.4rem; padding-bottom: 4rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, .stDeployButton {{ visibility: hidden; display: none; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}

.cl-head h1 {{ font-size: 2.3rem !important; font-weight: 700 !important; margin: 0 !important; padding: 0 !important; line-height: 1.1 !important; }}
.cl-head p {{ color: {SLATE}; font-size: 1.05rem; margin: .45rem 0 0 0; }}

.cl-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); background: #fff;
  border: 1px solid {LINE}; border-radius: 10px; margin: 1.3rem 0 1.6rem; overflow: hidden; }}
.cl-cell {{ padding: 1.05rem 1.3rem; border-left: 1px solid {LINE}; }}
.cl-cell:first-child {{ border-left: none; }}
.cl-k {{ color: {SLATE}; font-size: .85rem; font-weight: 500; }}
.cl-v {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.95rem; font-weight: 700; line-height: 1.2; margin: .1rem 0; color: {INK}; }}
.cl-n {{ color: {SLATE}; font-size: .82rem; line-height: 1.35; }}
.cl-cell.risk .cl-v {{ color: {RISK}; }}
.cl-cell.blue .cl-v {{ color: {BLUE}; }}

.cl-note {{ border-left: 3px solid {BLUE}; padding: .15rem 0 .15rem .95rem; margin: .3rem 0 1.1rem; font-size: .95rem; line-height: 1.5; color: {INK}; }}
.cl-note.risk {{ border-left-color: {RISK}; }}
.cl-note.amber {{ border-left-color: {AMBER}; }}

[data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: .35rem; }}
[data-testid="stForm"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button {{ border-radius: 8px; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────── feature vocabulary ─────────────────────────────
# feature -> (neutral name, phrase when value is above average / present, phrase when below / absent)
FEATURES = {
    "tenure": ("Tenure (months)", "Long tenure", "Short tenure"),
    "MonthlyCharges": ("Monthly charges", "High monthly charges", "Low monthly charges"),
    "SeniorCitizen": ("Senior citizen", "Senior citizen", "Not a senior citizen"),
    "gender": ("Male", "Male", "Female"),
    "Partner": ("Has partner", "Has a partner", "No partner"),
    "Dependents": ("Has dependents", "Has dependents", "No dependents"),
    "PhoneService": ("Phone service", "Has phone service", "No phone service"),
    "MultipleLines": ("Multiple lines", "Multiple lines", "Single line"),
    "OnlineSecurity": ("Online security", "Has online security", "No online security"),
    "OnlineBackup": ("Online backup", "Has online backup", "No online backup"),
    "DeviceProtection": ("Device protection", "Has device protection", "No device protection"),
    "TechSupport": ("Tech support", "Has tech support", "No tech support"),
    "StreamingTV": ("Streaming TV", "Streams TV", "No TV streaming"),
    "StreamingMovies": ("Streaming movies", "Streams movies", "No movie streaming"),
    "PaperlessBilling": ("Paperless billing", "Paperless billing", "Paper billing"),
    "InternetService_Fiber optic": ("Fiber optic internet", "On fiber optic", "Not on fiber optic"),
    "InternetService_No": ("No internet service", "No internet service", "Has internet service"),
    "Contract_One year": ("One-year contract", "On a one-year contract", "No one-year contract"),
    "Contract_Two year": ("Two-year contract", "On a two-year contract", "No two-year contract"),
    "PaymentMethod_Credit card (automatic)": ("Pays by credit card (auto)", "Pays by credit card", "Not paying by credit card"),
    "PaymentMethod_Electronic check": ("Pays by electronic check", "Pays by electronic check", "Not paying by e-check"),
    "PaymentMethod_Mailed check": ("Pays by mailed check", "Pays by mailed check", "Not paying by mailed check"),
}

# feature -> (sign of the standardized value that makes the play relevant, suggested play)
PLAYS = {
    "Contract_Two year": (-1, "Offer a one- or two-year contract with a loyalty discount"),
    "Contract_One year": (-1, "Offer a one- or two-year contract with a loyalty discount"),
    "TechSupport": (-1, "Bundle tech support free for the first three months"),
    "OnlineSecurity": (-1, "Add online security as a trial perk"),
    "tenure": (-1, "Schedule an early-life check-in call"),
    "InternetService_Fiber optic": (1, "Run a proactive speed and service-quality check"),
    "PaymentMethod_Electronic check": (1, "Move the customer to auto-pay with a small bill credit"),
    "MonthlyCharges": (1, "Review plan fit and offer a price lock"),
}


def label(feature: str) -> str:
    return FEATURES.get(feature, (feature,))[0]


def phrase(feature: str, z: float) -> str:
    entry = FEATURES.get(feature)
    if entry is None:
        return feature
    return entry[1] if z > 0 else entry[2]


# ───────────────────────────── UI helpers ─────────────────────────────
_PLOT_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
_DF_WIDTH = "width" in inspect.signature(st.dataframe).parameters


def plot(fig: go.Figure, key: str | None = None) -> None:
    cfg = {"displayModeBar": False}
    if _PLOT_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key, config=cfg)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)


def table(df: pd.DataFrame, **kwargs) -> None:
    if _DF_WIDTH:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def style(fig: go.Figure, height: int = 340, title: str | None = None, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.02, xanchor="left", font=dict(family=HEAD, size=16, color=INK)) if title else None,
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        font=dict(family=BODY, color=INK, size=13),
        margin=dict(l=16, r=16, t=56 if title else 16, b=16),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0),
        hoverlabel=dict(font_family=BODY),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=SLATE))
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=SLATE))
    return fig


def head(title: str, sub: str) -> None:
    st.markdown(f'<div class="cl-head"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def kpi_strip(cells: list[tuple[str, str, str, str]]) -> None:
    body = "".join(
        f'<div class="cl-cell {tone}"><div class="cl-k">{k}</div><div class="cl-v">{v}</div><div class="cl-n">{n}</div></div>'
        for k, v, n, tone in cells
    )
    st.markdown(f'<div class="cl-strip">{body}</div>', unsafe_allow_html=True)


def note(text: str, tone: str = "") -> None:
    st.markdown(f'<div class="cl-note {tone}">{text}</div>', unsafe_allow_html=True)


# ───────────────────────────── data and model ─────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "telco_churn.csv"


@st.cache_data(show_spinner=False)
def get_raw() -> pd.DataFrame:
    # A bundled copy keeps the app working offline; fall back to the notebook's public URL.
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return load_data()


@st.cache_resource(show_spinner="Training the model on first launch…")
def get_artifacts():
    return train_model(get_raw())


@st.cache_data(show_spinner=False)
def get_holdout() -> dict:
    """Rebuild the exact 20% hold-out split used by train_model (same seed, same stratification)."""
    raw, art = get_raw(), get_artifacts()
    prepared = prepare_features(raw, keep_target=True)
    X, y = prepared.drop(columns=["Churn"]), prepared["Churn"].astype(int)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    proba = art.model.predict_proba(art.scaler.transform(X_te[art.feature_names]))[:, 1]
    return {"y": y_te.to_numpy(), "p": proba, "monthly": raw.loc[X_te.index, "MonthlyCharges"].to_numpy()}


@st.cache_data(show_spinner=False)
def get_scored() -> pd.DataFrame:
    """Score the whole customer base and attach each customer's strongest risk factor."""
    raw, art = get_raw(), get_artifacts()
    X = align_for_model(prepare_features(raw), art.feature_names)
    Z = art.scaler.transform(X)
    contrib = Z * art.model.coef_[0]
    out = raw.copy()
    out["risk"] = art.model.predict_proba(Z)[:, 1]
    out["flag"] = out["risk"] >= art.threshold
    # Demographics and price are collinear with service choices, so they are not offered as "main risk factors".
    skip = {"gender", "SeniorCitizen", "Partner", "Dependents", "MonthlyCharges"}
    masked = contrib.copy()
    masked[:, [j for j, f in enumerate(art.feature_names) if f in skip]] = -np.inf
    top = masked.argmax(axis=1)
    out["top_factor"] = [phrase(art.feature_names[j], Z[i, j]) for i, j in enumerate(top)]
    return out


def score_customer(customer: pd.DataFrame):
    """Score one customer.

    prepare_features() uses get_dummies(drop_first=True), which silently drops the only category of a
    single-row frame, so a lone customer loses Contract / InternetService / PaymentMethod. Appending
    the customer to reference rows that contain every category level avoids that.
    """
    raw, art = get_raw(), get_artifacts()
    ref = raw.drop_duplicates(subset=CATEGORICAL_COLS).drop(columns=["customerID", "Churn", "TotalCharges"], errors="ignore")
    combined = pd.concat([ref, customer], ignore_index=True)
    X = align_for_model(prepare_features(combined), art.feature_names).iloc[[-1]]
    Z = art.scaler.transform(X)
    prob = float(art.model.predict_proba(Z)[0, 1])
    return prob, Z[0], Z[0] * art.model.coef_[0]


def net_value(y, p, monthly, threshold, cost, save_rate, months):
    flagged = p >= threshold
    reached = flagged & (y == 1)
    gain = float((monthly * months * save_rate)[reached].sum())
    return gain - cost * int(flagged.sum()), int(flagged.sum()), int(reached.sum())


art = get_artifacts()
THR = art.threshold


# ───────────────────────────── pages ─────────────────────────────
def page_overview():
    raw, sc = get_raw(), get_scored()
    churned = raw["Churn"].eq("Yes")
    active = sc[~churned]
    flagged = active[active["flag"]]
    head("Churn command center", "Where revenue is leaking, and which active customers need a call first.")
    kpi_strip([
        ("Customers", f"{len(raw):,}", f"{raw['tenure'].mean():.0f} months average tenure", ""),
        ("Observed churn", f"{churned.mean():.1%}", f"{int(churned.sum()):,} customers have left", "risk"),
        ("Monthly revenue lost", f"${raw.loc[churned, 'MonthlyCharges'].sum():,.0f}",
         f"${raw.loc[churned, 'MonthlyCharges'].mean():.0f} a month per churned customer", "risk"),
        ("Active customers flagged", f"{len(flagged):,}",
         f"{len(flagged) / len(active):.0%} of the active base, ${flagged['MonthlyCharges'].sum():,.0f} a month exposed", "blue"),
    ])

    rate = lambda col: raw.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean())
    by_contract, by_net = rate("Contract"), rate("InternetService")
    first_year = churned[raw["tenure"] <= 12].mean()
    later = churned[raw["tenure"] > 12].mean()
    n1, n2, n3 = st.columns(3)
    with n1:
        note(f"<b>Contracts decide retention.</b> Month-to-month customers churn at {by_contract['Month-to-month']:.0%}; "
             f"two-year customers at {by_contract['Two year']:.0%}.", "risk")
    with n2:
        note(f"<b>The first year is the danger zone.</b> {first_year:.0%} of customers in their first 12 months leave, "
             f"against {later:.0%} afterwards.", "risk")
    with n3:
        note(f"<b>Fiber customers leave more.</b> Fiber optic churn is {by_net['Fiber optic']:.0%}, DSL is {by_net['DSL']:.0%}, "
             f"no internet is {by_net['No']:.0%}.", "amber")

    left, right = st.columns(2)
    with left:
        tmp = raw.assign(group=pd.cut(raw["tenure"], [0, 12, 24, 36, 48, 60, 72],
                                      labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"], include_lowest=True))
        rates = tmp.groupby("group", observed=False)["Churn"].apply(lambda s: (s == "Yes").mean())
        fig = go.Figure(go.Bar(x=rates.index.astype(str), y=rates.values, marker_color=[RISK, "#E0707A", "#E99BA2", "#F0BFC4", "#F5D5D8", "#F9E6E8"],
                               text=[f"{v:.0%}" for v in rates.values], textposition="outside", cliponaxis=False, hovertemplate="%{x} months: %{y:.1%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", range=[0, rates.max() * 1.18])
        fig.update_xaxes(title="Tenure in months")
        plot(style(fig, 360, "Churn rate by tenure"))
    with right:
        below, above = active[~active["flag"]]["risk"], active[active["flag"]]["risk"]
        fig = go.Figure()
        for data, name, color in [(below, "Below threshold", BLUE), (above, "Flagged", RISK)]:
            fig.add_histogram(x=data, name=name, marker_color=color, xbins=dict(start=0, end=1, size=0.05),
                              hovertemplate="Risk %{x:.0%}: %{y} customers<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.06)
        fig.add_vline(x=THR, line_color=INK, line_width=2, annotation_text=f"Threshold {THR:.2f}", annotation_position="top")
        fig.update_xaxes(tickformat=".0%", title="Predicted churn risk")
        fig.update_yaxes(title="Active customers")
        plot(style(fig, 360, "Risk across active customers", legend=True))

    left, right = st.columns(2)
    with left:
        r = by_contract.sort_values()
        fig = go.Figure(go.Bar(x=r.values, y=r.index, orientation="h", marker_color=[BLUE, "#6FA3C4", RISK],
                               text=[f"{v:.1%}" for v in r.values], textposition="outside", cliponaxis=False))
        fig.update_xaxes(tickformat=".0%", range=[0, r.max() * 1.2])
        plot(style(fig, 330, "Churn rate by contract"))
    with right:
        d = coefficient_drivers(art, 8).sort_values("coefficient")
        fig = go.Figure(go.Bar(x=d["coefficient"], y=[label(f) for f in d["feature"]], orientation="h",
                               marker_color=[RISK if c > 0 else BLUE for c in d["coefficient"]],
                               hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.update_xaxes(title="Effect on churn (red raises risk, blue lowers it)")
        plot(style(fig, 330, "What the model weighs most"))


def page_scorer():
    head("Risk scorer", "Describe a customer to see their churn risk and what is driving it.")
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.form("customer"):
            a, b = st.columns(2)
            with a:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                phone = st.selectbox("Phone service", ["Yes", "No"])
                multiple = st.selectbox("Multiple lines", ["No", "Yes"])
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                security = st.selectbox("Online security", ["No", "Yes"])
                backup = st.selectbox("Online backup", ["No", "Yes"])
            with b:
                protection = st.selectbox("Device protection", ["No", "Yes"])
                support = st.selectbox("Tech support", ["No", "Yes"])
                tv = st.selectbox("Streaming TV", ["No", "Yes"])
                movies = st.selectbox("Streaming movies", ["No", "Yes"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless billing", ["No", "Yes"])
                payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                monthly = st.number_input("Monthly charges ($)", 0.0, 150.0, 70.0, 1.0)
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            st.form_submit_button("Update risk score", type="primary")
        st.caption("Add-on services are treated as 'No' when the customer has no internet, and extra lines when there is no phone service.")

    if internet == "No":
        security = backup = protection = support = tv = movies = "No"
    if phone == "No":
        multiple = "No"
    customer = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    }])
    prob, Z, contrib = score_customer(customer)
    flag = prob >= THR

    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number=dict(suffix="%", valueformat=".0f", font=dict(family=HEAD, size=56, color=RISK if flag else BLUE)),
            gauge=dict(axis=dict(range=[0, 100], tickwidth=0, tickfont=dict(size=11, color=SLATE)),
                       bar=dict(color=RISK if flag else BLUE, thickness=0.3), bgcolor="#fff", borderwidth=0,
                       steps=[dict(range=[0, THR * 100], color=BLUE_SOFT), dict(range=[THR * 100, 100], color=RISK_SOFT)],
                       threshold=dict(line=dict(color=INK, width=3), thickness=0.9, value=THR * 100))))
        plot(style(fig, 270, "Estimated churn risk"))
        if flag:
            note(f"<b>Flagged for outreach.</b> Risk is above the {THR:.0%} threshold. At ${monthly:,.0f} a month, "
                 f"this customer is worth contacting.", "risk")
        else:
            note(f"<b>No action flagged.</b> Risk is below the {THR:.0%} threshold.")

        order = np.argsort(-np.abs(contrib))[:8][::-1]
        names = [phrase(art.feature_names[i], Z[i]) for i in order]
        fig = go.Figure(go.Bar(x=contrib[order], y=names, orientation="h",
                               marker_color=[RISK if contrib[i] > 0 else BLUE for i in order],
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.update_xaxes(title="Effect on risk vs. the average customer (log-odds)")
        plot(style(fig, 360, "What moves this score"))

    plays, seen = [], set()
    for i in np.argsort(-contrib):
        feat = art.feature_names[i]
        if contrib[i] <= 0 or feat not in PLAYS:
            continue
        sign, text = PLAYS[feat]
        if np.sign(Z[i]) == sign and text not in seen:
            plays.append(text)
            seen.add(text)
    if flag and plays:
        st.subheader("Suggested plays")
        for text in plays[:3]:
            note(text)
        st.caption("Plays are rule-based suggestions tied to the model's strongest risk factors. They show association, not proven cause.")


def page_segments():
    raw = get_raw()
    head("Segment explorer", "Compare churn across customer groups and find the combinations that hurt most.")
    options = {"Contract": "Contract", "Internet service": "InternetService", "Tech support": "TechSupport",
               "Payment method": "PaymentMethod", "Partner": "Partner", "Dependents": "Dependents",
               "Paperless billing": "PaperlessBilling", "Gender": "gender", "Senior citizen": "SeniorCitizen"}
    choice = st.selectbox("Compare churn by", list(options))
    col = options[choice]
    g = raw.groupby(col).agg(customers=("Churn", "size"), churn=("Churn", lambda s: (s == "Yes").mean())).reset_index()
    g[col] = g[col].astype(str)
    g = g.sort_values("churn")
    overall = raw["Churn"].eq("Yes").mean()
    a, b = st.columns([3, 2])
    with a:
        fig = go.Figure(go.Bar(x=g["churn"], y=g[col], orientation="h", marker_color=[RISK if v > overall else BLUE for v in g["churn"]],
                               text=[f"{v:.1%}" for v in g["churn"]], textposition="outside", cliponaxis=False,
                               customdata=g["customers"], hovertemplate="%{y}: %{x:.1%} of %{customdata:,} customers<extra></extra>"))
        fig.add_vline(x=overall, line_dash="dot", line_color=INK, annotation_text=f"All customers {overall:.1%}", annotation_position="bottom right")
        fig.update_xaxes(tickformat=".0%", range=[0, g["churn"].max() * 1.22])
        plot(style(fig, 360, f"Churn rate by {choice.lower()}"))
    with b:
        view = g.sort_values("churn", ascending=False).rename(columns={col: choice, "customers": "Customers", "churn": "Churn rate"})
        view["Churn rate"] = view["Churn rate"].map(lambda v: f"{v:.1%}")
        view["Customers"] = view["Customers"].map(lambda v: f"{v:,}")
        st.markdown("&nbsp;")
        table(view)
        st.caption("Red bars are above the overall churn rate; blue bars are below it.")

    pivot = pd.crosstab(raw["InternetService"], raw["Contract"], values=raw["Churn"].eq("Yes"), aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=[[0, "#F4F6F9"], [1, RISK]],
                               text=[[f"{v:.1%}" for v in row] for row in pivot.values], texttemplate="%{text}", showscale=False,
                               hovertemplate="%{y}, %{x}: %{z:.1%}<extra></extra>"))
    plot(style(fig, 300, "Churn rate by contract and internet service"))


def page_actions():
    sc = get_scored()
    active = sc[sc["Churn"].eq("No")].copy()
    head("Action list", "Active customers ranked by churn risk, ready for the retention team.")
    c1, c2 = st.columns([2, 3])
    with c1:
        min_risk = st.slider("Minimum risk", 0.0, 1.0, float(THR), 0.05, format="%.2f")
    with c2:
        contracts = st.multiselect("Contract", sorted(active["Contract"].unique()), default=sorted(active["Contract"].unique()))
    view = active[(active["risk"] >= min_risk) & active["Contract"].isin(contracts)].sort_values("risk", ascending=False)
    kpi_strip([
        ("Customers on the list", f"{len(view):,}", f"of {len(active):,} active customers", ""),
        ("Monthly revenue exposed", f"${view['MonthlyCharges'].sum():,.0f}", f"${view['MonthlyCharges'].mean() if len(view) else 0:.0f} average per customer", "risk"),
        ("Average tenure", f"{view['tenure'].mean() if len(view) else 0:.0f} months", "How long these customers have stayed so far", "blue"),
    ])
    out = pd.DataFrame({
        "Customer": view["customerID"], "Risk score": (view["risk"] * 100).round(0), "Main risk factor": view["top_factor"],
        "Contract": view["Contract"], "Tenure": view["tenure"], "Monthly $": view["MonthlyCharges"],
        "Internet": view["InternetService"], "Payment": view["PaymentMethod"],
    })
    if out.empty:
        st.info("No active customers match these filters. Lower the minimum risk or add a contract type.")
        return
    table(out, height=520, column_config={
        "Risk score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d"),
        "Monthly $": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
        "Tenure": st.column_config.NumberColumn("Tenure (mo)"),
    })
    st.download_button("Download this list as CSV", out.to_csv(index=False).encode(), "churn_action_list.csv", "text/csv", type="primary")
    st.caption("The model was trained on this same customer base, so the ranking is illustrative. In production, score customers the model has not seen.")


def page_roi():
    h = get_holdout()
    y, p, m = h["y"], h["p"], h["monthly"]
    head("Retention ROI", "Pick the risk threshold by money, not by habit. Adjust the assumptions to match your business.")
    a, b, c = st.columns(3)
    cost = a.slider("Cost per contacted customer ($)", 5, 100, 25, 5, help="Agent time plus the discount or perk you offer.")
    save = b.slider("Share of contacted churners you save", 5, 60, 30, 5, format="%d%%") / 100
    months = c.slider("Months of revenue protected", 3, 24, 12, 1)

    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    nets = np.array([net_value(y, p, m, t, cost, save, months)[0] for t in grid])
    best = int(np.argmax(nets))
    best_t, best_net = float(grid[best]), float(nets[best])
    dep_net, dep_n, dep_hit = net_value(y, p, m, THR, cost, save, months)
    all_net = net_value(y, p, m, 0.0, cost, save, months)[0]
    total_churn = int((y == 1).sum())

    kpi_strip([
        ("Customers contacted", f"{dep_n:,}", f"{dep_n / len(y):.0%} of the {len(y):,}-customer hold-out set", ""),
        ("Churners reached", f"{dep_hit:,}", f"{dep_hit / total_churn:.0%} of the {total_churn} who actually left", "blue"),
        ("Net value at 0.40", f"${dep_net:,.0f}", f"${dep_net / len(y):,.0f} per customer scored", "blue" if dep_net >= 0 else "risk"),
        ("Best threshold", f"{best_t:.2f}", f"${best_net:,.0f} net, {best_net - dep_net:+,.0f} vs the deployed threshold", ""),
    ])

    fig = go.Figure()
    fig.add_scatter(x=grid, y=nets, mode="lines", line=dict(color=BLUE, width=3), fill="tozeroy", fillcolor=BLUE_SOFT,
                    hovertemplate="Threshold %{x:.2f}: $%{y:,.0f}<extra></extra>")
    fig.add_vline(x=THR, line_color=INK, line_dash="dot", annotation_text="Deployed 0.40", annotation_position="top left")
    fig.add_scatter(x=[best_t], y=[best_net], mode="markers", marker=dict(color=RISK, size=13, line=dict(color="#fff", width=2)),
                    hovertemplate=f"Best: {best_t:.2f}<extra></extra>")
    fig.update_xaxes(title="Risk threshold (customers at or above it get a retention offer)")
    fig.update_yaxes(title="Net value on hold-out set ($)", tickprefix="$")
    plot(style(fig, 380, "Net value of the retention program by threshold"))

    names = ["Contact nobody", "Contact everyone", "Model at 0.40", f"Model at {best_t:.2f}"]
    vals = [0, all_net, dep_net, best_net]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=["#B8C2CC", AMBER, BLUE, RISK],
                           text=[f"${v:,.0f}" for v in vals], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickprefix="$", range=[min(0, min(vals)) * 1.3, max(vals) * 1.25 if max(vals) > 0 else 1])
    plot(style(fig, 280, "Strategy comparison"))

    gap = (best_net - dep_net) / best_net if best_net > 0 else 0
    note(f"<b>How to read this.</b> Each flagged customer costs ${cost}. A contacted customer who really would have left is saved "
         f"{save:.0%} of the time and protects {months} months of their monthly bill. The 0.40 threshold earns "
         f"{dep_net / best_net:.0%} of the best achievable value; moving to {best_t:.2f} would "
         f"{'add' if gap > 0 else 'change'} ${abs(best_net - dep_net):,.0f}." if best_net > 0 else
         "<b>How to read this.</b> With these assumptions no threshold makes the program profitable. Lower the cost or raise the save rate.",
         "amber")
    st.caption("Calculated on the hold-out customers using their actual outcomes, so probability calibration does not distort the result. "
               "The deployed 0.40 threshold in the model is unchanged; this page shows what a different one would be worth.")


def page_model():
    h = get_holdout()
    y, p = h["y"], h["p"]
    m = art.metrics
    head("Model performance", "How well the tuned logistic regression separates churners from loyal customers.")
    kpi_strip([
        ("ROC-AUC", f"{m['roc_auc']:.3f}", f"{m['cv_roc_auc']:.3f} in 5-fold cross-validation", "blue"),
        ("Recall at 0.40", f"{m['recall']:.1%}", "Share of churners the model catches", ""),
        ("Precision at 0.40", f"{m['precision']:.1%}", "Share of flagged customers who really leave", ""),
        ("F1 at 0.40", f"{m['f1']:.3f}", f"Tuned C = {m['best_c']:g}, hold-out of {m['test_size']:,} customers", ""),
    ])

    cm = np.array(m["confusion_matrix"])
    flagged_share = (cm[0, 1] + cm[1, 1]) / cm.sum()
    baseline = 1 - m["test_churn_rate"]
    note(f"<b>The honest read.</b> At 0.40 the model catches {m['recall']:.0%} of churners but flags {flagged_share:.0%} of all customers, "
         f"so about {1 - m['precision']:.0%} of flags are false alarms. Accuracy is {m['accuracy']:.1%}, below the {baseline:.1%} you would get "
         f"by predicting that nobody churns. That is expected when you trade accuracy for recall, and it is why the Retention ROI page judges the threshold by money.",
         "amber")

    left, right = st.columns(2)
    with left:
        fpr, tpr, _ = roc_curve(y, p)
        op_fpr, op_tpr = cm[0, 1] / cm[0].sum(), cm[1, 1] / cm[1].sum()
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=LINE, dash="dash"), hoverinfo="skip")
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=BLUE, width=3), fill="tozeroy", fillcolor=BLUE_SOFT,
                        hovertemplate="False alarms %{x:.0%}, churners caught %{y:.0%}<extra></extra>")
        fig.add_scatter(x=[op_fpr], y=[op_tpr], mode="markers+text", text=[" Threshold 0.40"], textposition="bottom right",
                        marker=dict(color=RISK, size=12, line=dict(color="#fff", width=2)), hoverinfo="skip")
        fig.update_xaxes(title="Loyal customers wrongly flagged", tickformat=".0%")
        fig.update_yaxes(title="Churners caught", tickformat=".0%")
        plot(style(fig, 380, "ROC curve on the hold-out set"))
    with right:
        order = np.argsort(-p)
        gains = np.cumsum(y[order]) / y.sum()
        x = np.arange(1, len(y) + 1) / len(y)
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=LINE, dash="dash"), hoverinfo="skip")
        fig.add_scatter(x=x, y=gains, mode="lines", line=dict(color=RISK, width=3), fill="tozeroy", fillcolor=RISK_SOFT,
                        hovertemplate="Top %{x:.0%} by risk catches %{y:.0%} of churners<extra></extra>")
        fig.update_xaxes(title="Customers contacted, highest risk first", tickformat=".0%")
        fig.update_yaxes(title="Churners caught", tickformat=".0%")
        plot(style(fig, 380, "Cumulative gains"))

    left, right = st.columns(2)
    with left:
        ts = np.round(np.arange(0.05, 0.951, 0.01), 2)
        prec, rec = [], []
        for t in ts:
            f = p >= t
            tp = int((f & (y == 1)).sum())
            prec.append(tp / f.sum() if f.sum() else np.nan)
            rec.append(tp / y.sum())
        fig = go.Figure()
        fig.add_scatter(x=ts, y=rec, name="Recall", line=dict(color=RISK, width=3))
        fig.add_scatter(x=ts, y=prec, name="Precision", line=dict(color=BLUE, width=3))
        fig.add_vline(x=THR, line_color=INK, line_dash="dot", annotation_text="0.40", annotation_position="top")
        fig.update_xaxes(title="Threshold")
        fig.update_yaxes(tickformat=".0%")
        plot(style(fig, 340, "Precision and recall by threshold", legend=True))
    with right:
        z = cm
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted loyal", "Predicted churn"], y=["Actually loyal", "Actually churned"],
                                   colorscale=[[0, "#F4F6F9"], [1, BLUE]], showscale=False,
                                   text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=22, family=HEAD)))
        fig.update_yaxes(autorange="reversed")
        plot(style(fig, 340, "Confusion matrix at 0.40"))


def page_method():
    head("Methodology", "What the model is, how it was built, and where its limits are.")
    st.markdown(f"""
**Data.** Telco Customer Churn, 7,043 customers and 21 raw columns. A copy ships with the app so it runs offline.

**Preparation.** `TotalCharges` is converted to numeric and then dropped from the final feature set. `customerID` is removed. "No internet service" and
"No phone service" are collapsed to "No". Yes/No and gender fields are mapped to 0/1, `InternetService`, `Contract` and `PaymentMethod` are one-hot encoded,
and features are standardized.

**Model.** Logistic regression with balanced class weights, tuned over C in {{0.05, 0.1, 0.5, 1, 2}} using 5-fold stratified cross-validation on ROC-AUC,
then evaluated on a stratified 20% hold-out set. Customers are flagged at a probability of {THR:.2f}.

**How this dashboard scores a customer.** The model code builds one-hot columns with `drop_first=True`, which loses a category when only one customer is scored.
The scorer therefore adds the customer to reference rows that contain every category before encoding, so contract, internet and payment choices all count.

**Limits.**
- Probabilities come from a class-balanced model, so they overstate absolute churn likelihood. Use them to rank customers, not as literal odds.
- Coefficients and suggested plays show association, not cause. Monthly charges overlap heavily with service choices, so its individual coefficient is unstable.
- The data is a public sample from one telecom. Retrain on your own history, then add monitoring, calibration and governance before relying on it.
""")


# ───────────────────────────── navigation ─────────────────────────────
pg = st.navigation([
    st.Page(page_overview, title="Command center", icon=":material/space_dashboard:", url_path="overview", default=True),
    st.Page(page_scorer, title="Risk scorer", icon=":material/speed:", url_path="scorer"),
    st.Page(page_actions, title="Action list", icon=":material/call:", url_path="actions"),
    st.Page(page_roi, title="Retention ROI", icon=":material/payments:", url_path="roi"),
    st.Page(page_segments, title="Segments", icon=":material/donut_small:", url_path="segments"),
    st.Page(page_model, title="Model performance", icon=":material/monitoring:", url_path="model"),
    st.Page(page_method, title="Methodology", icon=":material/menu_book:", url_path="methodology"),
])
with st.sidebar:
    st.markdown("**ChurnLens**")
    st.caption(f"Tuned logistic regression, flag threshold {THR:.2f}. Hold-out ROC-AUC {art.metrics['roc_auc']:.3f}, recall {art.metrics['recall']:.0%}.")
pg.run()
