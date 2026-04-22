from services.data_loader import load_data
from services.preprocess import preprocess
from services.model import train_model
from services.fairness import compute_fairness
def generate_suggestions(dp, top_features, sensitive):

    suggestions = []

    # -----------------------------
    # HIGH BIAS
    # -----------------------------
    if dp > 0.4:
        suggestions.append({
            "id": 1,
            "severity": "High",
            "suggestion": f"High bias detected. Consider removing or transforming '{sensitive}' feature.",
            "estimatedImprovement": "+20% Bias Score"
        })

    # -----------------------------
    # MEDIUM BIAS
    # -----------------------------
    elif dp > 0.2:
        suggestions.append({
            "id": 2,
            "severity": "Medium",
            "suggestion": "Moderate bias detected. Try rebalancing dataset or using fairness-aware algorithms.",
            "estimatedImprovement": "+10% Bias Score"
        })

    # -----------------------------
    # LOW BIAS
    # -----------------------------
    else:
        suggestions.append({
            "id": 3,
            "severity": "Low",
            "suggestion": "Model is mostly fair. Monitor regularly and validate with more data.",
            "estimatedImprovement": "+2% Bias Score"
        })

    # -----------------------------
    # FEATURE CHECK
    # -----------------------------
    for f in top_features:
        if sensitive.lower() in f["feature"].lower():
            suggestions.append({
                "id": 4,
                "severity": "High",
                "suggestion": f"'{sensitive}' is highly influencing predictions. Consider removing it.",
                "estimatedImprovement": "+15% Bias Score"
            })

    return suggestions


def generate_response(url, target, sensitive):

    df = load_data(url)

    if df is None:
        return {"error": "Invalid CSV URL"}

    if target not in df.columns or sensitive not in df.columns:
        return {"error": "Invalid Column Name"}

    # Preprocess
    df_processed = preprocess(df)

    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    # Model
    model = train_model(X, y)
    y_pred = model.predict(X)

    # Fairness
    dp, eo, di = compute_fairness(
        y, y_pred, df_processed[sensitive]
    )

    # Group stats
    groups = df[sensitive].unique()
    group_stats = []

    for g in groups:
        mask = df[sensitive] == g
        total = mask.sum()
        approved = y_pred[mask].sum()

        rate = (approved / total) if total > 0 else 0

        group_stats.append({
            "group": str(g),
            "approvalRate": int(rate * 100),
            "totalCount": int(total)
        })

    # Feature importance
    importance = model.feature_importances_
    features = X.columns

    top_features = sorted([
        {"feature": str(f), "importance": round(float(i), 2)}
        for f, i in zip(features, importance)
    ], key=lambda x: x["importance"], reverse=True)[:5]

    # Bias score
    score = int(dp * 100)

    if score > 60:
        verdict = "HIGHLY BIASED"
    elif score > 20:
        verdict = "BIASED"
    else:
        verdict = "UNBIASED"

    return {
        "biasScore": score,
        "verdict": verdict,
        "metrics": {
            "demographicParity": round(dp, 2),
            "equalizedOdds": round(eo, 2),
            "disparateImpact": round(di, 2)
        },
        "groupStats": group_stats,
        "topFeatures": top_features,
        "fixSuggestions": [
            {
                "id": 1,
                "severity": "High",
                "suggestion": "Remove proxy features like location or zip_code.",
                "estimatedImprovement": "+10% Bias Score"
            }
        ]
    }