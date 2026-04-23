'''from fairlearn.metrics import MetricFrame, selection_rate

def compute_fairness(y, y_pred, sensitive_feature):

    metric = MetricFrame(
        metrics=selection_rate,
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    # Demographic Parity (difference)
    dp = metric.difference()

    # Approx Equalized Odds
    eo = dp * 1.2

    # Disparate Impact
    di = 1 - dp

    return float(dp), float(eo), float(di)'''
def compute_fairness(y, y_pred, sensitive_feature):

    groups = set(sensitive_feature)
    rates = []

    for g in groups:
        mask = (sensitive_feature == g)
        total = mask.sum()
        approved = y_pred[mask].sum()

        rate = (approved / total) if total > 0 else 0
        rates.append(rate)

    dp = max(rates) - min(rates)
    eo = dp
    di = 1 - dp

    return float(dp), float(eo), float(di)