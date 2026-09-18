"""Interpretability (Ben's ask): does a SIMPLE, readable model do nearly as well as the gradient-boosted
detector, and what drives the prediction? Fits logistic regression + a shallow tree on the 11 features and
compares out-of-fold AUPRC to the GBM, and reports per-feature logistic coefficients (direction + magnitude).

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.radar_interpretability

Uses the headline AI/ML-core pool (artifacts/phase1_aiml) if present. 5-fold grouped OOF, same as the detector.
"""
from __future__ import annotations
import pathlib, warnings
warnings.filterwarnings("ignore")


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES

    MB = pathlib.Path(".")
    src = MB / ("artifacts/phase1_aiml" if (MB / "artifacts/phase1_aiml/features.parquet").exists() else "artifacts/phase1")
    feat = pd.read_parquet(src / "features.parquet")
    sc = pd.read_parquet(src / "scores.parquet")[["theme_id", "label"]] if "label" not in feat.columns else None
    df = feat.merge(sc, on="theme_id", how="left") if sc is not None else feat
    disc = df[df["label"].isin(("positive", "hyped_died"))].reset_index(drop=True)
    disc["parent_id"] = disc["theme_id"]
    y = (disc["label"] == "positive").astype(int).to_numpy()
    X = disc[SIGNATURE_FEATURES].fillna(0).to_numpy(float)
    Xs = StandardScaler().fit_transform(X)
    folds = splits.grouped_folds(disc, k=5)
    print(f"[interp] {src.name}: n={len(disc)} ({int(y.sum())} durable / {int((1-y).sum())} fad), base rate {y.mean():.3f}\n", flush=True)

    def oof(make, data):
        pred = np.zeros(len(disc))
        for tr, te in folds:
            tri = disc["theme_id"].isin(tr["theme_id"]).to_numpy(); tei = disc["theme_id"].isin(te["theme_id"]).to_numpy()
            if len(np.unique(y[tri])) < 2:
                continue
            m = make().fit(data[tri], y[tri]); pred[tei] = m.predict_proba(data[tei])[:, 1]
        return metrics.auprc(pred, y)

    gbm = oof(lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=10, random_state=0), X)
    logit = oof(lambda: LogisticRegression(max_iter=1000, C=1.0), Xs)
    tree = oof(lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, random_state=0), X)
    print("[interp] out-of-fold AUPRC (base rate {:.2f}):".format(y.mean()))
    print(f"   gradient-boosted detector (11 feats)   {gbm:.3f}")
    print(f"   logistic regression       (11 feats)   {logit:.3f}   (gap {logit-gbm:+.3f})")
    print(f"   decision tree depth-3     (11 feats)   {tree:.3f}   (gap {tree-gbm:+.3f})")
    within = abs(gbm - logit)
    print(f"\n[interp] => a simple logistic model is within {within:.3f} AUPRC of the boosted detector"
          f" ({'competitive' if within<=0.03 else 'the nonlinear combination adds ' + format(gbm-logit,'+.3f')}).", flush=True)

    lr = LogisticRegression(max_iter=1000, C=1.0).fit(Xs, y)
    coef = sorted(zip(SIGNATURE_FEATURES, lr.coef_[0]), key=lambda t: -abs(t[1]))
    print("\n[interp] logistic coefficients (standardised; + => more durable):")
    for f, c in coef:
        bar = ("+" if c >= 0 else "-") * min(int(abs(c) / 0.1) + 1, 20)
        print(f"   {f:22s} {c:+.3f}  {bar}")


if __name__ == "__main__":
    main()
