from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def run_multiclass_tissue_task(Xn: pd.DataFrame, meta_df: pd.DataFrame, subset_type: str):
    """
    Train/evaluate tissue-name classifier inside a subset (only Tissue or only Organoid),
    using leakage-safe GroupKFold by GSE.
    """
    mask = meta_df["type"].astype(str).values == subset_type
    Xs = Xn.loc[mask]
    ys = meta_df.loc[mask, "tissue"].astype(str).values
    gs = meta_df.loc[mask, "GSE"].astype(str).values

    # Remove rare tissues (too few samples) to avoid meaningless metrics
    vc = pd.Series(ys).value_counts()
    keep_labels = set(vc[vc >= 10].index)  # threshold you can adjust
    keep_mask = np.array([y in keep_labels for y in ys])

    Xs = Xs.loc[keep_mask]
    ys = ys[keep_mask]
    gs = gs[keep_mask]

    print(f"\n[{subset_type}] samples:", Xs.shape[0], "classes:", len(set(ys)))

    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=8000,
            class_weight="balanced",
            n_jobs=-1,
            multi_class="auto"
        ))
    ])

    gkf = GroupKFold(n_splits=5)

    all_true, all_pred = [], []
    for fold, (tr, te) in enumerate(gkf.split(Xs, ys, groups=gs), start=1):
        model.fit(Xs.iloc[tr], ys[tr])
        yhat = model.predict(Xs.iloc[te])

        all_true.extend(ys[te].tolist())
        all_pred.extend(yhat.tolist())

        print(f"\n=== {subset_type} Fold {fold} ===")
        print(classification_report(ys[te], yhat, zero_division=0))

    print(f"\n=== {subset_type} Overall ===")
    print(classification_report(all_true, all_pred, zero_division=0))


# Build a meta_df aligned to X_n index order
meta_aligned = meta.set_index("SRR").loc[X_n.index].reset_index()

# Secondary-A: Tissue subset
run_multiclass_tissue_task(X_n, meta_aligned, subset_type="Tissue")

# Secondary-B: Organoid subset (optional)
run_multiclass_tissue_task(X_n, meta_aligned, subset_type="Organoid")
