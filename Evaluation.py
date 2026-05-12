from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Encode labels (binary)
y_type_series = pd.Series(y_type, index=X_n.index)
y_binary = y_type_series.map({"Tissue": 0, "Organoid": 1}).values

# If your labels contain lowercase or extra spaces, normalize:
# y_binary = y_type_series.str.strip().str.capitalize().map({"Tissue": 0, "Organoid": 1}).values

clf = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

gkf = GroupKFold(n_splits=5)

all_true = []
all_pred = []

for fold, (tr, te) in enumerate(gkf.split(X_n, y_binary, groups=groups), start=1):
    Xtr, Xte = X_n.iloc[tr], X_n.iloc[te]
    ytr, yte = y_binary[tr], y_binary[te]

    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    all_true.extend(yte.tolist())
    all_pred.extend(yhat.tolist())

    print(f"\n=== Fold {fold} ===")
    print(classification_report(yte, yhat, target_names=["Tissue", "Organoid"]))

print("\n=== Overall (all folds combined) ===")
print(classification_report(all_true, all_pred, target_names=["Tissue", "Organoid"]))
print("Confusion matrix:\n", confusion_matrix(all_true, all_pred))
