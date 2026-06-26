from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
all_prob = []

for fold, (tr, te) in enumerate(gkf.split(X_n, y_binary, groups=groups), start=1):
    Xtr, Xte = X_n.iloc[tr], X_n.iloc[te]
    ytr, yte = y_binary[tr], y_binary[te]

    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:, 1]

    all_true.extend(yte.tolist())
    all_pred.extend(yhat.tolist())
    all_prob.extend(yprob.tolist())

    print(f"\n=== Fold {fold} ===")
    print(classification_report(yte, yhat, target_names=["Tissue", "Organoid"]))

print("\n=== Overall (all folds combined) ===")
print(classification_report(all_true, all_pred, target_names=["Tissue", "Organoid"]))
print("Confusion matrix:\n", confusion_matrix(all_true, all_pred))

#Plot##############

#


from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

#PLOTS

# 1) Confusion Matrix
cm = confusion_matrix(all_true, all_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Tissue", "Organoid"]
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# 2) ROC Curve
fpr, tpr, thresholds = roc_curve(all_true, all_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 3) PCA Plot
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_n)

plt.figure(figsize=(7,6))

for label in np.unique(y_type):
    idx = np.array(y_type) == label
    plt.scatter(
        pcs[idx, 0],
        pcs[idx, 1],
        label=label,
        alpha=0.7
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Samples")
plt.legend()
plt.tight_layout()
plt.show()

#  4) Gene Heatmap

import seaborn as sns

# Gene Variance
gene_var = X_n.var(axis=0)

# chooce the most gene or only 50
top_genes = gene_var.nlargest(50).index

#
heat_data = X_n[top_genes]

plt.figure(figsize=(12,8))

sns.heatmap(
    heat_data,
    cmap="RdBu_r",
    center=heat_data.values.mean()
)

plt.title("Top 50 Most Variable Genes")
plt.xlabel("Genes")
plt.ylabel("Samples")

plt.tight_layout()
plt.show()