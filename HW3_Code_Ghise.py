import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import shap
import io
from PIL import Image


CSV_PATH = 'lncRNA_5_Cancers.csv'
ID_COL = 'Ensembl_ID'
CLASS_COL = 'Class'


# Read the CSV file
df = pd.read_csv(CSV_PATH)
print(f"Initial data shape: {df.shape}")

# Drop id and class columns for the features set
X = df.drop(columns=[ID_COL, CLASS_COL])
# Keep the class column for output variable set
y = df[CLASS_COL]

# Cancer types
cancer_types = ["KIRC", "LUAD", "LUSC", "PRAD", "THCA"]

# Target patients, one for each class
target_sample_names = [
    "TCGA-CJ-4641-01A",         # KIRC patient
    "TCGA-99-8033-01A",         # LUAD patient
    "TCGA-66-2771-01A",         # LUSC patient
    "TCGA-HC-7821-01A",         # PRAD patient
    "TCGA-E3-A3DY-01A"          # THCA patient
]



# Task 1

# Split data into training and testing (stratify=yes keeps the class proportions in train and test sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Preserve the target patients in the testing data, if the ended up in the training data move them

# Get the indexes of target patients from the dataframe
target_idx = df.index[df[ID_COL].isin(target_sample_names)]

# Find which of the target patients ended up in the training data -> save their indexes
move_idx = X_train.index.intersection(target_idx)

# Move them from train to test (indexes are preserved)
X_test  = pd.concat([X_test,  X_train.loc[move_idx]])
y_test  = pd.concat([y_test,  y_train.loc[move_idx]])
X_train = X_train.drop(move_idx)
y_train = y_train.drop(move_idx)



# Verify that the target patients ended up in the test split (they should be there in order to apply SHAP of them)
# Ensemble IDs of the rows that ended up in each split
test_ids  = set(df.loc[X_test.index,  ID_COL])
train_ids = set(df.loc[X_train.index, ID_COL])

for sid in target_sample_names:
    in_df    = sid in set(df[ID_COL])
    in_test  = sid in test_ids
    in_train = sid in train_ids
    print(f"{sid}:  in CSV={in_df}  |  in X_test={in_test}  |  in X_train={in_train}")
    


# Train decision tree model
dt_classifier = DecisionTreeClassifier(
    criterion="entropy",        # criteria to calculate node purity
    max_depth=5,                # limit depth to reduce overfitting
    min_samples_leaf=10,        # every leaf should hold at least 2 samples, if not that split fails => no overfitting
    random_state=42,            # seed for reproducibility
)
dt_classifier.fit(X_train, y_train)

# Predictions
y_pred = dt_classifier.predict(X_test)

# Per-class probabilities (one column for each class)
y_prob = dt_classifier.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
# Used one over rest(multiclass classification) + weighted AUCs for each 5 classes
auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

# Save the metrics to display them on a bar plot
dt_metrics = [accuracy, precision, recall, f1, auc]

print(f"ROC AUC of decision tree: {auc:.3f}")
print("Classification report of decision tree: \n", classification_report(y_test, y_pred))



# Train a random forest
rf_classifier = RandomForestClassifier(
    n_estimators=100,           # number of decision trees
    criterion="entropy",        # criteria to calculate node purity
    min_samples_leaf=2,         # every leaf should hold at least 2 samples, if not that split fails => no overfitting
    random_state=42,            # seed for reproducibility
    n_jobs=-1                   # use all available CPU cores
)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Per-class probabilitie (one column for each class)
y_prob = rf_classifier.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

# Save the metrics to display them on a bar plot
rf_metrics = [accuracy, precision, recall, f1, auc]

print(f"ROC AUC of random forest: {auc:.3f}")
print("Classification report of random forest: \n", classification_report(y_test, y_pred))



# Train Gradient Boosting Machine
gbm_classifier = GradientBoostingClassifier(
    n_estimators=100,           # number of boosting stages
    learning_rate=0.1,          # learning rate => how fast the weights change between trees 
    max_depth=3,                # max depth of each individual tree
    min_samples_leaf=2,         # every leaf should hold at least 2 samples, if not that split fails => no overfitting
    n_iter_no_change=10,        # stop after 5 iteratiosn where the model doesnt improve
    random_state=42,            # seed for reproducibility
    verbose=0                   # no logs displayed
)
gbm_classifier.fit(X_train, y_train)

# Predictions
y_pred = gbm_classifier.predict(X_test)

# Per-class probabilities
y_prob = gbm_classifier.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

# Save the metrics to display them on a bar plot
gbm_metrics = [accuracy, precision, recall, f1, auc]

print(f"ROC AUC of GBM: {auc:.3f}")
print("Classification report for GBM: \n", classification_report(y_test, y_pred))



# Train XGBoost
label_encoder = LabelEncoder()          # encode class names to numeric values (KIRC -> 0, LUAD -> 1 and so on)
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

xgb_classifier = XGBClassifier(
    n_estimators=200,        # number of boosting rounds (trees)
    learning_rate=0.1,       # how much each new tree contributes
    max_depth=5,             # max depth of each tree
    random_state=42,         # seed for reproducibility
    n_jobs=-1,               # use all available CPU cores
    verbose=-1               # don't display logs
)
xgb_classifier.fit(X_train, y_train_enc)

# Predicitons
y_pred = xgb_classifier.predict(X_test)

# Per-class probabilities
y_prob = xgb_classifier.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test_enc, y_pred)
precision = precision_score(y_test_enc, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test_enc, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="weighted")

# Save the metrics to display them on a bar plot
xgb_metrics = [accuracy, precision, recall, f1, auc]

print(f"ROC AUC of XGBoost: {auc:.3f}")
print("Classification report of XGBoost: \n", classification_report(y_test_enc, y_pred))



# Train LightGBM
lightgbm_classifier = LGBMClassifier(
    n_estimators=500,        # max number of trees
    learning_rate=0.1,       # how much each new tree contributes
    max_depth=-1,            # max depth of each tree
    random_state=42,         # seed for reproducibility
    n_jobs=-1,               # use all available CPU cores
    verbose=-1               # don't display logs
)
lightgbm_classifier.fit(X_train, y_train_enc)

# Predicitons
y_pred = lightgbm_classifier.predict(X_test)

# Per-class probabilities
y_prob = lightgbm_classifier.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test_enc, y_pred)
precision = precision_score(y_test_enc, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test_enc, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="weighted")

# Save the metrics to display them on a bar plot
lightgbm_metrics = [accuracy, precision, recall, f1, auc]

print(f"ROC AUC of LightGBM: {auc:.3f}")
print("Classification report of LightGBM: \n", classification_report(y_test_enc, y_pred))



# Train Catboost
catboost_classifier = CatBoostClassifier(
    n_estimators=200,             # max number of trees (boosting rounds)
    learning_rate=0.1,            # how much each new tree's correction counts
    max_depth=5,                  # max depth for each tree
    loss_function="MultiClass",   # multiclass log loss
    early_stopping_rounds=10,     # stop if validation loss doesn't improve for 20 rounds
    random_seed=42,               # seed for reproducibility
    thread_count=-1,              # use all available CPU cores
    verbose=0,                    # don't print logs
    allow_writing_files=False,    # don't create log folder
)
catboost_classifier.fit(X_train, y_train_enc)

# Predicitons
y_pred = catboost_classifier.predict(X_test)

# Per-class probabilities
y_prob = catboost_classifier.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test_enc, y_pred)
precision = precision_score(y_test_enc, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test_enc, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="weighted")

# Save the metrics to display them on a bar plot
catboost_metrics = [accuracy, precision, recall, f1, auc]
print(f"ROC AUC of CatBoost: {auc:.3f}")
print("Classification report of CatBoost: \n", classification_report(y_test_enc, y_pred))



# Display bar plots with metrics
results = {
    "DecisionTree":     dt_metrics,
    "RandomForest":     rf_metrics,
    "GradientBoosting": gbm_metrics,
    "XGBoost":          xgb_metrics,
    "LightGBM":         lightgbm_metrics,
    "CatBoost":         catboost_metrics,
}

# Metrics to plot
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
idx = list(range(len(metrics)))

# Sort models by accuracy, best first
models = sorted(results, key=lambda m: results[m][0], reverse=True)
scores = np.array([[results[m][i] for i in idx] for m in models])  # shape: (models, metrics)

colors = ["#3b75af", "#ef8636", "#519e3e", "#c53a32", "#8c6bb1"][:len(metrics)]

n_models, n_metrics = scores.shape
x = np.arange(n_models)
group_width = 0.8
bar_width = group_width / n_metrics

# Create stacked bar plot for each evaluation metric of all tree based classifiers
fig, ax = plt.subplots(figsize=(16, 7))
for j, (metric, color) in enumerate(zip(metrics, colors)):
    offset = (j - (n_metrics - 1) / 2) * bar_width
    ax.bar(x + offset, scores[:, j], width=bar_width, label=metric, color=color)

# Zoom the y-axis just below the lowest score so the differences show up
ymin = np.floor((scores.min() - 0.01) * 50) / 50   # round down to the nearest 0.02
ax.set_ylim(ymin, 1.0)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")
ax.set_ylabel("Score")
ax.set_title("Model Performance")
ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1), frameon=False)

plt.tight_layout()
plt.savefig("model_performance.png", dpi=300, bbox_inches="tight")
plt.show()



# Task 2
# Apply SHAP on LightGBM

# Create SHAP using TreeExplainer and applied it to LightGBM classifier using the training data
explainer = shap.TreeExplainer(
    lightgbm_classifier,
    data=X_train,
    feature_perturbation="interventional",
    model_output="probability"
)

# Compute SHAP values on the test set
# For this multiclass model, shape = (n_samples, n_features, n_classes):
# shap_values[s, f, c] = how much feature f pushed sample s toward class c
shap_values = explainer.shap_values(X_test)

# Expected value = the model's average prediction over the background data (one per class)
# For each sample and class: base_value + sum(shap values) = model prediction.
base_values = explainer.expected_value

# Global: top 20 features, stacked by class
shap_list = []
for c in range(0, len(cancer_types)):
    # Aggregate in shap_list the shap values based on each class
    shap_list.append(shap_values[:, :, c])          # shap_list[0] => (n_samples, n_features) for KIRC class and so on

# Plot the top 20 global features
shap.summary_plot(
    shap_list,                      # put the entire list of shap values (for all classes)
    X_test,
    plot_type="bar",
    class_names=cancer_types,
    max_display=20,                 # display top 20 global features (from all classes)
    show=False,
    plot_size=(14, 6)
)
plt.xlabel("Mean |SHAP| value (per class)")
plt.ylabel("Features")
plt.title("Top 20 lncRNAs by mean |SHAP value| (LightGBM)")
plt.tight_layout()
plt.savefig("global_20_features.png", dpi=300, bbox_inches="tight")
plt.show()


# Iterate through all classes
for c in range(0, len(cancer_types)):
    # Select only the shap values specific for the current class
    class_shap_list = shap_values[:, :, c]
    shap.summary_plot(
        class_shap_list,            # here just put the specific shap values for a class, not for all classes
        X_test,
        plot_type="bar",
        max_display=10,             # display top 10 features for class c
        show=False,
        plot_size=(14, 6) 
    )
    plt.xlabel("Mean |SHAP| value")
    plt.ylabel("Features")
    plt.title(f"{cancer_types[c]} - Top 10 Features by Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(f"{cancer_types[c]}-top-10-features.png", dpi=300, bbox_inches="tight")
    plt.show()



# Class names in the exact order the model / SHAP use
class_names = list(label_encoder.classes_)
n_classes = len(class_names)

# df row label of each target, in the order of target_sample_names
target_rows = [df.index[df[ID_COL] == sid][0] for sid in target_sample_names]

# Only the 5 target rows: row i belongs to target_sample_names[i]
X_targets = X_test.loc[target_rows]
y_targets = y_test.loc[target_rows]

# SHAP for these 5 only (same values as in shap_values, just a smaller array)
shap_targets = explainer.shap_values(X_targets)                # (5, n_features, n_classes)

# Shorter labels to fit the window
short_names = [col.split(".")[0] for col in X_test.columns]

# Iterate through all 5 target patients
for i, sid in enumerate(target_sample_names):
    true_c = y_targets.iloc[i]
    print(f"   - Class = {true_c}  |  SampleID: {sid}")

    # One window per patient: n_classes rows, one force plot per row
    fig, axes = plt.subplots(n_classes, 1, figsize=(20, 3.2 * n_classes))

    # Display force plot for each class in regard to current iterated target patient
    for c in range(n_classes):
        # Draw the force plot (SHAP creates its own figure)
        shap_fig = shap.force_plot(
            base_value=base_values[c],              # average predicted prob of class c
            shap_values=shap_targets[i, :, c],      # this patient's SHAP values for class c
            features=X_targets.iloc[i].round(3).values,
            feature_names=short_names,              # set labels to shorter version for better fitting in the plot
            matplotlib=True,
            show=False,
            figsize=(24, 3.5),
            text_rotation=20,                       # tilt labels so they don't overlap
            contribution_threshold=0.05,            # label only features contributing > 5%
        )
        if shap_fig is None:                        # older SHAP versions return None
            shap_fig = plt.gcf()

        # Save it to memory as PNG and close it
        buf = io.BytesIO()
        shap_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(shap_fig)
        buf.seek(0)

        # Paste the image into row c of the patient's window
        ax = axes[c]
        ax.imshow(Image.open(buf))
        ax.set_xticks([])
        ax.set_yticks([])

        # Red, thicker frame for the true class; green for the others
        is_true = (class_names[c] == true_c)
        for spine in ax.spines.values():
            spine.set_edgecolor("#c53a32" if is_true else "#4f7a28")
            spine.set_linewidth(3 if is_true else 2)

        # Class name on the left
        ax.set_ylabel(f"{class_names[c]}", rotation=0, fontsize=14, fontweight="bold", labelpad=60, va="center")

    fig.suptitle(f"Force plots for {true_c} patient {sid}",
                 fontsize=22, fontweight="bold", color="#6a2c91")
    fig.tight_layout(rect=[0, 0, 1, 0.97])        # leave room at the top for the title
    fig.savefig(f"force_plot_{true_c}_{sid}.png", dpi=200, bbox_inches="tight")
    plt.show()