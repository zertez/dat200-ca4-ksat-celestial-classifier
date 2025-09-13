# %%
# Standard library imports
import os

import matplotlib.pyplot as plt

# Common aliases
import numpy as np
import pandas as pd
import seaborn as sns

# Scikit-learn with specific imports
from sklearn import (
    decomposition as decomp,
)
from sklearn import (
    ensemble as ens,
)
from sklearn import (
    metrics as met,
)
from sklearn import (
    model_selection as msel,
)
from sklearn import (
    pipeline as pipe,
)
from sklearn import (
    preprocessing as prep,
)
from sklearn import (
    utils as util,
)

# [Keep your setup code and data loading code]
sns.set_style("white")
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

# Set working directory
if "CA4" in os.getcwd():
    os.chdir("..")  # Go up one level if we're in CA3

print(f"Working directory now: {os.getcwd()}")

# Load data
train_path = os.path.join("CA4", "assets", "train.csv")
test_path = os.path.join("CA4", "assets", "test.csv")

# Load data
# 1. Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# %% Data inspection

print("---TRAIN DATA---")
train_df.info()
print("---TEST DATA---")
test_df.info()

# %% Metadata Removal

print("""
    A lot of metadata in this dataset (data that isnt physical properties) that is just used to organize the data for the future by astronomers. Seems also that u might be missing values/has placeholder values since it shows up as non-null, will investigate after metadata removal
    """)


features = ["u", "g", "r", "i", "z", "redshift", "class"]
# Keeping everything except metadata columns
train_df = train_df[features]
print(train_df.head())

# Modifying test_data
test_df = test_df[features[:-1]]
print(test_df.head())


# %% EDA - Data description
print("--- Train Data ---")
print(train_df.describe())

print("--- Test Data ---")
print(test_df.describe())

# %%

print("""
    This is strange, we have -9999 values for u, g and z. But in our original data inspection we found no missing values, so this might indicate placeholder values. We go forward with a better NaN handling strategy
""")

print(train_df.isnull().sum())
print("""
We can now see that u is actually missing 362 values. Because of our datasize of 80k, we can comfortably remove 362 samples from our dataset of 80k.
""")

# %% dropping na
train_df = train_df.dropna()

print(train_df.isnull().sum())

# %% Data description after na removal
print("--- Train Data ---")
train_df.info()

print(train_df.describe())
print(
    "There are still values with -9999, will investigate this further. This might be placeholder values, lets check"
)


# %%
negative_counts = (train_df.drop(columns=["class"]) == -9999).sum()
print("Negative value counts:")
print(negative_counts)

print(
    "Yes there is exactly one value with -9999 in the u, g and z columns. Will remove these"
)

# %% Replacing them with nan, and checking that they are now removed
train_df = train_df.replace(-9999, np.nan)

train_df = train_df.dropna()

print(train_df.isnull().sum())
# %% New eda on cleaned up data
print(train_df.describe())
# %% Overflow value in redshift
train_df["redshift"] = train_df["redshift"].abs()
print(train_df.describe())

# %% Encode class labels
class_mapping = {"GALAXY": 0, "QSO": 1, "STAR": 2}
train_df["class_labels"] = train_df["class"]  # Keep original text labels
train_df["class"] = train_df["class"].map(class_mapping)  # Replace with numeric codes


# %% Checking class distribution
print(
    train_df["class"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %"
)

print("""
    The class distribution is heavily skewed towards galaxies. Roghly 60% of the dataset are galaxies. We will balance this out before we split our data into test/val.
    """)


# %% EDA - Boxplot of raw data by feature
features = train_df.columns.drop(["class", "class_labels"])
classes = train_df["class_labels"].unique()

print(features)

print(
    "There are a lot of observations in the original dataset, we therefore extract a sample of 5000 to visualize our data. This still represents our data well, but its less taxing to compute and plot."
)

sample_size = 5000
plot_df = train_df.columns.drop(["class"])
plot_df_sample = train_df.sample(n=sample_size, random_state=42)

print(
    plot_df_sample["class_labels"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .astype(str)
    + " %"
)

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.boxplot(
        x="class_labels",
        y=feature,
        data=plot_df_sample,
        hue="class_labels",
        palette="Set1",
        ax=axes[i],
    )
    axes[i].set_title(f"Boxplot of {feature}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

    axes[i].tick_params(axis="x", labelsize=14)
    axes[i].patch.set_edgecolor("black")
    axes[i].patch.set_linewidth(1)

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(hspace=0.2)
plt.savefig("boxplot")
plt.show()

print("""
    Based on the boxplot quasars has the most outliers.
    """)

# %% EDA - Distribution plots

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.histplot(
        data=plot_df_sample,
        x=feature,
        hue="class_labels",
        palette="Set1",
        ax=axes[i],
        alpha=0.4,
        multiple="layer",
        kde=True,
    )
    axes[i].set_title(f"Histplot of {feature}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis="x", labelsize=14)
    axes[i].patch.set_edgecolor("black")
    axes[i].patch.set_linewidth(1)
# Adjust layout
plt.tight_layout()
fig.subplots_adjust(hspace=0.2)
plt.show()
print("""
The histplot shows that galaxy is biomodal for all features. Star is somewhat biomodal for features: g, r and i. Quasars is only unimodal.

It is interesting that galaxies are biomodal for all features, this might be due to galaxies come in different
sizes and shapes, and also have varying degrees of luminosity. It is expected that only galaxies will show redshift as
redshifting mostly occurs with galaxies.

That some features show stars as biomodal is also to be expected as stars vary in size and shape, but we did believe
that this effect should be stronger than it is.

The quasars are unimodal for all features.
""")


# %% Correlation Matrix
samples = [
    plot_df_sample[plot_df_sample["class_labels"] == cls][features] for cls in classes
]

# Set up the figure and axes
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

# Plot correlation matrix for each class
for i, (df, title) in enumerate(zip(samples, classes)):
    correlation_matrix = df.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=axes[i],
        square=True,
        vmin=-1.0,
        vmax=1.0,
    )
    axes[i].set_title(f"Correlation Matrix - {title} (n={len(df)})")
fig.delaxes(axes[3])
plt.tight_layout()
plt.xticks(rotation=90)
fig.subplots_adjust(hspace=0.2)
plt.show()
# %% Corr map on entire dataset

# Get the correlation matrix of all features (excluding the target column)
class_columns = [col for col in train_df.columns if "class" in col.lower()]
corr_df = train_df.drop(columns=class_columns)

plt.figure(figsize=(8, 5))
correlation_matrix = corr_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# %% Feature engineering

print("""
    We can group the filters based on thier photometric system like this:

    Visible light:
        - g (green filter)
        - r (red filter)
    Ultraviolet light:
        - u (ultraviolet)
    Infrared spectrum:
        - i (near infrared)
        - z (infrared)
    """)


# %%

galaxies = train_df[train_df["class_labels"] == "GALAXY"]
stars = train_df[train_df["class_labels"] == "STAR"]
qsos = train_df[train_df["class_labels"] == "QSO"]

ratio = 1.3
target_size = int(len(stars) * ratio)
galaxies_downsampled = util.resample(galaxies, n_samples=target_size, random_state=42)

# creating new balanced datasets
balanced_train_df = pd.concat([galaxies_downsampled, stars, qsos])

# %%
# checking new class balance

print(
    balanced_train_df["class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .astype(str)
    + " %"
)
print(balanced_train_df.info())

# %% Creating test split data set with sample size for faster training

sample_size = 20000
set_test_size = 0.20
sample_balanced_train_df = balanced_train_df.sample(n=sample_size, random_state=42)


X = sample_balanced_train_df.drop(columns=["class", "class_labels"])
y = sample_balanced_train_df["class"]

# Create an initial feßßature selection split
X_train, X_test, y_train, y_test = msel.train_test_split(
    X, y, test_size=set_test_size, random_state=42, stratify=y
)
# %% Pipeline setup for random forest with pca
pca = decomp.PCA()
random_forest = ens.RandomForestClassifier(n_jobs=-1, random_state=42)
scaler = prep.MinMaxScaler()

pipe_non_pca = pipe.Pipeline([("random_forest", random_forest)])


param_grid = {
    "random_forest__n_estimators": [100, 250, 500],
    "random_forest__max_depth": [10, 25, 50, 100, None],
    "random_forest__criterion": ["gini", "entropy", "log_loss"],
    "random_forest__min_samples_split": [5, 10],
}

search = msel.GridSearchCV(
    pipe_non_pca, param_grid, cv=3, n_jobs=-1, scoring="f1_macro"
)

search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# Print best results
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# %% Evaluate on test set
best_model = search.best_estimator_
y_test_pred = best_model.score(X_test, y_test)
print(f"Test set score with best model: {y_test_pred:.3f}")


# %% Detailed evaluation on test set from sample
# First, evaluate the sample model performance more thoroughly

# Get predictions on the sample test set
y_pred = best_model.predict(X_test)

# Print classification report
print("Classification Report on Sample Test Set:")
print(met.classification_report(y_test, y_pred))

# Create confusion matrix
print("Confusion Matrix on Sample Test Set:")
conf_matrix = met.confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Sample Test Set)")
plt.tight_layout()
plt.show()

# %% Extract best parameters from the sample model
# best_max_features = search.best_params_['random_forest__max_features']
best_n_estimators = search.best_params_["random_forest__n_estimators"]

print("Best parameters from sample training:")
# print(f"- RF max_features: {best_max_features}")
print(f"- RF n_estimators: {best_n_estimators}")

# %% Final model training on the entire training dataset
print("Training final model on the entire balanced dataset...")

# Prepare full data
X_full = balanced_train_df.drop(columns=["class_labels", "class"])
y_full = balanced_train_df["class"]

# %% Create a pipeline with the best parameters
final_rf = ens.RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=search.best_params_["random_forest__max_depth"],
    criterion=search.best_params_["random_forest__criterion"],
    min_samples_split=search.best_params_["random_forest__min_samples_split"],
    random_state=42,
    n_jobs=-1,  # Use all available cores
)

# Create the final pipeline with scaling
final_pipeline = pipe.Pipeline([("random_forest", final_rf)])

# Train the pipeline on unscaled data - pipeline handles scaling internally
final_pipeline.fit(X_full, y_full)

print(f"Final model trained on {len(X_full)} samples")

# %% Make predictions using the pipeline (no additional scaling needed)
test_features = test_df.copy()

# The pipeline handles scaling internally
y_pred = final_pipeline.predict(test_features)

# Create the final dataframe with numeric class predictions
y_test = pd.DataFrame(y_pred, columns=["class"])
y_test.index.name = "ID"

# Add file path with appropriate naming related to model parameters
base_dir = os.path.join("CA4", "results")
os.makedirs(base_dir, exist_ok=True)
filename = f"submission_normal_data_est{best_n_estimators}_sample_size{sample_size}_test_size{set_test_size}.csv"
file_path = os.path.join(base_dir, filename)

# Save to CSV
y_test.to_csv(file_path)  # Keep the index as it's set to "ID"
print(f"Saved submission to {file_path}")

# %% Summary of model and prediction
print("\n===== Final Model Summary =====")
print("Model type: Random Forest")
print("Random Forest parameters:")
print(f"  - n_estimators: {best_n_estimators}")
print(f"  - max_depth: {search.best_params_['random_forest__max_depth']}")
print(f"  - criterion: {search.best_params_['random_forest__criterion']}")
print(
    f"  - min_samples_split: {search.best_params_['random_forest__min_samples_split']}"
)
print(f"Training data size: {len(X_full)} samples")
print(f"Test data size: {len(test_df)} samples")
print(f"Prediction file: {filename}")
