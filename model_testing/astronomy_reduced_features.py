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

# %% [markdown]
# # Metadata Removal

# A lot of metadata in this dataset (data that isnt physical properties) that is just used to organize the data for the future by astronomers. Seems also that u might be missing values/has placeholder values since it shows up as non-null, will investigate after metadata removal

# %%
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

# %% [markdown]
# This is strange, we have - 9999 values for u, g and z. But in our original data inspection we found no missing values, so this might indicate placeholder values. We go forward with a better NaN handling strategy

# %%
print(train_df.isnull().sum())

# %% [markdown]

# We can now see that u is actually missing 362 values. Because of our datasize of 80k, we can comfortably remove 362 samples from our dataset of 80k.

# %% dropping na
train_df = train_df.dropna()

print(train_df.isnull().sum())

# %% Data description after na removal
print("--- Train Data ---")
train_df.info()

print(train_df.describe())

# %% [markdown]
# There are still values with -9999, will investigate this further. This might be placeholder values, lets check"


# %%
negative_counts = (train_df.drop(columns=["class"]) == -9999).sum()
print("Negative value counts:")
print(negative_counts)

# %% [markdown]
# Yes there is exactly one value with -9999 in the u, g and z columns. Will remove these

# %% Replacing them with nan, and checking that they are now removed
train_df = train_df.replace(-9999, np.nan)

train_df = train_df.dropna()

print(train_df.isnull().sum())
# %% New eda on cleaned up data
print(train_df.describe())


# %% Encode class labels
class_mapping = {"GALAXY": 0, "QSO": 1, "STAR": 2}
train_df["class_labels"] = train_df["class"]  # Keep original text labels
train_df["class"] = train_df["class"].map(class_mapping)  # Replace with numeric codes


# %% Checking class distribution
print(
    train_df["class"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %"
)


# %% [markdown]
# The class distribution is heavily skewed towards galaxies. Roghly 60 % of the dataset are galaxies. We will balance this out before we split our data into test/val.


# %% EDA - Boxplot of raw data by feature
features = train_df.columns.drop(["class", "class_labels"])
classes = train_df["class_labels"].unique()

print(features)

# %%[markdown]

# There are a lot of observations in the original dataset, we therefore extract a sample of 5000 to visualize our data. This still represents our data well, but its less taxing to compute and plot.

# %%
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

# %% [markdown]

# Based on the boxplot quasars has the most outliers.

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

# %% [markdown]

# The histplot shows that galaxy is biomodal for all features. Star is somewhat biomodal for features: g, r and i. Quasars is only unimodal.

# It is interesting that galaxies are biomodal for all features, this might be due to galaxies come in different sizes and shapes, and also have varying degrees of luminosity. It is expected that only galaxies will show redshift as redshifting mostly occurs with galaxies.

# That some features show stars as biomodal is also to be expected as stars vary in size and shape, but we did believe that this effect should be stronger than it is .

# The quasars are unimodal for all features.


# %% Corr matrix for all classes

# Get the correlation matrix of all features (excluding the target column)
class_columns = [col for col in train_df.columns if "class" in col.lower()]
corr_df = train_df.drop(columns=class_columns)

plt.figure(figsize=(8, 5))
correlation_matrix = corr_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
# %% [markdown]
# # Feature Engineering
# We can group the filters based on thier photometric system like this:
#   Visible light:
#    - g(green filter)
#    - r(red filter)
#   Ultraviolet light:
#    - u(ultraviolet)
# Infrared spectrum:
#    - i(near infrared)
#    - z(infrared)

# %% Feature engineering


def add_color_indices(df):
    """Adding astronomical color indices to the dataframe."""
    df_features = df.copy()
    # creating color index features, which is widely used in astronomy
    # Standard astronomical color indices (shorter wavelength - longer wavelength)
    df_features["z-r"] = df_features["z"] - df_features["r"]
    df_features["g-r"] = df_features["g"] - df_features["r"]
    df_features["u-r"] = df_features["u"] - df_features["r"]
    df_features["i-z"] = df_features["i"] - df_features["z"]
    df_features["i-r"] = df_features["i"] - df_features["r"]
    return df_features


# %% [markdown]
# # Class imbalance

# %%
galaxies = train_df[train_df["class_labels"] == "GALAXY"]
stars = train_df[train_df["class_labels"] == "STAR"]
qsos = train_df[train_df["class_labels"] == "QSO"]

ratio = 1.3
target_size = int(len(stars) * ratio)
galaxies_downsampled = util.resample(galaxies, n_samples=target_size, random_state=42)

# creating new balanced datasets
balanced_train_df = pd.concat([galaxies_downsampled, stars, qsos])

train_features_df = add_color_indices(train_df)

top_features = [
    "redshift_squared",
    "log_redshift",
    "redshift",
    "redshift_u",
    "g-i",
    "g-r",
    "spectral_contrast",
    "r-i",
    "z-r",
    "u-r",
    "g",
]

train_df_features_reduced = train_features_df[top_features]
print(train_df_features_reduced.head())

# %%
# checking new class balance

print(
    train_features_df["class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .astype(str)
    + " %"
)
print(train_features_df.info())

# %% [markdown]
# # Training the models
# Since we are working with three categorical target values and a large dataset we have chosen to go for the following models:
# - Random Forest Classifier
# - Linear Regression model
# - Gradient Boosting Classifier with PCA

# %% Creating test split data set with sample size for faster training

X = train_df_features_reduced.drop(columns=["class", "class_labels"])
y = train_df_features_reduced["class"]

# Create an initial feature selection split
X_train, X_test, y_train, y_test = msel.train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# %% [markdown]
# ## Random Forest Classifier

# %% Pipeline setup for random forest with pca
random_forest = ens.RandomForestClassifier(n_jobs=-1, random_state=42)
scaler = prep.StandardScaler()

pipe_non_pca = pipe.Pipeline([("random_forest", random_forest)])


# Parameter distributions for random search
param_distributions = {
    # Uniform integer between 100-500
    "random_forest__n_estimators": [100, 200, 300, 400, 500],
    "random_forest__max_depth": [5, 10, 15, 20, 25, None],
    "random_forest__min_samples_split": [2, 5, 10, 15, 20],
    "random_forest__max_features": ["sqrt", "log2"],
    "random_forest__criterion": ["gini", "entropy", "log_loss"],
}

# Define the stratified cross-validation
stratified_cv = msel.StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Create the random search with StratifiedKFold
random_search = msel.RandomizedSearchCV(
    pipe_non_pca,
    param_distributions=param_distributions,
    n_iter=100,
    cv=stratified_cv,  # Using StratifiedKFold
    n_jobs=-1,
    scoring="f1_macro",
    return_train_score=True,
    verbose=3,
    random_state=42,  # For reproducibility
)

random_search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % random_search.best_score_)
print(random_search.best_params_)


# Print best results
print("Best parameter (CV score=%0.3f):" % random_search.best_score_)
print(random_search.best_params_)

# %% Evaluate on test set
best_model = random_search.best_estimator_
y_test_pred = best_model.score(X_test, y_test)
print(f"Test set score with best model: {y_test_pred:.3f}")
print(best_model)
# %% Plotting feature importance
importances = best_model.named_steps["random_forest"].feature_importances_
feature_names = X_train.columns
imp_df = pd.DataFrame()
imp_df["Features"] = feature_names
imp_df["Importance"] = importances * 100
imp_df = imp_df.sort_values(by=["Importance"], ascending=False)
imp_df["Cumulative"] = np.cumsum(a=imp_df["Importance"].values)

# Create figure
plt.figure(figsize=(10, 6))

# Create barplot using seaborn
bars = sns.barplot(data=imp_df, x="Features", y="Cumulative")

# Add annotations to each bar
for b in bars.patches:
    x = b.get_x() + (b.get_width() / 2)
    y = np.round(b.get_height(), 3)
    bars.annotate(
        text=format(y),
        xy=(x, y),
        ha="center",
        va="center",
        size=8,
        xytext=(0, 6),
        textcoords="offset points",
    )

# Set title
plt.title(label="Cumulative Importance")
plt.xticks(rotation=90)
# Display plot
plt.tight_layout()
plt.show()
# %% Making new dataset with selected that explains ~0.95


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
best_n_estimators = random_search.best_params_["random_forest__n_estimators"]
best_max_depth = random_search.best_params_["random_forest__max_depth"]
best_min_samples_split = random_search.best_params_["random_forest__min_samples_split"]
best_max_features = best_min_samples_split = random_search.best_params_[
    "random_forest__max_features"
]
best_criterion = random_search.best_params_["random_forest__criterion"]

print("Best parameters from sample training")
print(f"- RF max_features: {best_max_features}")
print(f"- RF n_estimators: {best_n_estimators}")
print(f"- RF max_depth: {best_max_depth}")
print(f"- RF min_samples_splot: {best_min_samples_split}")
print(f"- RF criterion: {best_criterion}")
# %% Testing on unbalanced

train_df_features = train_df.copy()

train_df_features = add_color_indices(train_df)


# %% Final model training on the entire training dataset
print("Training final model on the entire balanced dataset...")

# Prepare full data
X_full = train_df_features.drop(columns=["class_labels", "class"])
y_full = train_df_features["class"]


final_rf = ens.RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_features=best_max_features,
    max_depth=best_max_depth,
    criterion=best_criterion,
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

test_full_features = add_color_indices(test_df)

# %%

# The pipeline handles scaling internally
y_test = final_pipeline.predict(test_full_features)

# Create the final dataframe with numeric class predictions
y_test = pd.DataFrame(y_test, columns=["class"])
y_test.index.name = "ID"

# Add file path with appropriate naming related to model parameters
base_dir = os.path.join("CA4", "results")
os.makedirs(base_dir, exist_ok=True)
filename = f"submission_reduced_features_est{best_n_estimators}.csv"
file_path = os.path.join(base_dir, filename)

# Save to CSV
y_test[["class"]].to_csv(file_path)
print(f"Saved submission to {file_path}")
