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

# %%

# [Keep your setup code and data loading code]
sns.set_style("white")
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

# Set working directory
if "ca4_ksat_celestial_classifier" in os.getcwd():
    os.chdir("..")  # Go up one level if we're in CA3

print(f"Working directory now: {os.getcwd()}")

# Load data
train_path = os.path.join("ca4_ksat_celestial_classifier", "assets", "train.csv")
test_path = os.path.join("ca4_ksat_celestial_classifier", "assets", "test.csv")

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


# %% Selecting features to keep

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
print("There are still values with -9999, will investigate this further. This might be placeholder values, lets check")


# %%
negative_counts = (train_df.drop(columns=["class"]) == -9999).sum()
print("Negative value counts:")
print(negative_counts)

print("Yes there is exactly one value with -9999 in the u, g and z columns. Will remove these")

# %% Replacing them with nan, and checking that they are now removed
train_df = train_df.replace(-9999, np.nan)

train_df = train_df.dropna()

print(train_df.isnull().sum())
# %% New eda on cleaned up data
print(train_df.describe())
# %% Checking redshift values between galaxies and quasars

galaxy_df = train_df[train_df["class"] == "GALAXY"]
qso_df = train_df[train_df["class"] == "QSO"]
print(f"GALAXY REDSHIFT:\n{galaxy_df['redshift'].describe()}")
print(f"QUASAR REDSHIFT:\n{qso_df['redshift'].describe()}")
# %% Encode class labels
class_mapping = {"GALAXY": 0, "QSO": 1, "STAR": 2}
train_df["class_labels"] = train_df["class"]  # Keep original text labels
train_df["class"] = train_df["class"].map(class_mapping)  # Replace with numeric codes


# %% Checking class distribution
print(train_df["class"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")

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

print(plot_df_sample["class_labels"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")

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
fig.subplots_adjust(hspace=0.4)
plt.savefig("boxplot")
plt.show()

print("""
    Based on the boxplot quasars has the most outliers.
    """)


# %% EDA - Distribution plots

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.violinplot(
        data=plot_df_sample,
        x="class_labels",  # Categories on x-axis
        y=feature,
        hue="class_labels",
        palette="Set1",
        ax=axes[i],
        alpha=0.4,
        orient="v",
    )
    axes[i].set_title(f"Violinplot of {feature}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis="x", labelsize=14)
    axes[i].patch.set_edgecolor("black")
    axes[i].patch.set_linewidth(1)
# Adjust layout
plt.tight_layout()
fig.subplots_adjust(hspace=0.2)
plt.savefig("histplot")
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
samples = [plot_df_sample[plot_df_sample["class_labels"] == cls][features] for cls in classes]

# Set up the figure and axes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
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


# %% [markdown]
# It is a good rule of thumb to keep correlations that is not too large since this can create redundacy and overfitting. Based on this rule of thumb we should use correlations between 0.3 and 0.7
#
# based on this we will keep u, g, z, redshift. We will also create color indices


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


# creating a definition we can use later to create new feature datasets
def feature_engineering(df):
    """Adding astronomical color indices to the dataframe."""
    df_features = df.copy()

    df_features["r-z"] = df_features["r"] - df_features["z"]
    df_features["g-r"] = df_features["g"] - df_features["r"]
    df_features["u-r"] = df_features["u"] - df_features["r"]
    df_features["i-z"] = df_features["i"] - df_features["z"]
    df_features["i-r"] = df_features["i"] - df_features["r"]
    df_features["u-g"] = df_features["u"] - df_features["g"]
    df_features["g-i"] = df_features["g"] - df_features["i"]

    df_features["redshift_squared"] = df_features["redshift"] ** 2
    df_features["log_redshift"] = np.log1p(df_features["redshift"])

    # Calculate average magnitude across multiple bands
    df_features["brightness_avg"] = df_features[["u", "g", "r", "i", "z"]].median(axis=1)

    # Simple multiplication interactions
    df_features["redshift_u"] = df_features["redshift"] * df_features["u"]

    # Measure of contrast between adjacent bands
    df_features["spectral_contrast"] = df_features[["u-g", "g-r", "i-r", "i-z"]].max(axis=1) - df_features[
        ["u-g", "g-r", "i-r", "i-z"]
    ].min(axis=1)

    return df_features


# %%

# creating new datasets with features

train_features_df = feature_engineering(train_df)

print(train_features_df.head())


# %% EDA - Boxplot of raw data by feature
features_more = train_features_df.columns.drop(["class", "class_labels"])
classes = train_features_df["class_labels"].unique()

print(features_more)

print(
    "There are a lot of observations in the original dataset, we therefore extract a sample of 5000 to visualize our data. This still represents our data well, but its less taxing to compute and plot."
)

sample_size = 5000
plot_df_balanced = train_features_df.columns.drop(["class"])
plot_df_sample_balanced = train_features_df.sample(n=sample_size, random_state=42)

print(plot_df_sample_balanced.head())


fig, axes = plt.subplots(17, 2, figsize=(15, 150))
axes = axes.flatten()
for i, feature in enumerate(features_more):
    sns.violinplot(
        data=train_features_df,
        x="class_labels",  # Categories on x-axis
        y=feature,
        hue="class_labels",
        palette="Set1",
        ax=axes[i],
        alpha=0.4,
        orient="v",
    )
    axes[i].set_title(f"Violinplot of {feature}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis="x", labelsize=14)
    axes[i].patch.set_edgecolor("black")
    axes[i].patch.set_linewidth(1)
# Adjust layout
plt.tight_layout()
fig.subplots_adjust(hspace=0.2)
plt.savefig("violinplot")
plt.show()

# %%
# checking new class balance


print(train_features_df["class"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")
print(train_features_df.info())


# %% Creating test split data set with sample size for faster training

X = train_features_df.drop(columns=["class", "class_labels"])
y = train_features_df["class"]

# Create an initial feature selection split
X_train, X_test, y_train, y_test = msel.train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
# %% Pipeline setup for random forest with pca
random_forest = ens.RandomForestClassifier(n_jobs=-1, random_state=42)

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

# Since data is unbalanced we will use stratifiedkfold
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


#


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
best_max_features = random_search.best_params_["random_forest__max_features"]
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
