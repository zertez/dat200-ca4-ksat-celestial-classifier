# %% [markdown]
# # CA4
#
# Group 37
#
# Group members:
# * Jannicke Ådalen
# * Marcus Dalaker Figenschou
# * Rikke Sellevold Vegstein

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
    linear_model as lm,
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
    svm as svm,
)

# %%
# Setting the styles of plots so that they have same styling throughout
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

# %% [markdown]
# # Data inspection and cleaning

# %%
print("---TRAIN DATA---")
train_df.info()
print("---TEST DATA---")
test_df.info()

# %% [markdown]
# ## Metadata
# A lot of metadata in this dataset (data that isnt physical properties) that is just used to organize the data for the future by astronomers. Seems also that u might be missing values since it has fewer entries and shows up as non-null, will investigate after metadata removal.

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
# Two things to see here. From the .info we can see that there are fewer observations of u than the rest and we have a min value -9999 for u, g and z which only shows up in train data but not in test data. But in our original data inspection we found no missing values, so this might indicate placeholder.

# %%
print(train_df.isnull().sum())

# %% [markdown]
#
# We can now see that u is actually missing 362 values. Because of our datasize of 80k, we can comfortably remove 362 samples from our dataset of 80k.

# %% dropping na
train_df = train_df.dropna()

print(train_df.isnull().sum())

# %% Data description after na removal
print("--- Train Data ---")
train_df.info()

print(train_df.describe())

# %% [markdown]
# There are still values with -9999, will investigate this further. This might be placeholder values, lets check.


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


# %% Checking class distribution
print(train_df["class"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")


# %% [markdown]
# The class distribution is heavily skewed towards galaxies. Roghly 60 % of the dataset are galaxies. We can balance this out by downsampling galaxies, we will do this later just before we split our data into test/val.


# %% [markdown]
# # Data plotting

# %% [markdown]
# Before we go forward with plotting we will map galaxy, quasars and stars to their given values for submission. We will keep the original labels since this is more descriptive when we plot the data

# %% Encode class labels
class_mapping = {"GALAXY": 0, "QSO": 1, "STAR": 2}
# Keep original text labels for plotting
train_df["class_labels"] = train_df["class"]
# Replaceing with numeric codes
train_df["class"] = train_df["class"].map(class_mapping)


# %% [markdown]
# Since there is ~ 79k samples in our dataset we choose to extract a random sample of 5000. This is statistically enought to visualize the data distribution without needing to visualize ~ 79k samples which for some plots takes a lot of time.

# %%
features = train_df.columns.drop(["class", "class_labels"])
classes = train_df["class_labels"].unique()

sample_size = 5000
plot_df = train_df.columns.drop(["class"])
plot_df_sample = train_df.sample(n=sample_size, random_state=42)

# %% [markdown]
# ## Boxplot

# %%
# %% EDA - Boxplot of raw data by feature

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
#
# Based on the boxplot quasars has the most outliers.

# %% [markdown]
# ## Violinplot

# %% EDA - Distribution plots
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.violinplot(
        data=plot_df_sample,
        x="class_labels",
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
plt.show()

# %% [markdown]
# The violinplot shows that galaxy is biomodal for all features. Star is somewhat biomodal for features: g, r and i. Quasars is only unimodal.
#
# It is interesting that galaxies are biomodal for all features, this might be due to galaxies come in different sizes and shapes, and also have varying degrees of luminosity.
#
# That some features show stars as biomodal is also to be expected as stars vary in size and shape, but we did believe that this effect should be stronger than it is .
#
# The quasars are unimodal for all features.


# %% [markdown]
# ## Feature Correlation Matrix

# %% Corr matrix for all classes
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
# From the correlation matrix we can see that redshift will be our most important feature when it comes to seperating the three categories.

# %% [markdown]
# # Creating New Features

# %% [markdown]
# Now that we have a balanced dataset, we can create new features based on color indices—a method astronomers use to assess the light intensity of celestial objects. We must also remember to add these features to our original training data and test data.

# %% [markdown]
# We can group the filters based on thier photometric system like this:
#
# Visible light:
# - g(green filter)
# - r(red filter)
#
# Ultraviolet light:
# - u(ultraviolet)
#
# Infrared spectrum:
# - i(near infrared)
# - z(infrared)


# %% Feature engineering
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

    # redshift is the feature that seems to seperate the classes best so we will create some more

    df_features["redshift_squared"] = df_features["redshift"] ** 2
    df_features["log_redshift"] = np.log1p(df_features["redshift"])

    # Simple multiplication interactions
    df_features["redshift_u"] = df_features["redshift"] * df_features["u"]

    # We try to create a brightness feature here to
    df_features["spectral_contrast"] = df_features[["u-g", "g-r", "i-r", "i-z"]].max(axis=1) - df_features[
        ["u-g", "g-r", "i-r", "i-z"]
    ].min(axis=1)

    return df_features


# %%
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# %% [markdown]
# We ran first a random forest model with all the features above and then we plotted the cumuluative feature importances and chose to keep the features that had ~95% relevance for our model, these are the features we we ended up with.

# %%
top_features = [
    "redshift_squared",
    "log_redshift",
    "redshift",
    "redshift_u",
    "g-i",
    "g-r",
    "spectral_contrast",
    "i-r",
    "r-z",
    "u-r",
    "g",
    "class",
    "class_labels",
]

# %%
train_df_rf = train_df[top_features]
print(train_df_rf)

test_df_rf = test_df[top_features[:-2]]
print(test_df_rf.info())

# %% [markdown]
# # Training the models
# Since we are working with three categorical target values and a large dataset we have chosen to go for the following models:
# - Random Forest Classifier
# - Logistic Regression model
# - SVC

# %% Creating test split data set with sample size for faster training
X = train_df.drop(columns=["class", "class_labels"])
y = train_df["class"]

# Splitting the data into train and test data with a 60/40 split
X_train, X_test, y_train, y_test = msel.train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

print(X_train)

# %% [markdown]
# ## Random Forest Classifier

# %% [markdown]
# Random forest is especially good for unbalanced data and multi category classification problems.

# %%
random_forest = ens.RandomForestClassifier(n_jobs=-1, random_state=42)

rf_pipe = pipe.Pipeline([("random_forest", random_forest)])


# Parameter distributions for random search
param_grid_rf = {
    "random_forest__n_estimators": [100, 200, 300, 400, 500],
    "random_forest__max_depth": [5, 10, 15, 20, 25, None],
    "random_forest__min_samples_split": [2, 5, 10, 15, 20],
    "random_forest__max_features": ["sqrt", "log2"],
    "random_forest__criterion": ["gini", "entropy", "log_loss"],
}

# Since data is unbalanced we will use stratifiedkfold
stratified_cv = msel.StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# We will use randomizedsearchCV since we are doing hyperparamter tuning
rf_search = msel.RandomizedSearchCV(
    rf_pipe,
    param_distributions=param_grid_rf,
    n_iter=100,
    # Using StratifiedKFold
    cv=stratified_cv,
    n_jobs=-1,
    scoring="f1_macro",
    return_train_score=True,
)

rf_search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % rf_search.best_score_)
print(rf_search.best_params_)


# Print best results
print("Best parameter (CV score=%0.3f):" % rf_search.best_score_)
print(rf_search.best_params_)

# %% Evaluate on test set
# Evaluate on test set
best_rf_model = rf_search.best_estimator_
y_test_pred_rf = best_rf_model.score(X_test, y_test)
print(f"Test set score with best model: {y_test_pred_rf:.3f}")
print(best_rf_model)
# %% Detailed evaluation on test set from sample
# Get predictions on the sample test set
y_pred_rf = best_rf_model.predict(X_test)

# Print classification report
print("Classification Report on Sample Test Set:")
print(met.classification_report(y_test, y_pred_rf))

# Create confusion matrix
print("Confusion Matrix on Sample Test Set:")
conf_matrix = met.confusion_matrix(y_test, y_pred_rf)
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_rf_model.classes_,
    yticklabels=best_rf_model.classes_,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Sample Test Set)")
plt.tight_layout()
plt.show()

# %% Extract best parameters from the sample model
# Extract best parameters from the best model
best_n_estimators = rf_search.best_params_["random_forest__n_estimators"]
best_max_depth = rf_search.best_params_["random_forest__max_depth"]
best_min_samples_split = rf_search.best_params_["random_forest__min_samples_split"]
best_max_features = rf_search.best_params_["random_forest__max_features"]
best_criterion = rf_search.best_params_["random_forest__criterion"]
# %% Create a pipeline with the best parameters
print("Training final model on the entire dataset...")

# Prepare full data
X_full = train_df.drop(columns=["class_labels", "class"])
y_full = train_df["class"]


final_rf = ens.RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_features=best_max_features,
    max_depth=best_max_depth,
    criterion=best_criterion,
    random_state=42,
    n_jobs=-1,  # Use all available cores
)


# Create the final pipeline with scaling
final_rf_pipeline = pipe.Pipeline([("random_forest", final_rf)])

# Train the pipeline on unscaled data - pipeline handles scaling internally
final_rf_pipeline.fit(X_full, y_full)

print(f"Final rf model trained on {len(X_full)} samples")

# %% [markdown]
# ### Prediction

# %%
y_test_rf = final_rf_pipeline.predict(test_df)

# Create the final dataframe with numeric class predictions
y_test_rf = pd.DataFrame(y_test_rf, columns=["class"])
y_test_rf.index.name = "ID"

# Add file path with appropriate naming related to model parameters
base_dir = os.path.join("CA4", "results")
os.makedirs(base_dir, exist_ok=True)
filename = "random_forest_model.csv"
file_path = os.path.join(base_dir, filename)

# Save to CSV
y_test_rf[["class"]].to_csv(file_path)
print(f"Saved rf submission to {file_path}")


# %% [markdown]
# ## Logistic Regression model without PCA

# %%
# Splitting the data into train and test data with a 60/40 split
X_train, X_test, y_train, y_test = msel.train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# %%
# %% Pipeline setup for logreg without

logreg = lm.LogisticRegression(n_jobs=-1, random_state=42)
std_scaler = prep.StandardScaler()

logreg_pipe = pipe.Pipeline([("scaler", std_scaler), ("logreg", logreg)])

param_grid_logreg = {
    "logreg__max_iter": [50, 100, 200, 300],
    "logreg__penalty": ["l2"],
    "logreg__solver": ["sag", "saga", "newton-cg"],
    "logreg__C": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}


# Since data is unbalanced we will use stratifiedkfold
stratified_cv = msel.StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

logreg_search = msel.RandomizedSearchCV(
    logreg_pipe,
    param_distributions=param_grid_logreg,
    n_iter=100,
    # Using StratifiedKFold
    cv=stratified_cv,
    n_jobs=-1,
    scoring="f1_macro",
    return_train_score=True,
)

logreg_search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % logreg_search.best_score_)
print(logreg_search.best_params_)

# %%
# Evaluate on test set
best_logreg = logreg_search.best_estimator_
y_test_pred = best_logreg.score(X_test, y_test)
print(f"Test set score with best model: {y_test_pred:.3f}")

# %%
# Detailed evaluation on test set from sample
# First, evaluate the sample model performance more thoroughly

# Get predictions on the sample test set
y_pred_logreg = best_logreg.predict(X_test)

# Print classification report
print("Classification Report on Sample Test Set:")
print(met.classification_report(y_test, y_pred_logreg))

# Create confusion matrix
print("Confusion Matrix on Sample Test Set:")
conf_matrix = met.confusion_matrix(y_test, y_pred_logreg)
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_logreg.classes_,
    yticklabels=best_logreg.classes_,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Sample Test Set)")
plt.tight_layout()
plt.show()

# %%
# Final model training on the entire training dataset
print("Training final model on the entire balanced dataset...")

# Prepare full data
X_full = train_df.drop(columns=["class_labels", "class"])
y_full = train_df["class"]

# %% Create a pipeline with the best parameters
from sklearn.linear_model import LogisticRegression

final_logreg = LogisticRegression(
    max_iter=logreg_search.best_params_["logreg__max_iter"],
    penalty=logreg_search.best_params_["logreg__penalty"],
    solver=logreg_search.best_params_["logreg__solver"],
    C=logreg_search.best_params_["logreg__C"],
    random_state=42,
    n_jobs=-1,  # Use all available cores
)

# Create the final pipeline with scaling
final_logreg_pipe = pipe.Pipeline([("scaler", min_max_scaler), ("pca", pca), ("logreg", final_logreg)])

# Train the pipeline on unscaled data - pipeline handles scaling internally
final_logreg_pipe.fit(X_full, y_full)
print(f"Final logistic model trained on {len(X_full)} samples")

# %%
# Make predictions using the pipeline (no additional scaling needed)

# The pipeline handles scaling internally
y_test_logreg = final_logreg_pipe.predict(test_df)

# Create the final dataframe with numeric class predictions
y_test_logreg = pd.DataFrame(y_test_logreg, columns=["class"])
y_test_logreg.index.name = "ID"

# Add file path with appropriate naming related to model parameters
base_dir = os.path.join("CA4", "results")
os.makedirs(base_dir, exist_ok=True)
filename = "logreg_model.csv"
file_path = os.path.join(base_dir, filename)

# Save to CSV
y_test_logreg.to_csv(file_path)  # Keep the index as it's set to "ID"
print(f"Saved logistic submission to {file_path}")

# %% [markdown]
# ## SVC Model With PCA

# %%
svm_model = svm.SVC(random_state=42)
pca = decomp.PCA()
std_scaler = prep.StandardScaler()


pipe_svm = pipe.Pipeline([("scaler", std_scaler), ("pca", pca), ("svm", svm_model)])

param_grid_svm = {
    "pca__n_components": [0.95],  # Add this if you want to tune PCA
    "svm__kernel": ["rbf"],
    "svm__decision_function_shape": ["ovr", "ova"],
    "svm__C": [0.1, 1, 10],
    "svm__degree": [1, 2],
    "svm__gamma": ["scale", "auto", 0.01, 0.1],
}

# Since data is unbalanced we will use stratifiedkfold
stratified_cv = msel.StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
svc_search = msel.RandomizedSearchCV(
    pipe_svm,
    param_distributions=param_grid_svm,
    n_iter=100,
    cv=stratified_cv,
    n_jobs=-1,
    scoring="f1_macro",
    return_train_score=True,
)
svc_search.fit(X_train, y_train)

# Print best results - fix the variable name
print("Best parameter (CV score=%0.3f):" % svc_search.best_score_)
print(svc_search.best_params_)  # Not search.best_params_

# %%
# %% Evaluate on test set
best_svc_model = svc_search.best_estimator_
y_test_pred = best_svc_model.score(X_test, y_test)
print(f"Test set score with best model: {y_test_pred:.3f}")


# %% Detailed evaluation on test set from sample
# First, evaluate the sample model performance more thoroughly

# Get predictions on the sample test set
y_pred_svc = best_svc_model.predict(X_test)

# Print classification report
print("Classification Report on Sample Test Set:")
print(met.classification_report(y_test, y_pred_svc))

# Create confusion matrix
print("Confusion Matrix on Sample Test Set:")
conf_matrix = met.confusion_matrix(y_test, y_pred_svc)
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_svc_model.classes_,
    yticklabels=best_svc_model.classes_,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Sample Test Set)")
plt.tight_layout()
plt.show()


# %%
# Final model training on the entire training dataset
print("Training final model on the entire balanced dataset...")
# Prepare full data
X_full = train_df.drop(columns=["class_labels", "class"])
y_full = train_df["class"]

final_svm = svm.SVC(
    kernel=svc_search.best_params_["svm__kernel"],
    C=svc_search.best_params_["svm__C"],
    degree=svc_search.best_params_["svm__degree"],
    gamma=svc_search.best_params_["svm__gamma"],
    random_state=42,
    # SVC doesn't support n_jobs parameter
)

# Create the final pipeline with scaling and PCA
final_svm_pipe = pipe.Pipeline([("scaler", std_scaler), ("pca", pca), ("svm", final_svm)])

# Train the pipeline on unscaled data - pipeline handles scaling internally
final_svm_pipe.fit(X_full, y_full)
print(f"Final SVM model trained on {len(X_full)} samples")

# %%
# Make predictions using the pipeline (no additional scaling needed)

# The pipeline handles scaling internally
y_test_svc = final_svm_pipe.predict(test_df)

# Create the final dataframe with numeric class predictions
y_test_svc = pd.DataFrame(y_test_svc, columns=["class"])
y_test_svc.index.name = "ID"

# Add file path with appropriate naming related to model parameters
base_dir = os.path.join("CA4", "results")
os.makedirs(base_dir, exist_ok=True)
filename = "svc_model.csv"
file_path = os.path.join(base_dir, filename)

# Save to CSV
y_test_svc.to_csv(file_path)  # Keep the index as it's set to "ID"
print(f"Saved svc submission to {file_path}")

# %% [markdown]
# Why we are getting all of these errors we do not know. We have tried to figure this out but we can't find a solution. From the confusion matrixes, all models handles star really good, this is mostlikely because redshift doesnt really apply to stars so its a really good feature for separation. All models struggle to seperate quasars from galaxies, this is most likely due to there beeing fewer quasars than galaxies in the dataset, and quasars do happen in the centre of galaxies. Overall our best model was the random forest model.
#
# Kaggle public score:
#
# rf: 0.976
# logreg: 0.958
# svc: 0.966
