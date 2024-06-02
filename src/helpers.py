import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, f1_score, recall_score, make_scorer, brier_score_loss, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
sns.set(style='darkgrid',)
color_pal = sns.color_palette("muted")
sns.set_palette(color_pal)
sns.set_context("paper")

# plot dimensions
plot_width = 12
plot_height = 8
palette_success ={0: color_pal[1], 1: color_pal[2]}

def cyclical_encoding(value, denominator):
    """
    Encodes cyclical time features into sine- & cosine-values.

    Parameters:
    -------------
    value: to be encoded as numerical
    denominator: as numerical reflects frequence or intervall (e.g. 24 for h a day)

    Return:
    -------------
    value_sin: sine-value
    value_cos: cosine-value
    """
    value_sin = np.sin(2 * np.pi * value / denominator)
    value_cos = np.cos(2 * np.pi * value / denominator)

    return [value_sin, value_cos]


def binning_h (h):
    # bucketizing or binning of h into daytime: night, morning, afternoon, evening
    daytime = "n/a"
    
    if (h >= 6) and (h < 12):
        daytime = "morning"
    elif (h >= 12) and (h < 18):
        daytime = "afternoon"
    elif (h >= 18) and (h < 24):
        daytime = "evening"
    else:
        daytime = "night"

    return daytime



def track_model(training_info, model, x_test, y_test, y_pred, y_pred_auc, feature_names, rs):
    """
    Logging all relevant information (metrics, parameters and feature importances) of a trained model to MLflow.
    Model-specific feature importances is used (if existing) and Permutation Feature Importance.
    """
    # Set a tag as detailed description of run
    mlflow.set_tag("Training Info", training_info)
    
    # Log the hyperparameters
    mlflow.log_params(model.get_params())
    
    # Log the metrics
    mlflow.log_metrics({"AUC": roc_auc_score(y_test, y_pred_auc), 
                       "Precision": precision_score(y_test, y_pred, zero_division=np.nan), 
                       "Recall": recall_score(y_test, y_pred, zero_division=np.nan), 
                       "F1-score": f1_score(y_test, y_pred, zero_division=np.nan)})
     
    # check options to find out model-specific FEATURE IMPORTANCES
    plot_title = "No information on feature importances"
    if (hasattr(model, "feature_importances_") == True):
        feat_imp = model.feature_importances_
        plot_title = "Feature importances"
    elif (hasattr(model, "coef_") == True):
        feat_imp = model.coef_[0]
        plot_title = "Coefficients"
    else:
        feat_imp = len(feature_names) * [0]
        print("Has NO feature importances")
    
    # save feature importance to DF
    df_feat_imp = (pd.DataFrame(list(zip(feature_names, feat_imp)), 
                             columns=["Feature", "Model_specific_imp"])
                .set_index("Feature"))
    
    # PERMUTATION FEATURE IMPORTANCE (PFI) 
    pfi_result = permutation_importance(estimator=model, X=x_test, y=y_test, scoring="roc_auc", random_state=rs, n_repeats=10)
    df_pfi = (pd.DataFrame(list(zip(feature_names, pfi_result.importances_mean)), 
                           columns=["Feature", "PFI"])
              .set_index("Feature"))
    # join both dataframes
    df_feat_imp = df_feat_imp.join(df_pfi, how="left").sort_values("PFI", ascending=False).reset_index()
    
    csv_path = "artifacts/feature_importance.csv"
    df_feat_imp.to_csv(csv_path, index=True)
    mlflow.log_artifact(csv_path, "feature_importance.csv")
    
    # Log plot of feature importance
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    sns.barplot(y="Feature", x="Model_specific_imp", data=df_feat_imp, ax=ax[0])
    ax[0].set_title("Model-specific feature importances", fontsize=10)
    sns.barplot(y="Feature", x="PFI", data=df_feat_imp, ax=ax[1])
    ax[1].set_title("Permutation feature importances", fontsize=10)
    #plt.title(plot_title)
    #plt.xlabel("Importance")
    #plt.ylabel("Feature")
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1, wspace=0.6)
    mlflow.log_figure(fig, "feature_importance.png")
    
    
    # Infer the model signature
    #signature = infer_signature(x_train, y_train)
    
    # Log the model
    model_info = mlflow.sklearn.log_model(sk_model=model, 
                                          artifact_path="model_saved")




def train_model(zip_iter, x_train_raw, x_train_numerical, x_test_raw, x_test_numerical, y_train, y_test, features_numerical, exp_name, info_text, rs):
    """
    Iterates over a list of classifiers and parameters to train models. The trained models are tracked in MLflow.
    """
    for n, c, fm in zip_iter:
        print(f"Training and logging of model {n} as classifier {c} with feature mode {fm}")
        if fm == "raw":
            x_train = x_train_raw
            x_test = x_test_raw
            feature_names = list(x_train_raw.columns)
        else:
            x_train = x_train_numerical
            x_test = x_test_numerical
            feature_names = features_numerical
            
        # Create a new MLflow Experiment
        mlflow.set_experiment(exp_name)
        
        # Start an MLflow run
        run_name = n + "_" + exp_name
        with mlflow.start_run(run_name=run_name) as run:
            training_info = n + " - " + info_text
            # Training and prediction
            model = c
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
    
            if (hasattr(model, "decision_function") == True):
                y_pred_auc = model.decision_function(x_test)
            else:
                y_pred_auc = model.predict_proba(x_test)[:, 1]
    
            track_model(training_info, model, x_test, y_test, y_pred, y_pred_auc, feature_names, rs)
    
            run_id = run.info.run_id
        
        print(f"For {n}: Run logged in MLflow with {run_id}.")
        print("---------------------- \n")



def hp_tune_model(search_mode, zip_iter, 
                  x_train_raw, x_train_numerical, 
                  x_test_raw, x_test_numerical, 
                  y_train, y_test, 
                  features_numerical, exp_name, info_text, rs):
    """
    Iterates over a list of classifiers and parameters to train models. The trained models are tracked in MLflow.
    Choose between RandomSearchCV and GridSearchCV.
    search_mode = random/grid
    """
    
    # Config RandomizedSearchCV
    scorer_auc = make_scorer(roc_auc_score, response_method=["predict_proba", "decision_function"])
    scorer_precision = make_scorer(precision_score)
    scorer_recall = make_scorer(recall_score)
    scorer_f1 = make_scorer(f1_score)
    
    cv_scoring = {"auc": scorer_auc, 
                          "precision": scorer_precision, 
                          "recall": scorer_recall, 
                          "F1": scorer_f1}
    randomized_iter = 30
    cv_folds = 5
    
    for n, c, p, fm in zip_iter:
        print(f"Training and logging of model {n} as classifier {c} with feature mode {fm}")
        if fm == "raw":
            x_train = x_train_raw
            x_test = x_test_raw
            feature_names = list(x_train_raw.columns)
        else:
            x_train = x_train_numerical
            x_test = x_test_numerical
            feature_names = features_numerical

        # Create a new MLflow Experiment
        mlflow.set_experiment(exp_name)
    
        # Start an MLflow run
        run_name = n + "_BEST_" + exp_name
        with mlflow.start_run(run_name=run_name) as run:
            training_info = n + " - " + info_text

            # RandomizedSearchCV or GridSearchCV
            if (search_mode == "random"):
                random_search = RandomizedSearchCV(c, 
                                                   param_distributions=p, 
                                                   n_iter=randomized_iter, 
                                                   verbose=2, 
                                                   cv=cv_folds, 
                                                   scoring=cv_scoring, 
                                                   refit="auc", 
                                                   random_state=rs, 
                                                   error_score="raise")
                # Training and prediction
                random_search.fit(x_train, y_train)
                model_best = random_search.best_estimator_
            else:
                grid_search = GridSearchCV(c, 
                                           param_grid=p, 
                                           scoring=cv_scoring, 
                                           refit="auc", 
                                           cv=cv_folds, 
                                           verbose=2, 
                                           error_score="raise")
                # Training and prediction
                grid_search.fit(x_train, y_train)
                model_best = grid_search.best_estimator_
                
            y_pred = model_best.predict(x_test)

            if (hasattr(model_best, "decision_function") == True):
                y_pred_auc = model_best.decision_function(x_test)
            else:
                y_pred_auc = model_best.predict_proba(x_test)[:, 1]
    
            track_model(training_info, model_best, x_test, y_test, y_pred, y_pred_auc, feature_names, rs)
    
            run_id = run.info.run_id
        
        print(f"For {n}: Run logged in MLflow with {run_id}.")
        print("---------------------- \n")