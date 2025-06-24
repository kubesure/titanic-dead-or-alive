
from sklearn.metrics import precision_recall_curve,PrecisionRecallDisplay, auc
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame,Series



#Get numerical and categorical features from dataframe
def get_numerical_and_categorical_features(X_train):
    numerical_features = X_train.select_dtypes(exclude=['object']).columns.to_list()
    categorical_features = X_train.select_dtypes(include=['object']).columns.to_list()
    return (numerical_features,categorical_features)

#Check for categorical features if they have either too few unique values or too many as both do not provide enough 
#signal to the model

def filter_categorical_features(X_train,categorical_features):
    filter_more_than_one = [f for f in categorical_features if X_train[f].nunique() > 1]
    unique_ratios = X_train[filter_more_than_one].nunique(dropna=True) / len(X_train)
    filtered_columns = unique_ratios[unique_ratios <= 0.99].index.to_list()
    return filtered_columns

# Check numerical featrues whose variance is too low as the featuure is too constant and will not provide signal
# to model

def filter_numerical_features(X_train,numerical_features, model_features):
    variances = X_train[model_features].var()
    # Find features with very low variance (e.g., less than 0.01)
    low_variance_features = variances[variances <0.01].index.tolist()
    return low_variance_features


#features that tpp corelated do not provide additional lift to the model but confuse how would the model prioritize
# one feature over other if they provide same input. this method provide possible featrues that should be removed

def get_highly_correlated_pairs(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = upper.stack().reset_index() 
    high_corr.columns = ['feature1', 'feature2', 'corr']
    high_corr = high_corr[high_corr['corr'] >= threshold]
    #print(high_corr.sort_values(by='corr', ascending=False).reset_index(drop=True))
    return list(zip(high_corr['feature1'], high_corr['feature2']))

def evaluate_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model: XGBClassifier,
    model_features
):
    
    y_train_pred = model.predict_proba(X_train[model_features])[:, 1]
    y_val_pred = model.predict_proba(X_val[model_features])[:, 1]
    y_test_pred = model.predict_proba(X_test[model_features])[:, 1]

    train_precision, train_recall, _ = precision_recall_curve(y_train, y_train_pred)
    val_precision, val_recall, _ = precision_recall_curve(y_val, y_val_pred)
    test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_pred)

    train_auc = round(auc(train_recall, train_precision), 5)
    val_auc = round(auc(val_recall, val_precision), 5)
    test_auc = round(auc(test_recall, test_precision), 5)

    # 4. Print AUCs
    print(f"Train AUC (PR): {train_auc}")
    print(f"Validation AUC (PR): {val_auc}")
    print(f"Test AUC (PR): {test_auc}")

    # 5. Plot all on same PR graph
    fig, ax = plt.subplots(figsize=(10, 5))
    PrecisionRecallDisplay.from_estimator(model, X_train[xgb_model.get_booster().feature_names], y_train, ax=ax, name="Train")
    PrecisionRecallDisplay.from_estimator(model, X_val[xgb_model.get_booster().feature_names], y_val, ax=ax, name="Validation")
    PrecisionRecallDisplay.from_estimator(model, X_test[xgb_model.get_booster().feature_names], y_test, ax=ax, name="Test")
    plt.title("Precision-Recall Curve for Train, Validation, and Test")
    plt.grid(True)
    plt.show()


 # create train test validation data sets 
def create_train_val_test_split(
    data: pd.DataFrame,
    target_col: str,
    group_col: str = "id",
    train_size: float = 0.64,
    val_size: float = 0.16,
    test_size: float = 0.20,
    random_state: int = 42
) -> tuple:
    """
    Splits the dataset into train, validation, and test using group-based splitting.
    Returns X and y for each split, and a stats DataFrame.
    """

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # First split: Train+Val vs Test
    gss1 = GroupShuffleSplit(n_splits=1, 
                             train_size=(train_size + val_size), 
                             test_size=test_size, 
                             random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(X, y, groups=data[group_col]))

    train_val_data = data.iloc[train_val_idx]
    test_data = data.iloc[test_idx]

    # Second split: Train vs Val from Train+Val
    gss2 = GroupShuffleSplit(n_splits=1, train_size=train_size / (train_size + val_size),
                             test_size=val_size / (train_size + val_size), random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_data.drop(columns=[target_col]),
                                         train_val_data[target_col],
                                         train_val_data[group_col]))

    train_data = train_val_data.iloc[train_idx]
    val_data = train_val_data.iloc[val_idx]

    # Build statistics for each split
    df_stats = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Rows": [len(train_data), len(val_data), len(test_data)],
        "Positive Rate": [
            train_data[target_col].mean(),
            val_data[target_col].mean(),
            test_data[target_col].mean()
        ]
    })

    return (
        train_data.drop(columns=[target_col]),
        train_data[target_col],
        val_data.drop(columns=[target_col]),
        val_data[target_col],
        test_data.drop(columns=[target_col]),
        test_data[target_col],
        df_stats
    )



   

