# train.py
from sklearn.tree import DecisionTreeRegressor
from misc import load_boston_manual, make_xy, evaluate_cv, train_test_holdout_mse

if __name__ == "__main__":
    df = load_boston_manual()
    X, y = make_xy(df, "MEDV")

    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=None,      # can tune later
        min_samples_split=2
    )

    cv_mse = evaluate_cv(model, X, y, cv_splits=5)
    test_mse = train_test_holdout_mse(model, X, y, test_size=0.2, random_state=42, scale=True)

    print(f"[DecisionTreeRegressor] 5-fold CV average MSE: {cv_mse:.4f}")
    print(f"[DecisionTreeRegressor] Holdout test MSE:      {test_mse:.4f}")
