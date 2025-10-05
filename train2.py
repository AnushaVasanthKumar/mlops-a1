# train2.py
from sklearn.kernel_ridge import KernelRidge
from misc import load_boston_manual, make_xy, evaluate_cv, train_test_holdout_mse

if __name__ == "__main__":
    df = load_boston_manual()
    X, y = make_xy(df, "MEDV")

    model = KernelRidge(
        alpha=1.0,
        kernel="rbf",
        gamma=None # if None, scikit-learn uses 1/n_features by default; you can tune
    )

    cv_mse = evaluate_cv(model, X, y, cv_splits=5)
    test_mse = train_test_holdout_mse(model, X, y, test_size=0.2, random_state=42, scale=True)

    print(f"[KernelRidge] 5-fold CV average MSE: {cv_mse:.4f}")
    print(f"[KernelRidge] Holdout test MSE:      {test_mse:.4f}")
