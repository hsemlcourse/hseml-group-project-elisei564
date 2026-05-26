import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    from src.preprocessing import SEED, prepare_target
except (ImportError, ModuleNotFoundError):
    from preprocessing import SEED, prepare_target


def evaluate(name, y_true, y_pred, results=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(
        np.abs((np.expm1(y_true) - np.expm1(y_pred)) /
               (np.expm1(y_true) + 1e-9))
    ) * 100
    metrics = {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
    if results is not None:
        results.append(metrics)
    return metrics


def train_final_stacking(X, y):
    print("Начало обучения финального Stacking-ансамбля...")
    stack = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(
                n_estimators=200, max_depth=10, n_jobs=-1, random_state=SEED)),
            ("xgb", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, random_state=SEED, verbosity=0)),
            ("lgbm", LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                subsample=0.8, random_state=SEED, verbose=-1)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )
    stack.fit(X, y)
    print("Обучение завершено.")
    return stack


def run_cp2(data_dir: str = "data/processed", models_dir: str = "models"):
    os.makedirs(models_dir, exist_ok=True)

    # ── league_rank_dict сохраняется в 01_eda.ipynb, здесь только загружаем ──
    # C_League отсутствует в processed CSV — она удаляется в encode_data
    league_rank_path = os.path.join(models_dir, "league_rank_dict.joblib")
    if not os.path.exists(league_rank_path):
        raise FileNotFoundError(
            f"Файл {league_rank_path} не найден.\n"
            "Добавь в 01_eda.ipynb после build_league_rank(train_raw):\n"
            "  joblib.dump(league_rank, '../models/league_rank_dict.joblib')"
        )
    print(f"league_rank_dict загружен: {league_rank_path}")

    print(f"Загрузка данных из {data_dir}...")
    train = pd.read_csv(f"{data_dir}/train.csv")
    val = pd.read_csv(f"{data_dir}/val.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    X_trainval, y_trainval = prepare_target(pd.concat([train, val]))
    X_test, y_test = prepare_target(test)

    model = train_final_stacking(X_trainval, y_trainval)

    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(model, model_path)
    print(f"Модель сохранена: {model_path}")

    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЙ ТЕСТ НА ОТЛОЖЕННОЙ ВЫБОРКЕ")
    print("=" * 60)
    f = evaluate("Stacking (Final)", y_test, model.predict(X_test))
    print(
        f"RMSE: {f['rmse']:.4f} | MAE: {f['mae']:.4f} | "
        f"R²: {f['r2']:.4f} | MAPE: {f['mape']:.1f}%"
    )


if __name__ == "__main__":
    run_cp2()
