# 파일: src/main_sleep_stress.py

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use("seaborn-v0_8")
sns.set(font_scale=1.2)

RANDOM_STATE = 42


# =========================
# 1. 데이터 로드 / 기본 정보
# =========================

def load_data():
    """CSV 데이터 불러오기"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "expanded_sleep_health_dataset.csv")
    df = pd.read_csv(data_path)
    return df

def basic_info(df):
    """데이터 기본 정보 출력"""
    print("\n===== 데이터 상위 5행 =====")
    print(df.head())
    print("\n===== info() =====")
    print(df.info())
    print("\n===== describe() =====")
    print(df.describe())

def check_missing_outliers(df):
    """결측치 / 이상치 간단 체크"""
    print("\n===== 결측치 개수 =====")
    print(df.isnull().sum())

    numeric_cols = ["Sleep Duration", "Physical Activity Level",
                    "Stress Level", "Quality of Sleep",
                    "Heart Rate", "Daily Steps", "Age"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()


# =========================
# 2. 연구 주제 관련 EDA 그래프
# =========================

def plot_stress_vs_sleep(df):
    """스트레스 vs 수면 시간 산점도"""
    if {"Sleep Duration", "Stress Level"}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x="Sleep Duration", y="Stress Level", alpha=0.6)
        plt.title("Stress Level vs Sleep Duration")
        plt.xlabel("Sleep Duration (hours)")
        plt.ylabel("Stress Level (1-10)")
        plt.tight_layout()
        plt.show()

def plot_stress_vs_activity(df):
    """스트레스 vs 신체 활동량 산점도"""
    if {"Physical Activity Level", "Stress Level"}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x="Physical Activity Level", y="Stress Level", alpha=0.6)
        plt.title("Stress Level vs Physical Activity Level")
        plt.xlabel("Physical Activity Level (minutes/day)")
        plt.ylabel("Stress Level (1-10)")
        plt.tight_layout()
        plt.show()

def plot_corr_heatmap(df):
    """수면·운동·스트레스 관련 변수들 상관관계 히트맵"""
    cols = ["Sleep Duration", "Quality of Sleep",
            "Physical Activity Level", "Stress Level",
            "Heart Rate", "Daily Steps", "Age"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap (Sleep, Activity, Stress and Health)")
    plt.tight_layout()
    plt.show()


# =========================
# 3. 전처리
# =========================

def preprocess_data(df):
    """
    전처리:
    - 불필요 컬럼 제거(Person ID 등)
    - 결측치 처리
    - 범주형 One-Hot 인코딩
    - 타깃/피처 분리
    - 학습/테스트 분리
    - 스케일링(선형회귀용)
    """
    df = df.copy()

    # 1) 불필요 컬럼 제거
    drop_cols = []
    if "Person ID" in df.columns:
        drop_cols.append("Person ID")
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 2) 결측치 처리 (수치형: 평균, 범주형: 최빈값)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "str"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3) 범주형 원-핫 인코딩
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4) 타깃/피처 분리
    if "Stress Level" not in df_encoded.columns:
        raise ValueError("타깃 컬럼 'Stress Level' 이(가) 데이터에 없습니다.")

    X = df_encoded.drop("Stress Level", axis=1)
    y = df_encoded["Stress Level"]

    # 5) 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 6) 스케일링 (선형회귀용)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X.columns


# =========================
# 4. 모델 학습
# =========================

def train_linear_regression(X_train_scaled, y_train):
    """선형회귀 베이스라인 모델"""
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    return lr

def train_random_forest(X_train, y_train):
    """기본 랜덤포레스트 회귀 모델"""
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf

def tune_random_forest(X_train, y_train):
    """RandomForestRegressor 하이퍼파라미터 튜닝 (GridSearchCV)"""
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\n===== RandomForest 하이퍼파라미터 튜닝 결과 =====")
    print("최적 파라미터:", grid_search.best_params_)
    print("교차검증 최고 점수(음수 RMSE):", grid_search.best_score_)

    best_rf = grid_search.best_estimator_
    return best_rf, grid_search


# =========================
# 5. 평가 / 비교 / 해석
# =========================

def compare_models(lr, rf,
                   X_test, X_test_scaled,
                   y_test):
    """선형회귀 vs 랜덤포레스트 성능 비교"""
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test)

    def metrics(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    rmse_lr, mae_lr, r2_lr = metrics(y_test, y_pred_lr)
    rmse_rf, mae_rf, r2_rf = metrics(y_test, y_pred_rf)

    results = pd.DataFrame({
        "Model": ["LinearRegression", "RandomForestRegressor"],
        "RMSE": [rmse_lr, rmse_rf],
        "MAE": [mae_lr, mae_rf],
        "R2": [r2_lr, r2_rf]
    })

    print("\n===== 모델 성능 비교 =====")
    print(results)

    # 실제 vs 예측 (랜덤포레스트) 산점도
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.5, label="RandomForest")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", label="Ideal")
    plt.xlabel("Actual Stress Level")
    plt.ylabel("Predicted Stress Level")
    plt.title("Actual vs Predicted Stress Level (RandomForest)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results, y_pred_lr, y_pred_rf

def plot_feature_importance(rf, feature_names, top_n=15):
    """랜덤포레스트 변수 중요도 상위 N개 시각화"""
    importances = rf.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi, x="importance", y="feature")
    plt.title(f"Top {top_n} Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.show()


# =========================
# 6. main
# =========================

def main():
    # 1) 데이터 불러오기 및 기본 확인
    df = load_data()
    basic_info(df)
    check_missing_outliers(df)

    # 2) 연구 주제 관련 EDA 그래프
    plot_stress_vs_sleep(df)
    plot_stress_vs_activity(df)
    plot_corr_heatmap(df)

    # 3) 전처리 + 분할
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, feature_names) = preprocess_data(df)

    # 4) 모델 학습
    lr = train_linear_regression(X_train_scaled, y_train)
    rf_base = train_random_forest(X_train, y_train)

    # 5) 랜덤포레스트 하이퍼파라미터 튜닝
    rf_best, grid_search = tune_random_forest(X_train, y_train)

    # 튜닝 전/후 랜덤포레스트 비교
    print("\n===== 튜닝 전/후 RandomForest 성능 =====")
    for name, rf in [("Base RF", rf_base), ("Tuned RF", rf_best)]:
        y_pred = rf.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(name, "RMSE:", rmse, "MAE:", mae, "R2:", r2)

    # 6) 선형회귀 vs 튜닝된 랜덤포레스트 최종 비교
    results, y_pred_lr, y_pred_rf = compare_models(
        lr, rf_best, X_test, X_test_scaled, y_test
    )

    # 7) 변수 중요도
    plot_feature_importance(rf_best, feature_names, top_n=15)

    # 8) 실제값 vs 예측값 일부 비교 출력
    comparison_df = pd.DataFrame({
        "Actual": y_test.values,
        "Pred_LR": y_pred_lr,
        "Pred_RF": y_pred_rf
    }).head(10)
    print("\n===== 실제값 vs 예측값 (일부) =====")
    print(comparison_df)


if __name__ == "__main__":
    main()
