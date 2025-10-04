import pandas as pd
import numpy as np
import joblib
import warnings
import os
import json
import io
import sys
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def run_training_pipeline(model_choice: str, custom_params: dict, input_csv_path: str):
    """
    The main training pipeline function, callable from other scripts like app.py.
    """
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    trained_artifacts = {
        "model": None,
        "imputer": None,
        "scaler": None,
        "feature_names": None
    }
    output_model_path = None
    # ### 修改開始 ###：初始化混淆矩陣變數
    confusion_matrix_data = None
    # ### 修改結束 ###

    try:
        print(f"--- 開始線上訓練流程：{model_choice} ---")
        print(f"使用資料集: {input_csv_path}")
        print(f"自訂超參數: {json.dumps(custom_params, indent=2)}")

        # --- 1. Data Loading and Preparation ---
        df = pd.read_csv(input_csv_path, dtype={'source_id': str})
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        training_df = df[df['disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
        if training_df.empty:
            raise ValueError("提供的資料集中沒有 'CONFIRMED' 或 'FALSE POSITIVE' 標籤的資料。")

        training_df['disposition'] = training_df['disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})
        
        if 'source_telescope' in training_df.columns:
            training_df = pd.get_dummies(training_df, columns=['source_telescope'], drop_first=True)
        
        X = training_df.drop(columns=['disposition', 'source_id'])
        y = training_df['disposition']
        feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        print("✅ 資料載入與預處理完成。")

        # --- 2. Model Configuration ---
        MODELS_CONFIG = {
            "LogisticRegression": {"model": LogisticRegression, "params": {"random_state": 42, "class_weight": "balanced", "max_iter": 5000}},
            "RandomForest": {"model": RandomForestClassifier, "params": {"random_state": 42, "class_weight": "balanced"}},
            "GradientBoosting": {"model": GradientBoostingClassifier, "params": {"random_state": 42}},
            "LightGBM": {"model": LGBMClassifier, "params": {"random_state": 42, "verbosity": -1, "class_weight": "balanced"}}
        }

        if model_choice not in MODELS_CONFIG:
            raise ValueError(f"選擇的模型 '{model_choice}' 不被支援。")

        model_config = MODELS_CONFIG[model_choice]
        model_config["params"].update(custom_params)
        model = model_config["model"](**model_config["params"])
        print(f"\n✅ 模型 '{model_choice}' 實例化成功，使用最終參數: {model.get_params()}")

        # --- 3. Training ---
        print("\n🚀 開始訓練模型...")
        model.fit(X_train_scaled, y_train)
        print("✅ 模型訓練完成。")
        
        # --- 4. Evaluation and Reporting ---
        print("\n📋 產生模型在測試集上的表現報告...")
        target_names = ['FALSE POSITIVE', 'CONFIRMED']
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)
        
        # ### 修改開始 ###：將混淆矩陣打包成字典以供回傳
        confusion_matrix_data = {
            "matrix": cm.tolist(),  # 將 numpy array 轉為 python list
            "labels": target_names
        }
        print("✅ 混淆矩陣資料已產生。")
        # ### 修改結束 ###

        print("\n--- 分類報告 ---")
        print(report)
        print("\n--- 混淆矩陣 ---")
        print(f"{'':<15} | {'預測為非行星':<15} | {'預測為行星':<15}")
        print("-" * 50)
        print(f"{'實際為非行星':<15} | {cm[0][0]:<15} | {cm[0][1]:<15}")
        print(f"{'實際為行星':<15} | {cm[1][0]:<15} | {cm[1][1]:<15}")
        
        # --- 5. Saving Artifacts (for potential debug) and preparing for return ---
        model_dir = "temp_models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_name_key = model_choice.lower()
        output_model_path = os.path.join(model_dir, f"{model_name_key}_model.joblib")
        joblib.dump(model, output_model_path)
        print(f"\n💾 (Debug) 模型副本已儲存至: {output_model_path}")
        
        trained_artifacts["model"] = model
        trained_artifacts["imputer"] = imputer
        trained_artifacts["scaler"] = scaler
        trained_artifacts["feature_names"] = feature_names
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ 訓練流程發生錯誤: {e}")
        traceback.print_exc(file=sys.stdout)
        print("="*50)

    finally:
        sys.stdout = old_stdout

    # ### 修改開始 ###：回傳 4 個項目，包含打包好的混淆矩陣
    return output_model_path, captured_output.getvalue(), trained_artifacts, confusion_matrix_data
    # ### 修改結束 ###