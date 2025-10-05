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
    # ### 修改開始 ###：初始化混淆矩陣與分類報告變數
    confusion_matrix_data = None
    classification_report_data = None
    accuracy_value = None
    # ### 修改結束 ###

    try:
        print(f"--- 開始線上訓練流程：{model_choice} ---")
        print(f"使用資料集: {input_csv_path}")
        print(f"自訂超參數: {json.dumps(custom_params, indent=2)}")

        # --- 1. Data Loading and Preparation ---
        df = pd.read_csv(input_csv_path, dtype={'source_id': str})
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 使用所有標籤進行三元分類
        disposition_map = {
            'CONFIRMED': 2,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0
        }
        training_df = df[df['disposition'].isin(disposition_map.keys())].copy()
        if training_df.empty:
            raise ValueError("提供的資料集中沒有可用的標籤資料。")

        # 顯示類別分布
        print("\n類別分布：")
        print(training_df['disposition'].value_counts())
        
        training_df['disposition'] = training_df['disposition'].map(disposition_map)
        
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
            "LogisticRegression": {
                "model": LogisticRegression,
                "params": {
                    "random_state": 42,
                    "class_weight": "balanced",
                    "max_iter": 5000,
                    "multi_class": "multinomial",  # 多分類設置
                    "solver": "lbfgs"
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier,
                "params": {
                    "random_state": 42,
                    "class_weight": "balanced",
                    "n_estimators": 1000,
                    "max_depth": 15
                }
            },
            "GradientBoosting": {
                "model": GradientBoostingClassifier,
                "params": {
                    "random_state": 42,
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "max_depth": 7
                }
            },
            "LightGBM": {
                "model": LGBMClassifier,
                "params": {
                    "random_state": 42,
                    "verbosity": -1,
                    "class_weight": "balanced",
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "num_leaves": 50,
                    "max_depth": 7,
                    "objective": "multiclass",  # 多分類設置
                    "num_class": 3  # 三個類別
                }
            }
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
        target_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # 將混淆矩陣打包成字典以供回傳
        confusion_matrix_data = {
            "matrix": cm.tolist(),  # 將 numpy array 轉為 python list
            "labels": target_names
        }
        print("✅ 混淆矩陣資料已產生。")

        # 打包分類報告與整體準確度供回傳
        classification_report_data = report
        accuracy_value = float(report.get('accuracy', 0.0))

        # 計算每個類別的性能指標
        print("\n=== 分類報告 ===")
        for class_name in target_names:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  精確度: {metrics['precision']:.3f}")
            print(f"  召回率: {metrics['recall']:.3f}")
            print(f"  F1分數: {metrics['f1-score']:.3f}")
        
        print(f"\n整體準確度: {report['accuracy']:.3f}")
        print(f"Macro avg F1: {report['macro avg']['f1-score']:.3f}")
        print(f"Weighted avg F1: {report['weighted avg']['f1-score']:.3f}")

        print("\n=== 混淆矩陣 ===")
        print("預測類別:")
        print(f"{'實際類別':<15} | {'FALSE POSITIVE':<15} | {'CANDIDATE':<15} | {'CONFIRMED':<15}")
        print("-" * 65)
        for i, class_name in enumerate(target_names):
            row = [f"{class_name:<15}"]
            row.extend([f"{cm[i][j]:<15}" for j in range(3)])
            print(" | ".join(row))
        
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

    # ### 修改開始 ###：回傳 6 個項目，包含打包好的混淆矩陣與分類報告、準確度
    return output_model_path, captured_output.getvalue(), trained_artifacts, confusion_matrix_data, classification_report_data, accuracy_value
    # ### 修改結束 ###