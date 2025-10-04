import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
import os
import json
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier

# --- 環境設定 ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 組態設定 (Configuration) ---
class Config:
    """
    集中管理所有超參數和設定。
    """
    # --- 1. 檔案與路徑設定 ---
    CSV_FILE_PATH = "exoplanet_data_processed.csv"
    MODEL_DIR = "model"

    # --- 2. 資料分割設定 ---
    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    # --- 3. GridSearchCV 超參數網格設定 ---
    # ✨ MODIFICATION: 從固定參數改為搜尋網格
    # 警告：網格越大，搜尋時間越長！以下是供快速測試用的小網格。
    PARAM_GRIDS = {
        "LogisticRegression": {
            "C": [0.1, 1.0],
            "solver": ["saga"],
            "max_iter": [3000],
            "multi_class": ["multinomial"],
            "penalty": ["l2"]
        },
        "RandomForest": {
            "n_estimators": [500],
            "max_depth": [10, None],
            "min_samples_split": [5],
            "class_weight": ["balanced"]
        },
        "GradientBoosting": {
            "n_estimators": [500],
            "learning_rate": [0.1],
            "max_depth": [5],
            "subsample": [0.8]
        },
        "LightGBM": {
            "n_estimators": [1000],
            "learning_rate": [0.1],
            "num_leaves": [31],
            "max_depth": [7],
            "class_weight": ["balanced"],
            "boosting_type": ["gbdt"]
        }
    }

    # 建立模型實例 (不帶參數，參數由 GridSearch 提供)
    MODELS_TO_TUNE = {
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, class_weight='balanced')
    }
    
    # GridSearch 的設定
    CV_FOLDS = 5  # 交叉驗證折數 (5折通常是個好起點)
    # 使用 macro-averaged F1-score，適合多分類問題
    GRID_SEARCH_SCORING_METRIC = 'f1_macro'

    # --- 4. 模型解釋與報告設定 ---
    N_FEATURES_TO_DISPLAY = 20
    SHAP_SAMPLE_SIZE = 100
    N_TOP_CANDIDATES = 20


class ExoplanetDataProcessor:
    # (此類別內容與前一版相同，保持不變)
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.imputer = None
        self.candidate_df_processed = None
        self.candidate_info = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path, dtype={'source_id': str})
            print(f"✅ 成功載入資料 '{self.file_path}'。")
            return True
        except FileNotFoundError:
            print(f"❌ 錯誤：找不到檔案 '{self.file_path}'。請先執行 data_preprocessing_augment.py。")
            return False

    def prepare_data(self):
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 使用所有數據進行訓練
        training_df = self.df.copy()
        
        if training_df.empty:
            print("❌ 錯誤：沒有找到訓練資料。")
            return False
            
        # 將 disposition 轉換為數值標籤
        disposition_map = {
            'CONFIRMED': 2,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0
        }
        training_df['disposition'] = training_df['disposition'].map(disposition_map)
        
        # 保存標籤映射關係，用於後續報告
        self.label_map = {v: k for k, v in disposition_map.items()}
        
        # One-hot encoding for telescope
        training_df = pd.get_dummies(training_df, columns=['source_telescope'], drop_first=True)
        # 提取特徵和標籤
        X = training_df.drop(columns=['disposition', 'source_id'])
        y = training_df['disposition']
        self.feature_names = X.columns.tolist()

        # 分割訓練集和測試集
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )

        # 數據預處理：缺失值填充和標準化
        self.imputer = SimpleImputer(strategy='median')
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        self.X_test_scaled = self.scaler.transform(X_test_imputed)

        print("✅ 訓練資料預處理與分割完成。")
        print(f"   - 訓練集大小: {self.X_train_scaled.shape}")
        print(f"   - 測試集大小: {self.X_test_scaled.shape}")
        print(f"   - 類別分布: {pd.Series(self.y_train).value_counts().to_dict()}")
        
        return True

    def run(self):
        if self.load_data():
            return self.prepare_data()
        return False


# ✨ NEW CLASS: 專門用於執行 GridSearchCV 的類別
class HyperparameterTuner:
    def __init__(self, models, param_grids, X_train, y_train):
        self.models = models
        self.param_grids = param_grids
        self.X_train = X_train
        self.y_train = y_train
        self.results = {}

    def tune(self):
        print("\n" + "="*70)
        print("🚀 開始進行 GridSearchCV 超參數搜尋...")
        print(f"評分標準: '{Config.GRID_SEARCH_SCORING_METRIC}'")
        print("警告：此過程可能非常耗時！")
        print("="*70)
        
        for name, model in self.models.items():
            start_time = time.time()
            print(f"\n--- 正在為 '{name}' 進行網格搜尋 ---")
            param_grid = self.param_grids[name]
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=Config.CV_FOLDS,
                scoring=Config.GRID_SEARCH_SCORING_METRIC,
                n_jobs=1,   # 改用單線程以避免記憶體問題
                verbose=2,  # 顯示更詳細的進度
                pre_dispatch='2*n_jobs'  # 限制並行作業數量
            )
            grid_search.fit(self.X_train, self.y_train)
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.results[name] = {
                "best_estimator": grid_search.best_estimator_,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "duration_seconds": duration
            }
            print(f"✅ '{name}' 搜尋完成，耗時: {duration:.2f} 秒")
            print(f"   - 最佳分數 ({Config.GRID_SEARCH_SCORING_METRIC}): {grid_search.best_score_:.4f}")
            print(f"   - 最佳參數: {grid_search.best_params_}")

    def get_best_model(self):
        if not self.results:
            return None, None, -1

        best_model_name = max(self.results, key=lambda name: self.results[name]['best_score'])
        best_result = self.results[best_model_name]
        
        return best_model_name, best_result['best_estimator'], best_result['best_score']


class ModelInterpreter:
    # (此類別內容與前一版相同，保持不變)
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def generate_and_save_feature_importance(self, X_train, y_train, model_path, n_features):
        model_name = self.model.__class__.__name__
        json_filename = os.path.join(model_path, f"{model_name.lower()}_feature_importance.json")
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            total_importance = np.sum(importances)
            normalized_importances = (importances / total_importance) * 100 if total_importance > 0 else importances
            importance_dict = dict(zip(self.feature_names, normalized_importances))
            sorted_importances = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
            top_n_importances = {feature: float(score) for feature, score in sorted_importances[:n_features]}
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(top_n_importances, f, ensure_ascii=False, indent=4)
            print(f"   - ✅ 前 {n_features} 名特徵重要性已儲存至 '{json_filename}'")
            features, scores = list(top_n_importances.keys()), list(top_n_importances.values())
            plt.figure(figsize=(12, 10)); plt.title(f'前 {n_features} 名特徵重要性 ({model_name})'); plt.barh(range(len(scores)), scores[::-1], color='dodgerblue'); plt.yticks(range(len(scores)), features[::-1]); plt.xlabel('相對重要性 (%)'); plt.tight_layout(); plt.show()
        else:
            print(f"   - ℹ️ 模型 {model_name} 不支援 'feature_importances_' 屬性。")

    def explain_with_shap(self, X_train, y_train, X_test_sample, n_features):
        print(f"\n🔮 正在為 '{self.model.__class__.__name__}' 產生 SHAP 解釋圖...")
        if 'predict_proba' in dir(self.model):
            explainer = shap.KernelExplainer(self.model.predict_proba, X_test_sample)
            shap_values = explainer.shap_values(X_test_sample)
            shap.summary_plot(shap_values[1], X_test_sample, feature_names=self.feature_names, max_display=n_features, show=False)
        else:
            explainer = shap.Explainer(self.model, X_test_sample)
            shap_values = explainer(X_test_sample)
            shap.summary_plot(shap_values, X_test_sample, feature_names=self.feature_names, max_display=n_features, show=False)
        plt.title(f'SHAP 特徵影響力分析 ({self.model.__class__.__name__})', fontsize=16); plt.tight_layout(); plt.show()


# --- 主執行流程 (已重構) ---
if __name__ == "__main__":
    # 1. 資料準備
    processor = ExoplanetDataProcessor(file_path=Config.CSV_FILE_PATH)
    if not processor.run():
        print("❌ 資料處理失敗，程式終止。")
        exit()

    # 2. 執行超參數網格搜尋
    tuner = HyperparameterTuner(
        models=Config.MODELS_TO_TUNE,
        param_grids=Config.PARAM_GRIDS,
        X_train=processor.X_train_scaled,
        y_train=processor.y_train
    )
    tuner.tune()
    
    # 3. 取得最佳模型
    best_model_name, best_model, best_score = tuner.get_best_model()

    if best_model is None:
        print("❌ 未找到任何最佳模型，程式終止。")
        exit()

    print("\n" + "="*70)
    print(f"🏆 總冠軍模型: {best_model_name}")
    print(f"   - 最佳交叉驗證分數 ({Config.GRID_SEARCH_SCORING_METRIC}): {best_score:.4f}")
    print(f"   - 最佳參數: {best_model.get_params()}")
    print("="*70)

    # 4. 使用最佳模型進行後續所有分析
    # (首先，在完整的訓練集上重新訓練最佳模型)
    print("\n💾 正在完整的訓練集上重新訓練最佳模型並儲存...")
    best_model.fit(processor.X_train_scaled, processor.y_train)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(Config.MODEL_DIR, f"best_model_{best_model_name.lower()}.joblib"))
    print("   - ✅ 最佳模型已儲存。")
    
    # 5. 產生最佳模型的評估報告
    print("\n📋 正在為最佳模型生成測試集評估報告...")
    y_pred = best_model.predict(processor.X_test_scaled)
    target_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    report = classification_report(processor.y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(processor.y_test, y_pred)
    model_stats = {
        "model_name": best_model_name,
        "best_cv_score": best_score,
        "best_parameters": best_model.get_params(),
        "test_set_classification_report": report,
        "test_set_confusion_matrix": {"labels": target_names, "matrix": cm.tolist()}
    }
    json_filename = os.path.join(Config.MODEL_DIR, f"best_model_{best_model_name.lower()}_stats.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(model_stats, f, ensure_ascii=False, indent=4)
    print(f"   - ✅ 評估報告已儲存至 '{json_filename}'")
    print("\n--- 測試集表現 ---")
    print(classification_report(processor.y_test, y_pred, target_names=target_names))
    print("-" * 20)

    # 6. 進行模型解釋 (Feature Importance & SHAP)
    interpreter = ModelInterpreter(model=best_model, feature_names=processor.feature_names)
    interpreter.generate_and_save_feature_importance(
        processor.X_train_scaled, processor.y_train, Config.MODEL_DIR, Config.N_FEATURES_TO_DISPLAY
    )
    X_test_sample = processor.X_test_scaled[:Config.SHAP_SAMPLE_SIZE]
    interpreter.explain_with_shap(
        processor.X_train_scaled, processor.y_train, X_test_sample, Config.N_FEATURES_TO_DISPLAY
    )

    # 7. 顯示混淆矩陣的詳細分析
    print("\n" + "="*70)
    print(f"📊 混淆矩陣分析")
    print("="*70)
    
    cm_df = pd.DataFrame(
        cm,
        index=['實際: ' + tn for tn in target_names],
        columns=['預測: ' + tn for tn in target_names]
    )
    print("\n混淆矩陣:")
    print(cm_df)
    
    # 計算每個類別的精確度、召回率和F1分數
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print("\n各類別詳細指標:")
    metrics_df = pd.DataFrame({
        '精確度': precision,
        '召回率': recall,
        'F1分數': f1
    }, index=target_names)
    print(metrics_df.round(4))
    print("="*70)

    # 8. 儲存預處理工具
    print("\n" + "="*70)
    print(f"💾 正在保存預處理工具至 '{Config.MODEL_DIR}' 資料夾...")
    joblib.dump(processor.scaler, os.path.join(Config.MODEL_DIR, 'exoplanet_scaler.joblib'))
    joblib.dump(processor.imputer, os.path.join(Config.MODEL_DIR, 'exoplanet_imputer.joblib'))
    joblib.dump(processor.feature_names, os.path.join(Config.MODEL_DIR, 'exoplanet_feature_names.joblib'))
    print(f"   - ✅ Scaler, Imputer, 和 Feature Names 已成功保存。")
    print("="*70)