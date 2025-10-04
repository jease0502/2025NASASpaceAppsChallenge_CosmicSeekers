import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
import os
import json

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier

# --- Environment Setup ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_style("whitegrid")
# Set font properties for plots if needed, though plots are disabled in this version.
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False

# --- Configuration ---
class Config:
    """
    Centralized management of all hyperparameters and settings.
    """
    # --- 1. File and Path Settings ---
    CSV_FILE_PATH = "exoplanet_data_processed_augment.csv"
    MODEL_DIR = "model"

    # --- 2. Data Splitting Settings ---
    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    # --- 3. Model Hyperparameter Settings ---
    MODELS = {
        "LogisticRegression": {
            "model": LogisticRegression,
            "params": {
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
                "solver": "saga",
                "penalty": "l1",
                "C": 0.5,
                "max_iter": 5000
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier,
            "params": {
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
                "n_estimators": 1000,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt"
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier,
            "params": {
                "random_state": RANDOM_STATE,
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.8,
                "max_features": "sqrt"
            }
        },
        "LightGBM": {
            "model": LGBMClassifier,
            "params": {
                "random_state": RANDOM_STATE,
                "verbosity": -1,
                "class_weight": "balanced",
                "n_estimators": 1500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "colsample_bytree": 0.8,
                "subsample": 0.8
            }
        }
    }
    # Select the best model name for SHAP analysis and candidate prediction
    BEST_MODEL_NAME = "LightGBM"

    # --- 4. Cross-Validation Settings ---
    CV_FOLDS = 10

    # --- 5. Model Interpretation and Reporting Settings ---
    N_FEATURES_TO_DISPLAY = 20
    SHAP_SAMPLE_SIZE = 100
    N_TOP_CANDIDATES = 20


class ExoplanetDataProcessor:
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
            print(f"✅ Successfully loaded data from '{self.file_path}'.")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File not found at '{self.file_path}'. Please run the preprocessing script first.")
            return False

    def prepare_data(self):
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        candidate_df = self.df[self.df['disposition'] == 'CANDIDATE'].copy()
        self.candidate_info = candidate_df[['source_id', 'source_telescope']].copy()
        
        training_df = self.df[self.df['disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

        if training_df.empty:
            print("❌ Error: No data labeled 'CONFIRMED' or 'FALSE POSITIVE' found for training.")
            return False

        training_df['disposition'] = training_df['disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})
        
        training_df = pd.get_dummies(training_df, columns=['source_telescope'], drop_first=True)
        candidate_df = pd.get_dummies(candidate_df, columns=['source_telescope'], drop_first=True)
        
        X = training_df.drop(columns=['disposition', 'source_id'])
        y = training_df['disposition']
        self.feature_names = X.columns.tolist()

        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )

        self.imputer = SimpleImputer(strategy='median')
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        self.X_test_scaled = self.scaler.transform(X_test_imputed)

        if not candidate_df.empty:
            candidate_features = candidate_df.drop(columns=['disposition', 'source_id'], errors='ignore')
            candidate_features = candidate_features.reindex(columns=self.feature_names, fill_value=0)
            candidate_imputed = self.imputer.transform(candidate_features)
            self.candidate_df_processed = self.scaler.transform(candidate_imputed)

        print("✅ Data preprocessing and splitting completed.")
        return True

    def run(self):
        if self.load_data():
            return self.prepare_data()
        return False

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}
        for name, config in Config.MODELS.items():
            self.models[name] = config["model"](**config["params"])
            
        specificity_scorer = make_scorer(recall_score, pos_label=0)
        self.scoring = {
            'accuracy': 'accuracy', 'precision': 'precision',
            'sensitivity_recall': 'recall', 'f1_score': 'f1',
            'specificity': specificity_scorer
        }

    def cross_validate_models(self):
        results = {}
        print("\n🚀 Starting model 10-fold cross-validation...")
        for name, model in self.models.items():
            print(f"   - Cross-validating: {name}...")
            cv_results = cross_validate(
                model, self.X_train, self.y_train, cv=Config.CV_FOLDS,
                scoring=self.scoring, n_jobs=-1
            )
            results[name] = {
                'Accuracy': cv_results['test_accuracy'].mean(),
                'Precision': cv_results['test_precision'].mean(),
                'Sensitivity (Recall)': cv_results['test_sensitivity_recall'].mean(),
                'F1-Score': cv_results['test_f1_score'].mean(),
                'Specificity': cv_results['test_specificity'].mean()
            }
        print("✅ All models cross-validated.")
        return pd.DataFrame(results).T

    def train_and_save_all_models(self, path=Config.MODEL_DIR):
        print(f"\n💾 Training and saving all models to '{path}' directory...")
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            print(f"   - Training: {name}...")
            model.fit(self.X_train, self.y_train)
            filename = os.path.join(path, f"{name.lower()}_model.joblib")
            joblib.dump(model, filename)
            print(f"     - ✅ Model saved to '{filename}'")

class ModelInterpreter:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def generate_and_save_feature_importance(self, X_train, y_train, model_path, n_features):
        model_name = self.model.__class__.__name__
        print(f"   - Generating feature importance for '{model_name}'...")
        self.model.fit(X_train, y_train)

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            total_importance = np.sum(importances)
            
            normalized_importances = (importances / total_importance) * 100 if total_importance > 0 else importances

            importance_dict = dict(zip(self.feature_names, normalized_importances))
            sorted_importances = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
            
            top_n_importances = {feature: float(score) for feature, score in sorted_importances[:n_features]}

            json_filename = os.path.join(model_path, f"{model_name.lower()}_feature_importance.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(top_n_importances, f, ensure_ascii=False, indent=4)
            print(f"     - ✅ Top {n_features} feature importances saved to '{json_filename}'")
            
        else:
            print(f"   - ℹ️ Model {model_name} does not support 'feature_importances_' attribute.")

    def explain_with_shap(self, X_train, y_train, X_test_sample, n_features):
        model_name = self.model.__class__.__name__
        print(f"\n🔮 Generating SHAP explanations for '{model_name}'...")
        self.model.fit(X_train, y_train)
        
        if hasattr(self.model, 'feature_importances_'):
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test_sample)
        else:
            explainer = shap.LinearExplainer(self.model, X_train)
            shap_values = explainer.shap_values(X_test_sample)

        print("   - SHAP summary plot generation is disabled.")
        # The following lines are intentionally commented out to prevent plot display.
        # shap.summary_plot(
        #     shap_values, X_test_sample, feature_names=self.feature_names,
        #     max_display=n_features, show=False
        # )
        # plt.title(f'SHAP Feature Impact Analysis ({model_name})', fontsize=16)
        # plt.tight_layout()
        # plt.show()
        print(f"   - SHAP value calculation for '{model_name}' is complete.")
        
def generate_model_stats_json(model_path, X_test, y_test):
    print("\n" + "="*70)
    print("📋 Generating model performance reports (JSON) on the test set...")
    print("="*70)
    target_names = ['FALSE POSITIVE', 'CONFIRMED']
    
    for model_file in os.listdir(model_path):
        if model_file.endswith(".joblib") and '_model' in model_file:
            model_name_key = model_file.replace('_model.joblib', '')
            model_name_title = model_name_key.replace('_', ' ').title()
            print(f"\n--- Processing model: {model_name_title} ---")
            
            model = joblib.load(os.path.join(model_path, model_file))
            y_pred = model.predict(X_test)
            
            report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            model_stats = {
                "model_name": model_name_title,
                "classification_report": report,
                "confusion_matrix": {"labels": target_names, "matrix": cm.tolist()}
            }
            
            json_filename = os.path.join(model_path, f"{model_name_key}_stats.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(model_stats, f, ensure_ascii=False, indent=4)
            print(f"   - ✅ Performance report saved to '{json_filename}'")

# --- Main Execution Flow ---
if __name__ == "__main__":
    processor = ExoplanetDataProcessor(file_path=Config.CSV_FILE_PATH)
    
    if processor.run():
        trainer = ModelTrainer(processor.X_train_scaled, processor.y_train)
        
        results_df = trainer.cross_validate_models()
        print("\n" + "="*70)
        print(f"📊 Cross-Validation Performance Results ({Config.CV_FOLDS}-fold Average)")
        print("="*70)
        print(results_df.to_string(float_format="%.4f"))
        print("="*70)

        trainer.train_and_save_all_models()

        generate_model_stats_json(
            model_path=Config.MODEL_DIR,
            X_test=processor.X_test_scaled,
            y_test=processor.y_test
        )
        
        print("\n" + "="*70)
        print("📈 Generating and saving feature importance for all models...")
        print("="*70)
        for model_name, model_instance in trainer.models.items():
            interpreter = ModelInterpreter(
                model=model_instance,
                feature_names=processor.feature_names
            )
            interpreter.generate_and_save_feature_importance(
                processor.X_train_scaled,
                processor.y_train,
                model_path=Config.MODEL_DIR,
                n_features=Config.N_FEATURES_TO_DISPLAY
            )
        
        best_model_config = Config.MODELS[Config.BEST_MODEL_NAME]
        best_model_instance = best_model_config["model"](**best_model_config["params"])
        
        best_model_interpreter = ModelInterpreter(
            model=best_model_instance,
            feature_names=processor.feature_names
        )

        X_test_sample = processor.X_test_scaled[:Config.SHAP_SAMPLE_SIZE]
        best_model_interpreter.explain_with_shap(
            processor.X_train_scaled, processor.y_train, X_test_sample,
            n_features=Config.N_FEATURES_TO_DISPLAY
        )
        
        print("\n" + "="*70)
        print("🚀 Predicting 'CANDIDATE' exoplanets...")
        print("="*70)
        
        final_model_filename = f"{Config.BEST_MODEL_NAME.lower()}_model.joblib"
        final_model = joblib.load(os.path.join(Config.MODEL_DIR, final_model_filename))
        
        if processor.candidate_df_processed is not None and len(processor.candidate_df_processed) > 0:
            candidate_predictions = final_model.predict_proba(processor.candidate_df_processed)[:, 1]
            
            prediction_results = processor.candidate_info.copy()
            prediction_results['Prediction_Score'] = candidate_predictions
            
            top_candidates = prediction_results.sort_values(by='Prediction_Score', ascending=False)
            
            print("   ✅ Prediction complete.")
            print("\n" + "="*70)
            print(f"🎯 Top {Config.N_TOP_CANDIDATES} CANDIDATEs most likely to be true planets")
            print("   (Higher score indicates higher probability)")
            print("="*70)
            print(top_candidates.head(Config.N_TOP_CANDIDATES).to_string(index=False))
            print("="*70)
        else:
            print("   - ℹ️ No data labeled 'CANDIDATE' found for prediction.")

        print("\n" + "="*70)
        print(f"💾 Saving preprocessing tools to '{Config.MODEL_DIR}' directory...")
        print("="*70)
        joblib.dump(processor.scaler, os.path.join(Config.MODEL_DIR, 'exoplanet_scaler.joblib'))
        joblib.dump(processor.imputer, os.path.join(Config.MODEL_DIR, 'exoplanet_imputer.joblib'))
        joblib.dump(processor.feature_names, os.path.join(Config.MODEL_DIR, 'exoplanet_feature_names.joblib'))
        
        print(f"   - ✅ Scaler, Imputer, and Feature Names saved successfully.")
        print("="*70)