import mimetypes
mimetypes.add_type('image/webp', '.webp')

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import random
import os
import numpy as np
import joblib
import json
import traceback
from werkzeug.utils import secure_filename
# shutil 已不再需要
# import shutil 

from data_preprocessing import run_pipeline_on_single_file, identify_and_validate_dataset
# ### 修改開始 ###：從 Train.py 導入的函式名稱維持不變
from Train import run_training_pipeline
# ### 修改結束 ###

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'tmp/uploads'
MODELS_FOLDER = 'models' 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 穩定模型 ---
MODELS = {}
MODEL_STATS = {}
PREPROCESSORS = {}
# --- 使用者線上訓練的臨時模型 (存放在記憶體中) ---
LATEST_TRAINED_ARTIFACTS = {} # ### 修改開始 ###：新的全域變數

def load_ml_artifacts():
    model_names = ["randomforest", "gradientboosting", "lightgbm", "logisticregression"]
    print("--- 正在載入機器學習元件 ---")
    try:
        PREPROCESSORS['imputer'] = joblib.load(os.path.join(MODELS_FOLDER, 'exoplanet_imputer.joblib'))
        PREPROCESSORS['scaler'] = joblib.load(os.path.join(MODELS_FOLDER, 'exoplanet_scaler.joblib'))
        PREPROCESSORS['features'] = joblib.load(os.path.join(MODELS_FOLDER, 'exoplanet_feature_names.joblib'))
        print(f"✅ 預處理工具與特徵列表({len(PREPROCESSORS['features'])}個)載入成功。")
    except Exception as e:
        print(f"⚠️ 警告：載入預處理元件或特徵檔案時出錯: {e}。")

    for name in model_names:
        model_path = os.path.join(MODELS_FOLDER, f"{name}_model.joblib")
        stats_path = os.path.join(MODELS_FOLDER, f"{name}_stats.json")
        key_name = name.replace("randomforest", "random_forest").replace("gradientboosting", "gradient_boosting").replace("logisticregression", "logistic_regression")
        if os.path.exists(model_path) and os.path.exists(stats_path):
            try:
                MODELS[key_name] = joblib.load(model_path)
                with open(stats_path, 'r') as f:
                    MODEL_STATS[key_name] = json.load(f)
                print(f"✅ 模型與狀態 '{key_name}' 載入成功。")
            except Exception as e:
                print(f"❌ 載入 '{name}' 時發生錯誤: {e}")
        else:
            print(f"⚠️ 警告：找不到模型 '{model_path}' 或狀態 '{stats_path}'。")
    print("-" * 28)

# ... load_and_process_exoplanet_data() 函式維持不變 ...
def load_and_process_exoplanet_data():
    file_path = 'exoplanet_data_processed.csv'
    if not os.path.exists(file_path): 
        print(f"錯誤: 主要資料檔案 '{file_path}' 不存在。")
        return []
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df.replace({np.nan: None}, inplace=True)
        df_clean = df.dropna(subset=['period', 'source_id'])
        df_clean = df_clean[df_clean['disposition'].isin(['CONFIRMED', 'CANDIDATE', 'CONFIRMED_TESS'])]
        def get_host_id(row):
            source_id_str = str(row['source_id'])
            if '.' in source_id_str and not source_id_str.startswith('TIC'): return source_id_str.split('.')[0]
            return source_id_str
        df_clean['host_star_id'] = df_clean.apply(get_host_id, axis=1)
        systems = {}
        for host_id, group in df_clean.groupby('host_star_id'):
            system_id, first_row = host_id, group.iloc[0]
            telescope = first_row.get('source_telescope', 'Unknown')
            system_name = f"{telescope} {system_id}"
            if telescope.lower() == 'kepler' and system_id.isdigit(): system_name = f"KIC {system_id}"
            star_data = {key: first_row.get(key) for key in ['stellar_radius', 'stellar_temp', 'stellar_logg', 'ra', 'dec', 'stellar_magnitude']}
            star_data.update({"system_id": system_id, "system_name": system_name, "source_id": system_id, "source_telescope": telescope})
            systems[system_id] = { "star": star_data, "planets": [] }
            for i, (_, row) in enumerate(group.iterrows()):
                planet_data = row.to_dict()
                planet_data['planet_index'] = i + 1
                systems[system_id]['planets'].append(planet_data)
        print(f"✅ 成功載入並處理 {len(df_clean)} 顆行星，分屬於 {len(systems)} 個星系。")
        return list(systems.values())
    except Exception as e:
        print(f"!!! 處理 {file_path} 時發生錯誤: {e}")
        return []

load_ml_artifacts()
ALL_EXOPLANET_DATA = load_and_process_exoplanet_data()

@app.route('/upload_and_process', methods=['POST'])
def upload_and_process_route():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file found in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
        
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() == 'csv':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            dataset_type, df, column_map = identify_and_validate_dataset(filepath)
            output_file_path, log_output = run_pipeline_on_single_file(dataset_type, df, column_map)
            os.remove(filepath)
            return jsonify({
                'success': True,
                'message': f'The uploaded {dataset_type} dataset has been successfully processed.',
                'file_url': f'/{output_file_path}',
                'log_output': log_output
            })
        except ValueError as e:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'File validation failed', 'details': str(e)}), 400
        except Exception as e:
            os.remove(filepath)
            traceback.print_exc()
            return jsonify({'success': False, 'error': 'An unexpected error occurred while processing the file.', 'details': str(e)}), 500
    return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'}), 400


@app.route('/run_training', methods=['POST'])
def run_training_route():
    data = request.get_json()
    model_choice = data.get('model_choice')
    hyperparameters = data.get('hyperparameters', {})
    input_csv_path = data.get('input_csv_path')

    if not all([model_choice, input_csv_path]):
        return jsonify({'success': False, 'error': '缺少模型選擇或資料路徑。'}), 400
    
    sanitized_path = input_csv_path.lstrip('/')
    if not os.path.exists(sanitized_path):
        return jsonify({'success': False, 'error': f'伺服器上找不到資料檔案: {sanitized_path}'}), 404

    try:
        for key, value in hyperparameters.items():
            try:
                float_val = float(value)
                if float_val == int(float_val): hyperparameters[key] = int(float_val)
                else: hyperparameters[key] = float_val
            except (ValueError, TypeError): pass

        # ### 修改開始 ###: 接收 Train.py 回傳的物件，包含混淆矩陣
        output_path, log_output, trained_artifacts, confusion_matrix_data = run_training_pipeline(model_choice, hyperparameters, sanitized_path)

        if trained_artifacts and trained_artifacts.get("model"):
            # 將訓練好的物件存到記憶體中的 LATEST_TRAINED_ARTIFACTS
            model_key = model_choice.replace("RandomForest", "random_forest").replace("GradientBoosting", "gradient_boosting").replace("LightGBM", "lightgbm").replace("LogisticRegression", "logistic_regression")
            LATEST_TRAINED_ARTIFACTS[model_key] = trained_artifacts
            print(f"✅ 臨時模型 '{model_key}' 已成功訓練並載入記憶體。")
        else:
             print(f"⚠️ 警告：訓練完成但未收到有效的模型物件。")
        # ### 修改結束 ###

        return jsonify({
            'success': True,
            'message': f'模型 {model_choice} 訓練體驗完成！',
            'log_output': log_output,
            'confusion_matrix': confusion_matrix_data # <<< 新增：將混淆矩陣回傳給前端
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': '執行訓練流程時發生錯誤。', 'details': str(e)}), 500
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json()
        model_name = payload.get("model_name")
        planets_data = payload.get("planets")
        # ### 修改開始 ###：檢查是否要使用臨時模型
        use_latest_trained = payload.get("use_latest_trained", False)
        # ### 修改結束 ###

        if not model_name or not planets_data: return jsonify({"error": "請求無效"}), 400

        # ### 修改開始 ###：根據 use_latest_trained 旗標選擇模型和工具來源
        artifacts_source = None
        if use_latest_trained and model_name in LATEST_TRAINED_ARTIFACTS:
            print(f"🧠 使用者剛訓練的臨時模型進行預測: {model_name}")
            artifacts_source = LATEST_TRAINED_ARTIFACTS[model_name]
            model = artifacts_source.get('model')
            imputer = artifacts_source.get('imputer')
            scaler = artifacts_source.get('scaler')
            expected_features = artifacts_source.get('feature_names')
        else:
            print(f"🗄️ 使用穩定的正式模型進行預測: {model_name}")
            model = MODELS.get(model_name)
            imputer = PREPROCESSORS.get('imputer')
            scaler = PREPROCESSORS.get('scaler')
            expected_features = PREPROCESSORS.get('features')
        # ### 修改結束 ###

        if not all([model, expected_features, imputer, scaler]):
            return jsonify({"error": f"找不到 '{model_name}' 相關的機器學習元件"}), 404

        df_predict = pd.DataFrame(planets_data)
        
        if 'source_telescope' in df_predict.columns:
            df_predict = pd.get_dummies(df_predict, columns=['source_telescope'], drop_first=True)
            
        df_aligned = df_predict.reindex(columns=expected_features, fill_value=0)
        df_imputed = imputer.transform(df_aligned)
        df_scaled = scaler.transform(df_imputed)
        
        probabilities = model.predict_proba(df_scaled)
        predictions = []
        for prob in probabilities:
            confidence = prob[1]
            prediction_label = "CONFIRMED" if confidence >= 0.5 else "FALSE POSITIVE"
            predictions.append({"prediction": prediction_label, "confidence": np.float64(confidence)})
            
        return jsonify(predictions)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"預測時發生錯誤: {e}"}), 500

# ... (所有其他 API 路由維持不變) ...
@app.route('/get_triple_system_data')
def get_triple_system_data():
    if not ALL_EXOPLANET_DATA or len(ALL_EXOPLANET_DATA) < 3: return jsonify({"error": "沒有足夠的星系資料來顯示三個系統"}), 500
    three_systems = random.sample(ALL_EXOPLANET_DATA, 3)
    return jsonify(three_systems)
@app.route('/get_model_stats')
def get_model_stats():
    if not MODEL_STATS: return jsonify({"error": "沒有載入任何模型狀態"}), 500
    return jsonify(MODEL_STATS)
@app.route('/get_specific_triple_system_data/<string:system_id>')
def get_specific_triple_system_data(system_id):
    if not ALL_EXOPLANET_DATA or len(ALL_EXOPLANET_DATA) < 3: return jsonify({"error": "沒有足夠的星系資料"}), 500
    main_system = next((s for s in ALL_EXOPLANET_DATA if str(s["star"]["system_id"]) == system_id), None)
    if not main_system: return jsonify({"error": "找不到指定的星系"}), 404
    other_systems = [s for s in ALL_EXOPLANET_DATA if str(s["star"]["system_id"]) != system_id]
    random_systems = random.sample(other_systems, 2)
    final_systems = [main_system] + random_systems
    return jsonify(final_systems)
@app.route('/get_all_systems')
def get_all_systems():
    if not ALL_EXOPLANET_DATA: return jsonify({"error": "沒有載入任何星系資料"}), 500
    system_list = [{"system_id": s["star"]["system_id"], "name": s["star"]["system_name"]} for s in ALL_EXOPLANET_DATA]
    return jsonify(system_list)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/partials/<path:page_name>')
def partials(page_name):
    return render_template(f'partials/{page_name}.html')

@app.route('/demo/<path:filename>')
def download_demo(filename):
    return send_from_directory('static/demo', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)