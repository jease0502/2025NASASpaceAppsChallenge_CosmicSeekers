# TSFresh + LightGBM 系外行星偵測系統

## 概述

`tsfresh_LightGBM.py` 是項目的核心機器學習組件，實現了基於 TSFresh 特徵萃取和 LightGBM 分類器的系外行星偵測系統。該系統重現了 Malik et al. (2022, MNRAS 513:5505) 的研究方法，並提供了完整的端到端機器學習管道。

## 🎯 核心功能

### 1. 光變曲線處理
- **多源支持**: 同時處理克普勒 (KOI) 和 TESS 數據
- **智能下載**: 自動下載和快取光變曲線數據
- **數據清理**: 移除 NaN 值、平坦化、正規化
- **格式轉換**: 將 LightKurve 對象轉換為 TSFresh 格式

### 2. 特徵萃取
- **TSFresh 集成**: 自動萃取數百個時間序列特徵
- **特徵選擇**: 支持 Efficient 和 Comprehensive 兩種模式
- **數據品質控制**: 自動過濾低品質樣本
- **特徵清理**: 移除無效和重複特徵

### 3. 機器學習
- **LightGBM 分類器**: 高效的梯度提升樹模型
- **智能參數調整**: 根據樣本量自動調整模型參數
- **類別平衡**: 處理不平衡數據集
- **交叉驗證**: 自動分割訓練和測試集

### 4. 性能評估
- **多維度指標**: AUC、Precision、Recall、F1-Score
- **詳細報告**: 完整的分類報告和混淆矩陣
- **結果保存**: 模型和評估結果的持久化

## 📋 系統要求

### 依賴套件
```bash
pip install lightkurve tsfresh lightgbm scikit-learn pandas numpy joblib tqdm astropy
```

### 硬體要求
- **記憶體**: 建議 8GB+ RAM
- **儲存**: 光變曲線數據需要大量空間
- **網路**: 需要穩定的網路連接下載數據

## 🚀 使用方法

### 基本命令
```bash
python tsfresh_LightGBM.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --out_dir ./out_malik2022 \
    --max_targets 100 \
    --lc_dir ./lightkurve_data \
    --comprehensive
```

### 參數說明

| 參數 | 類型 | 必需 | 說明 |
|------|------|------|------|
| `--processed_csv` | str | ✓ | 處理過的數據 CSV 文件路徑 |
| `--out_dir` | str | ✓ | 輸出目錄路徑 |
| `--max_targets` | int | ✗ | 最大處理樣本數 (預設: 200) |
| `--lc_dir` | str | ✗ | 本地光變曲線目錄 |
| `--features_dir` | str | ✗ | 預處理特徵目錄 |
| `--comprehensive` | flag | ✗ | 使用完整特徵集 (~789個特徵) |
| `--offline` | flag | ✗ | 離線模式，不連網下載 |

### 使用範例

#### 1. 完整流程（線上模式）
```bash
python tsfresh_LightGBM.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --out_dir ./results \
    --max_targets 500 \
    --comprehensive
```

#### 2. 離線模式（使用本地數據）
```bash
python tsfresh_LightGBM.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --out_dir ./results \
    --lc_dir ./lightkurve_data \
    --offline
```

#### 3. 使用預處理特徵
```bash
python tsfresh_LightGBM.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --out_dir ./results \
    --features_dir ./tsfresh_features_individual
```

## 🔧 核心函數說明

### 數據處理函數

#### `load_processed_data(csv_path: Path) -> pd.DataFrame`
**功能**: 讀取處理過的系外行星數據
**參數**:
- `csv_path`: CSV 文件路徑
**返回**: 包含所有處理後數據的 DataFrame

#### `filter_labeled_data(df: pd.DataFrame) -> pd.DataFrame`
**功能**: 過濾並標記二元分類數據
**參數**:
- `df`: 原始數據框
**返回**: 只包含 CONFIRMED 和 FALSE POSITIVE 的標記數據
**標籤映射**:
- `CONFIRMED` → 1 (系外行星)
- `FALSE POSITIVE` → 0 (誤報)

#### `pick_targets(df: pd.DataFrame, max_targets: int) -> pd.DataFrame`
**功能**: 隨機選擇指定數量的目標進行處理
**參數**:
- `df`: 標記數據框
- `max_targets`: 最大目標數
**返回**: 隨機選擇的子集

### 光變曲線處理函數

#### `download_kepler_lc_by_kepid(kepid: int) -> lk.LightCurve`
**功能**: 下載克普勒光變曲線
**參數**:
- `kepid`: 克普勒星體 ID
**返回**: 處理後的 LightCurve 對象
**特點**:
- 優先使用 PDCSAP_FLUX（去除系統性趨勢）
- 自動回退到 SAP_FLUX
- 自動清理和正規化

#### `download_tess_lc_by_tic(tic: int) -> lk.LightCurve`
**功能**: 下載 TESS 光變曲線
**參數**:
- `tic`: TESS 星體 ID
**返回**: 處理後的 LightCurve 對象

#### `lc_to_timeseries_df(lc: lk.LightCurve, sample_id: int) -> pd.DataFrame`
**功能**: 將 LightCurve 轉換為 TSFresh 格式
**參數**:
- `lc`: LightKurve 光變曲線對象
- `sample_id`: 樣本 ID
**返回**: 包含 (id, time, flux) 的 DataFrame

### 特徵萃取函數

#### `extract_tsfresh_features(ts_long: pd.DataFrame, comprehensive: bool) -> pd.DataFrame`
**功能**: 使用 TSFresh 萃取時間序列特徵
**參數**:
- `ts_long`: 長格式時間序列數據
- `comprehensive`: 是否使用完整特徵集
**返回**: 特徵矩陣 DataFrame
**特徵類型**:
- **統計特徵**: 均值、標準差、偏度、峰度等
- **頻域特徵**: FFT 係數、功率譜密度等
- **時域特徵**: 自相關、趨勢分析等
- **非線性特徵**: 熵、複雜度指標等

### 機器學習函數

#### `train_and_eval(X: pd.DataFrame, y: np.ndarray, out_dir: Path) -> Dict`
**功能**: 訓練 LightGBM 模型並評估性能
**參數**:
- `X`: 特徵矩陣
- `y`: 標籤數組
- `out_dir`: 輸出目錄
**返回**: 包含評估指標的字典
**評估指標**:
- AUC (Area Under Curve)
- Precision, Recall, F1-Score
- 分類報告

## 📊 特徵萃取詳解

### TSFresh 特徵類型

#### 1. 統計特徵
```python
# 基本統計量
'mean', 'std', 'var', 'skewness', 'kurtosis'
'median', 'min', 'max', 'range'
```

#### 2. 頻域特徵
```python
# FFT 相關特徵
'fft_coefficient', 'power_spectral_density'
'welch_density', 'periodogram'
```

#### 3. 時域特徵
```python
# 自相關和趨勢
'autocorrelation', 'partial_autocorrelation'
'trend', 'linear_trend'
```

#### 4. 非線性特徵
```python
# 複雜度和熵
'approximate_entropy', 'sample_entropy'
'permutation_entropy', 'complexity'
```

### 特徵選擇策略

#### Efficient 模式 (~776 個特徵)
- 快速計算
- 適合大規模數據
- 包含最重要的特徵

#### Comprehensive 模式 (~789 個特徵)
- 完整特徵集
- 論文原始設定
- 最高準確度

## 🤖 模型配置

### LightGBM 參數

#### 小樣本模式 (< 100 樣本)
```python
{
    'n_estimators': min(100, total_samples // 2),
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.8
}
```

#### 標準模式 (≥ 100 樣本)
```python
{
    'n_estimators': 600,
    'learning_rate': 0.05,
    'max_depth': -1,
    'subsample': 0.9,
    'colsample_bytree': 0.8
}
```

### 數據預處理管道
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("lgbm", LGBMClassifier(...))
])
```

## 📈 性能優化

### 記憶體優化
- **分批處理**: 大量數據分批載入
- **特徵選擇**: 移除無效特徵
- **數據清理**: 提前過濾低品質數據

### 計算優化
- **並行處理**: TSFresh 多核萃取
- **快取機制**: 避免重複下載
- **智能參數**: 根據數據量調整模型

### 網路優化
- **離線模式**: 使用本地數據
- **重試機制**: 處理網路錯誤
- **快取策略**: 本地保存下載數據

## 🐛 錯誤處理

### 常見錯誤及解決方案

#### 1. 光變曲線下載失敗
```python
# 錯誤: FileNotFoundError: No Kepler LCF for KIC xxx
# 解決: 檢查星體 ID 是否正確，或使用離線模式
```

#### 2. 特徵萃取失敗
```python
# 錯誤: ValueError: 時間序列資料為空
# 解決: 檢查光變曲線數據品質，調整 min_points 參數
```

#### 3. 記憶體不足
```python
# 錯誤: MemoryError
# 解決: 減少 max_targets 參數，或增加系統記憶體
```

### 調試技巧

#### 1. 啟用詳細輸出
```python
# 在函數中添加 print 語句
print(f"處理樣本數: {len(df)}")
print(f"特徵維度: {X.shape}")
```

#### 2. 檢查數據品質
```python
# 檢查每個樣本的數據點數
sample_counts = ts_long.groupby('id').size()
print(f"樣本數據點數: {sample_counts.describe()}")
```

#### 3. 逐步驗證
```python
# 先處理少量樣本
python tsfresh_LightGBM.py --max_targets 10
```

## 📁 輸出文件

### 主要輸出
- **`features.csv`**: 特徵矩陣
- **`labels.csv`**: 標籤數據
- **`model_lgbm.joblib`**: 訓練好的模型
- **`train_report.json`**: 訓練報告

### 訓練報告格式
```json
{
    "auc": 0.85,
    "report": {
        "0": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85},
        "1": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85}
    },
    "n_train": 150,
    "n_test": 50,
    "n_features": 789,
    "total_samples": 200
}
```

## 🔬 科學背景

### 研究基礎
- **Malik et al. (2022)**: "Automated classification of transiting exoplanet candidates using machine learning"
- **期刊**: Monthly Notices of the Royal Astronomical Society
- **卷期**: 513, 5505-5520

### 方法論
1. **數據來源**: 克普勒和 TESS 光變曲線
2. **特徵萃取**: TSFresh 自動特徵工程
3. **分類器**: LightGBM 梯度提升樹
4. **評估**: 二元分類（系外行星 vs 誤報）

### 創新點
- **多源數據融合**: 統一處理不同望遠鏡數據
- **自動特徵工程**: 減少人工特徵設計
- **可重現性**: 完整的實驗記錄和代碼

## 🚀 進階使用

### 自定義特徵萃取
```python
# 修改特徵參數
from tsfresh.feature_extraction import ComprehensiveFCParameters

# 自定義特徵集
custom_params = {
    'mean': None,
    'std': None,
    'autocorrelation': [{'lag': 1}, {'lag': 2}],
    'fft_coefficient': [{'coeff': 0}, {'coeff': 1}]
}
```

### 模型調優
```python
# 網格搜索最佳參數
from sklearn.model_selection import GridSearchCV

param_grid = {
    'lgbm__n_estimators': [100, 300, 600],
    'lgbm__learning_rate': [0.01, 0.05, 0.1],
    'lgbm__max_depth': [3, 5, -1]
}
```

### 批量實驗
```bash
# 不同參數組合的批量實驗
for targets in 50 100 200 500; do
    for comprehensive in "" "--comprehensive"; do
        python tsfresh_LightGBM.py \
            --max_targets $targets \
            $comprehensive \
            --out_dir ./results_${targets}${comprehensive}
    done
done
```

## 📚 參考文獻

1. Malik, M., et al. (2022). "Automated classification of transiting exoplanet candidates using machine learning." MNRAS, 513, 5505-5520.

2. Borucki, W. J., et al. (2010). "Kepler planet-detection mission: introduction and first results." Science, 327, 977-980.

3. Ricker, G. R., et al. (2015). "Transiting Exoplanet Survey Satellite (TESS)." Journal of Astronomical Telescopes, Instruments, and Systems, 1, 014003.

4. Christ, M., et al. (2018). "Time series feature extraction on basis of scalable hypothesis tests (tsfresh - A Python package)." Neurocomputing, 307, 72-77.

---

**注意**: 本系統需要大量計算資源。建議在具有充足記憶體和儲存空間的機器上運行，並確保網路連接穩定。
