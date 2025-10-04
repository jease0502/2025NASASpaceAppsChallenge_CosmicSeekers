# 系外行星數據預處理與擴增系統

## 概述

本系統是一個專門用於處理系外行星觀測數據的預處理和擴增工具，支持克普勒 (KOI) 和 TESS 望遠鏡的數據集。系統能夠自動識別數據來源、標準化欄位、清理異常值、填補缺失值，並進行數據擴增以增強機器學習模型的訓練效果。

## 主要功能

### 1. 數據集識別與驗證
- 自動識別 KOI 和 TESS 數據集
- 驗證數據格式和欄位完整性
- 提供詳細的欄位對應報告

### 2. 數據標準化
- 統一不同來源數據的欄位命名
- 處理數據類型轉換
- 標準化行星狀態標籤

### 3. 數據清理
- 基於 IQR 方法識別和移除離群值
- 支持指定欄位的異常值清理
- 提供清理統計報告

### 4. 缺失值處理
- 使用 KNN (K-Nearest Neighbors) 算法填補缺失值
- 智能處理完全為空的欄位
- 保持數據完整性

### 5. 特徵工程
- 計算凌日訊噪比 (SNR)
- 計算行星與恆星半徑比
- 自動生成新的物理特徵

### 6. 數據擴增
- 基於測量誤差進行高斯擾動
- 支持自定義擴增倍數
- 保持數據的物理合理性

## 標準化欄位說明

| 欄位名稱 | 中文說明 | 單位/格式 |
|---------|---------|-----------|
| `source_id` | 數據源ID | KOI Name 或 TOI |
| `source_telescope` | 數據源望遠鏡 | KOI 或 TESS |
| `disposition` | 行星狀態 | CONFIRMED, CANDIDATE, FALSE POSITIVE |
| `lightcurve_id` | 光變曲線ID | kepid (KOI) 或 tid (TESS) |
| `period` | 軌道週期 | 天 |
| `duration` | 凌日持續時間 | 小時 |
| `transit_midpoint` | 凌日中點時間 | BJD (儒略日) |
| `depth` | 凌日深度 | ppm (百萬分之一) |
| `planet_radius` | 行星半徑 | 地球半徑 |
| `equilibrium_temp` | 行星平衡溫度 | Kelvin |
| `insolation_flux` | 恆星入射通量 | 以地球接收的通量為單位 |
| `stellar_magnitude` | 恆星亮度 | 星等 |
| `stellar_temp` | 恆星有效溫度 | Kelvin |
| `stellar_radius` | 恆星半徑 | 太陽半徑 |
| `stellar_logg` | 恆星表面重力 | log10(cm/s²) |
| `ra` | 赤經 | 度 |
| `dec` | 赤緯 | 度 |
| `snr` | 凌日訊噪比 | depth / avg_depth_error |
| `planet_to_star_radius_ratio` | 行星與恆星的半徑比 | 無量綱 |

## 核心函數說明

### `identify_and_validate_dataset(file_path: str)`
**功能**: 識別數據集類型並驗證欄位
**參數**:
- `file_path`: CSV 文件路徑
**返回**: `(dataset_type, df, column_map)`
**說明**: 自動檢測 KOI 或 TESS 數據集，提供欄位對應診斷報告

### `standardize_dataset(df, dataset_type, column_map)`
**功能**: 標準化數據集欄位
**參數**:
- `df`: 原始數據框
- `dataset_type`: 數據集類型 ('KOI' 或 'TESS')
- `column_map`: 欄位對應字典
**返回**: 標準化後的數據框
**說明**: 統一欄位命名，處理數據類型，標準化狀態標籤

### `remove_outliers_from_columns(df, columns_to_clean)`
**功能**: 移除指定欄位的離群值
**參數**:
- `df`: 輸入數據框
- `columns_to_clean`: 需要清理的欄位列表
**返回**: 清理後的數據框
**說明**: 使用 IQR 方法 (Q1-1.5×IQR, Q3+1.5×IQR) 識別離群值

### `impute_missing_values_knn(df)`
**功能**: 使用 KNN 算法填補缺失值
**參數**:
- `df`: 輸入數據框
**返回**: 填補後的數據框
**說明**: 使用 5 個最近鄰居進行缺失值填補，保持數據分布特性

### `engineer_features(df)`
**功能**: 進行特徵工程
**參數**:
- `df`: 輸入數據框
**返回**: 包含新特徵的數據框
**說明**: 計算 SNR 和行星-恆星半徑比等物理特徵

### `augment_data(df, n_samples_per_row=1, random_state=42)`
**功能**: 數據擴增
**參數**:
- `df`: 輸入數據框
- `n_samples_per_row`: 每筆原始數據生成的擴增樣本數
- `random_state`: 隨機種子
**返回**: 擴增後的數據框
**說明**: 基於測量誤差進行高斯擾動，保持物理合理性

## 使用範例

### 基本使用
```python
# 處理單一數據集
dataset_type, raw_df, column_map = identify_and_validate_dataset('koi_data.csv')
standardized_df = standardize_dataset(raw_df, dataset_type, column_map)
cleaned_df = remove_outliers_from_columns(standardized_df, ['stellar_radius', 'stellar_logg'])
imputed_df = impute_missing_values_knn(cleaned_df)
augmented_df = augment_data(imputed_df, n_samples_per_row=2)
final_df = engineer_features(augmented_df)
```

### 批量處理多個數據集
```python
# 處理多個數據集並合併
datasets = ['koi_data.csv', 'tess_data.csv']
all_standardized_dfs = []

for dataset_path in datasets:
    try:
        dataset_type, raw_df, column_map = identify_and_validate_dataset(dataset_path)
        standardized_df = standardize_dataset(raw_df, dataset_type, column_map)
        all_standardized_dfs.append(standardized_df)
    except Exception as e:
        print(f"處理 {dataset_path} 時發生錯誤: {e}")

# 合併所有數據集
if all_standardized_dfs:
    combined_df = pd.concat(all_standardized_dfs, ignore_index=True, sort=False)
```

## 數據擴增策略

### 基於測量誤差的擾動
系統會根據以下欄位的測量誤差進行高斯擾動：
- 軌道週期 (`period`)
- 凌日持續時間 (`duration`)
- 凌日中點時間 (`transit_midpoint`)
- 凌日深度 (`depth`)
- 行星半徑 (`planet_radius`)
- 恆星入射通量 (`insolation_flux`)
- 恆星溫度 (`stellar_temp`)
- 恆星半徑 (`stellar_radius`)
- 恆星表面重力 (`stellar_logg`)

### 啟發式擾動
對於沒有明確誤差信息的欄位，使用啟發式方法：
- 恆星亮度 (`stellar_magnitude`): 5% 標準差擾動

### 擾動公式
對於有誤差信息的欄位：
```
σ = (|err_upper| + |err_lower|) / 2
new_value = original_value + N(0, σ)
```

## 輸出文件

處理完成後，系統會生成：
- `exoplanet_data_processed.csv`: 最終處理完成的數據集
- 控制台輸出：詳細的處理統計和診斷信息

## 依賴套件

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## 注意事項

1. **數據格式**: 輸入文件必須是 CSV 格式，支持 '#' 開頭的註釋行
2. **記憶體使用**: 數據擴增會顯著增加記憶體使用量，請根據系統配置調整 `n_samples_per_row` 參數
3. **隨機性**: 數據擴增使用固定隨機種子確保結果可重現
4. **物理合理性**: 擴增過程保持數據的物理合理性，避免生成不合理的數值

## 錯誤處理

系統包含完整的錯誤處理機制：
- 文件不存在錯誤
- 數據格式錯誤
- 欄位缺失錯誤
- 數據類型轉換錯誤

所有錯誤都會提供詳細的錯誤信息，幫助用戶快速定位和解決問題。
