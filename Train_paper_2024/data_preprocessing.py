import pandas as pd
import os
import numpy as np

# --- 常數定義 (此處省略以節省空間，與前次相同) ---
STANDARD_COLUMNS = [
    'source_id', 'source_telescope', 'disposition', 'period', 'period_err_upper',
    'period_err_lower', 'duration', 'duration_err_upper', 'duration_err_lower',
    'transit_midpoint', 'transit_midpoint_err_upper', 'transit_midpoint_err_lower',
    'depth', 'depth_err_upper', 'depth_err_lower', 'planet_radius',
    'planet_radius_err_upper', 'planet_radius_err_lower', 'equilibrium_temp',
    'insolation_flux', 'stellar_magnitude', 'stellar_temp', 'stellar_temp_err_upper',
    'stellar_temp_err_lower', 'stellar_radius', 'stellar_radius_err_upper',
    'stellar_radius_err_lower', 'stellar_logg', 'stellar_logg_err_upper',
    'stellar_logg_err_lower', 'ra', 'dec'
]
KOI_COLUMN_MAP = {
    'kepoi_name': 'source_id', 'koi_disposition': 'disposition', 'koi_period': 'period',
    'koi_period_err1': 'period_err_upper', 'koi_period_err2': 'period_err_lower',
    'koi_duration': 'duration', 'koi_duration_err1': 'duration_err_upper',
    'koi_duration_err2': 'duration_err_lower', 'koi_time0bk': 'transit_midpoint',
    'koi_time0bk_err1': 'transit_midpoint_err_upper', 'koi_time0bk_err2': 'transit_midpoint_err_lower',
    'koi_depth': 'depth', 'koi_depth_err1': 'depth_err_upper', 'koi_depth_err2': 'depth_err_lower',
    'koi_prad': 'planet_radius', 'koi_prad_err1': 'planet_radius_err_upper',
    'koi_prad_err2': 'planet_radius_err_lower', 'koi_teq': 'equilibrium_temp',
    'koi_insol': 'insolation_flux', 'koi_kepmag': 'stellar_magnitude',
    'koi_steff': 'stellar_temp', 'koi_steff_err1': 'stellar_temp_err_upper',
    'koi_steff_err2': 'stellar_temp_err_lower', 'koi_srad': 'stellar_radius',
    'koi_srad_err1': 'stellar_radius_err_upper', 'koi_srad_err2': 'stellar_radius_err_lower',
    'koi_slogg': 'stellar_logg', 'koi_slogg_err1': 'stellar_logg_err_upper',
    'koi_slogg_err2': 'stellar_logg_err_lower', 'ra': 'ra', 'dec': 'dec'
}
TESS_COLUMN_MAP = {
    'toi': 'source_id', 'tfopwg_disp': 'disposition', 'pl_orbper': 'period',
    'pl_orbpererr1': 'period_err_upper', 'pl_orbpererr2': 'period_err_lower',
    'pl_trandurh': 'duration', 'pl_trandurherr1': 'duration_err_upper',
    'pl_trandurherr2': 'duration_err_lower', 'pl_tranmid': 'transit_midpoint',
    'pl_tranmiderr1': 'transit_midpoint_err_upper', 'pl_tranmiderr2': 'transit_midpoint_err_lower',
    'pl_trandep': 'depth', 'pl_trandeperr1': 'depth_err_upper', 'pl_trandeperr2': 'depth_err_lower',
    'pl_rade': 'planet_radius', 'pl_radeerr1': 'planet_radius_err_upper',
    'pl_radeerr2': 'planet_radius_err_lower', 'pl_eqt': 'equilibrium_temp',
    'pl_insol': 'insolation_flux', 'st_tmag': 'stellar_magnitude', 'st_teff': 'stellar_temp',
    'st_tefferr1': 'stellar_temp_err_upper', 'st_tefferr2': 'stellar_temp_err_lower',
    'st_rad': 'stellar_radius', 'st_raderr1': 'stellar_radius_err_upper',
    'st_raderr2': 'stellar_radius_err_lower', 'st_logg': 'stellar_logg',
    'st_loggerr1': 'stellar_logg_err_upper', 'st_loggerr2': 'stellar_logg_err_lower',
    'ra': 'ra', 'dec': 'dec'
}

# --- 函式定義 ---

def identify_and_validate_dataset(file_path: str):
    # (函式內容不變)
    try:
        df = pd.read_csv(file_path, comment='#')
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。請確認路徑和檔名是否正確。")
        raise
    columns = set(df.columns)
    if 'kepoi_name' in columns:
        dataset_type = 'KOI'
        column_map = KOI_COLUMN_MAP
    elif 'toi' in columns:
        dataset_type = 'TESS'
        column_map = TESS_COLUMN_MAP
    else:
        raise ValueError(f"無法識別檔案 '{file_path}' 的資料類型。")
    required_columns = set(column_map.keys())
    missing_cols = required_columns - columns
    if missing_cols:
        raise ValueError(f"檔案 '{file_path}' 被識別為 {dataset_type} 資料，但缺少以下必要欄位：{list(missing_cols)}")
    return dataset_type, df, column_map

def standardize_dataset(df: pd.DataFrame, dataset_type: str, column_map: dict):
    # (函式內容不變)
    standardized_df = df[list(column_map.keys())].copy()
    standardized_df.rename(columns=column_map, inplace=True)
    standardized_df['source_telescope'] = dataset_type
    if dataset_type == 'TESS':
        disposition_mapping = {
            'CP': 'CONFIRMED', 'KP': 'CONFIRMED', 'PC': 'CANDIDATE',
            'APC': 'CANDIDATE', 'FP': 'FALSE POSITIVE', 'FA': 'FALSE POSITIVE'
        }
        standardized_df['disposition'] = standardized_df['disposition'].map(lambda x: disposition_mapping.get(x, x))
    return standardized_df

# *** 新增功能：根據指定欄位刪除離群值 ***
def remove_outliers_from_columns(df: pd.DataFrame, columns_to_clean: list) -> pd.DataFrame:
    """
    使用 IQR 方法，刪除在指定欄位列表中為離群值的資料行。
    """
    print("\n" + "="*50)
    print(f"--- 刪除指定欄位的離群值 ---")
    print(f"目標欄位: {columns_to_clean}")
    print("="*50 + "\n")

    initial_rows = len(df)
    indices_to_drop = set()

    for col in columns_to_clean:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        print(f"在欄位 '{col}' 中找到 {len(outlier_indices)} 個離群值。")
        indices_to_drop.update(outlier_indices)
    
    if indices_to_drop:
        df_cleaned = df.drop(index=list(indices_to_drop))
        rows_dropped = len(indices_to_drop)
        print(f"\n總共識別出 {rows_dropped} 筆需要刪除的資料。")
        print(f"原始資料筆數: {initial_rows}")
        print(f"清理後資料筆數: {len(df_cleaned)}")
        print(f"刪除比例: {rows_dropped/initial_rows:.2%}")
    else:
        print("\n在指定欄位中未找到需要刪除的離群值。")
        df_cleaned = df.copy()
        
    print("\n" + "-"*50 + "\n")
    return df_cleaned


# --- 主程式執行區塊 ---
base_path = 'Data'
koi_file_name = 'cumulative_2025.09.26_19.48.09.csv'
tess_file_name = 'TOI_2025.09.26_19.47.35.csv'
koi_file_path = os.path.join(base_path, koi_file_name)
tess_file_path = os.path.join(base_path, tess_file_name)
all_dfs = []

# ... (資料讀取與標準化部分維持不變) ...
try:
    print("--- 開始處理克普勒 (KOI) 資料 ---")
    koi_type, koi_raw_df, koi_map = identify_and_validate_dataset(koi_file_path)
    standardized_koi_df = standardize_dataset(koi_raw_df, koi_type, koi_map)
    all_dfs.append(standardized_koi_df)
except (ValueError, FileNotFoundError) as e:
    print(f"處理 KOI 資料時發生錯誤：{e}")

try:
    print("\n--- 開始處理 TESS 資料 ---")
    tess_type, tess_raw_df, tess_map = identify_and_validate_dataset(tess_file_path)
    standardized_tess_df = standardize_dataset(tess_raw_df, tess_type, tess_map)
    all_dfs.append(standardized_tess_df)
except (ValueError, FileNotFoundError) as e:
    print(f"處理 TESS 資料時發生錯誤：{e}")


# --- 合併、分析與儲存 ---
if len(all_dfs) > 0:
    print("\n" + "="*50)
    print("--- 開始合併標準化資料 ---")
    if 'is_outlier' not in STANDARD_COLUMNS:
        STANDARD_COLUMNS.append('is_outlier')
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df[STANDARD_COLUMNS[:-1]] # 暫時不含 is_outlier
    
    print(f"資料合併完成！總資料筆數: {combined_df.shape[0]}")

    # *** 呼叫新增的離群值刪除功能 ***
    # 只針對 stellar_radius 和 stellar_logg 進行清理
    cleaned_df = remove_outliers_from_columns(
        combined_df, 
        columns_to_clean=['stellar_radius', 'stellar_logg']
    )
    
    # *** 後續所有分析都基於 cleaned_df ***
    
    # 重新為清理後的數據加上離群值標記
    # 注意：這裡的離群值是相對於「清理後」的數據集，會與原始的不同
    cleaned_df = add_outlier_flag_column(cleaned_df)
    cleaned_df = cleaned_df[STANDARD_COLUMNS] # 確保欄位順序
    
    print("\n--- 現在，所有分析將在清理後的數據上進行 ---")

    find_and_rank_extremes(cleaned_df, top_n=20)
    
    count_values_greater_than(cleaned_df, threshold=5)
    
    print("\n清理後資料集的前 5 筆資料預覽：")
    print(cleaned_df.head())
    
    # 儲存清理後的檔案
    output_filename = 'combined_exoplanet_data_cleaned.csv'
    cleaned_df.to_csv(output_filename, index=False)
    print(f"\n已將清理後的資料集儲存為 '{output_filename}'")
else:
    print("\n錯誤：沒有任何資料集被成功處理，無法進行合併。")

# 其他函式定義，為了完整性貼上，內容不變
def add_outlier_flag_column(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*50)
    print("--- 為數據集新增離群值標記 ('is_outlier') 欄位 ---")
    numeric_cols_to_check = [
        'period', 'duration', 'depth', 'planet_radius', 'equilibrium_temp',
        'insolation_flux', 'stellar_magnitude', 'stellar_temp',
        'stellar_radius', 'stellar_logg'
    ]
    outlier_indices = set()
    for col in numeric_cols_to_check:
        data = df[col].dropna()
        if len(data) == 0: continue
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        current_outliers = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_indices.update(current_outliers)
    df['is_outlier'] = False
    df.loc[list(outlier_indices), 'is_outlier'] = True
    num_outliers = df['is_outlier'].sum()
    print(f"標記完成！共找到 {num_outliers} 筆資料在至少一個欄位中為離群值。")
    print("="*50 + "\n")
    return df

def find_and_rank_extremes(df: pd.DataFrame, top_n: int = 20):
    print("\n" + "="*60)
    print("--- 尋找極端值之最 (Top Outliers by Robust Z-score) ---")
    print("="*60 + "\n")
    numeric_cols_to_check = [
        'period', 'duration', 'depth', 'planet_radius', 'equilibrium_temp',
        'insolation_flux', 'stellar_magnitude', 'stellar_temp',
        'stellar_radius', 'stellar_logg'
    ]
    all_extremes = []
    for col in numeric_cols_to_check:
        data = df[col].dropna()
        if len(data) < 3: continue
        median_val = data.median()
        mad_val = (data - median_val).abs().median()
        if mad_val == 0: continue
        robust_z_scores = (data - median_val) / (1.4826 * mad_val)
        df.loc[robust_z_scores.index, 'temp_robust_z'] = robust_z_scores
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)].copy()
        for index, row in outliers_df.iterrows():
            all_extremes.append({
                'source_id': row['source_id'], 'extreme_field': col,
                'extreme_value': row[col], 'robust_z_score': row.get('temp_robust_z'),
                'median_value': median_val
            })
        df.drop(columns=['temp_robust_z'], inplace=True, errors='ignore')
    if not all_extremes:
        print("在清理後的數據中未找到極端值。")
        return
    extremes_report_df = pd.DataFrame(all_extremes)
    extremes_report_df['abs_z_score'] = extremes_report_df['robust_z_score'].abs()
    final_report = extremes_report_df.sort_values(by='abs_z_score', ascending=False).head(top_n)
    print(f"報告：清理後數據集中最極端的 Top {top_n} 筆數據")
    print("穩健 Z 分數越高，代表該值在它的屬性中越罕見、越極端。\n")
    print(final_report[['source_id', 'extreme_field', 'extreme_value', 'robust_z_score', 'median_value']].to_string(index=False))
    print("-" * 60 + "\n")

def count_values_greater_than(df: pd.DataFrame, threshold: int = 5):
    print("\n" + "="*50)
    print(f"--- 統計數值大於 {threshold} 的資料筆數 ---")
    print("="*50 + "\n")
    numeric_cols_to_check = [
        'period', 'duration', 'depth', 'planet_radius', 'equilibrium_temp',
        'insolation_flux', 'stellar_magnitude', 'stellar_temp',
        'stellar_radius', 'stellar_logg'
    ]
    for col in numeric_cols_to_check:
        data = df[col].dropna()
        if len(data) == 0:
            print(f"欄位 '{col}': 沒有有效數據。")
            continue
        count_gt = (data > threshold).sum()
        percentage = (count_gt / len(data)) * 100 if len(data) > 0 else 0
        print(f"欄位 '{col}': 有 {count_gt} 筆資料 > {threshold} (佔總 {len(data)} 筆的 {percentage:.2f}%)")
    print("\n" + "-"*50 + "\n")