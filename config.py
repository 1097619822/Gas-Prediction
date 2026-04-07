import os

# ==========================================
# 路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_results")

# ==========================================
# 列名映射 (修正后，精确匹配原始乱码字段)
# ==========================================
COL_MAP = {
    '鏃ユ湡': 'date',
    ' 浜曞彿': 'well_id',
    '鐢熶骇\n灞備綅': 'layer',
    '鎶曚骇澶╂暟': 'prod_days',
    '鐢熶骇鏃堕棿': 'prod_hours',
    '浜曞彛鍘嬪姏': 'wellhead_press',
    '鏃ヤ骇閲': 'gas_volume',
    '绱浜ф皵閲104m3': 'cum_gas',
    '绱浜ф按閲弇3': 'cum_water',
    '杩涚珯\n鍘嬪姏\nMpa': 'inlet_press',
    '杩涚珯\n娓╁害\n鈩': 'inlet_temp',
}

# ==========================================
# 实验参数
# ==========================================
SIGMA_THRESHOLD = 3.0
INTERPOLATE_METHOD = 'linear'
LAG_DAYS = [1, 3, 7]
ROLLING_WINDOWS = [3, 7]

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
