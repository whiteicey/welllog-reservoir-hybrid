"""v3 方案全局配置（树-序列混合 + Bayes 解码 + 嵌套融合）。"""

# ---------- 数据 ----------
SENTINEL = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
MISSING = -99999.0
TARGETS = ["POR", "PERM", "SW"]
CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM"]
# NN 输入用的 10 条曲线（AZIM 周期性弱、参考方案未用）
NN_CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI"]

# 预测裁剪范围
BOUNDS = {"POR": (0.0, 50.0), "PERM": (0.01, 10000.0), "SW": (8.3, 99.9)}

# ---------- 评分 ----------
EPS = 1e-5
POR_TOL = 0.08
SW_TOL = 0.05
SCORE_W = {"POR": 0.30, "PERM": 0.35, "SW": 0.35}

# ---------- 交叉验证 ----------
N_FOLDS = 5
SEED = 2026

# ---------- 分布模型 ----------
N_BINS = {"POR": 80, "PERM": 60, "SW": 60}
# 目标变换空间: POR log1p / PERM log10 / SW 恒等
DIST_PARAMS = dict(loss_function="MultiClass", iterations=1200, depth=8,
                   learning_rate=0.04, l2_leaf_reg=3.0, max_bin=128,
                   bootstrap_type="Bayesian", verbose=200)
DIST_SEEDS = [42, 2026]

# ---------- 门控（二分类）----------
GATE_ITERS = {"POR": 500, "PERM": 800, "SW": 950}
GATE_PARAMS = dict(loss_function="Logloss", depth=8, learning_rate=0.04,
                   max_bin=128, bootstrap_type="Bayesian", verbose=200)
GATE_SEEDS = [42, 2026]

# ---------- 4 状态联合模型 ----------
STATES_PARAMS = dict(loss_function="MultiClass", iterations=900, depth=8,
                     learning_rate=0.05, max_bin=128,
                     bootstrap_type="Bayesian", verbose=200)
STATES_SEEDS = [42, 2026]

# ---------- Bayes 解码 ----------
N_ACTIONS = 800

# ---------- WellNet ----------
NN_CHANNELS = 160
NN_BLOCKS = 12                       # dilation 1..32 循环两遍
NN_DILATIONS = [1, 2, 4, 8, 16, 32] * 2
NN_DROPOUT = 0.10
NN_CROP = 3072
NN_BATCH = 16
NN_EPOCHS = 60
NN_STEPS = 100
NN_LR = 2e-3
NN_WD = 1e-4
NN_GRAD_CLIP = 2.0
NN_GAIN_STD = 0.08
NN_OFFSET_STD = 0.15
# 两组配置: gate_b 偏门控 3 种子；base_c 拼树特征 2 种子
NN_CONFIGS = {
    "gate_b": dict(seeds=[1, 2, 3], bin_weight=0.3, use_base=False,
                   boundary_aware=False),
    "base_c": dict(seeds=[1, 2], bin_weight=1.0, use_base=True,
                   boundary_aware=False),
    # 超越线: 边界感知门控（层界邻域加权 + 薄层加权）
    "edge_b": dict(seeds=[1, 2, 3], bin_weight=0.3, use_base=False,
                   boundary_aware=True),
}
# 边界感知参数
EDGE_RADIUS = 5          # 层界 ± 半径内的点视为边界带
EDGE_WEIGHT = 4.0        # 边界带 BCE 权重倍数
THIN_LEN = 10            # 段长 < 该点数视为薄层
THIN_WEIGHT = 4.0        # 薄层段内点的 BCE 权重倍数
# base_c 拼接的树特征数上限（按方差排序取前 N，控制显存）
NN_BASE_TOPK = 113

# ---------- 融合 ----------
BLEND_STEP = 0.25                    # 权重单纯形步长
TH_GRID_N = 89                       # 阈值 0.02~0.90
