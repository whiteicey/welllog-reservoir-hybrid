# PERM frozen error sources

{
  "screening_fold": 5,
  "complete_five_folds": false,
  "PERM": 89.40556485691857,
  "groups": {
    "default_pred_default": {
      "rows": 91846,
      "fraction": 0.6348874987039022,
      "PERM": 100.0,
      "deficit_contribution": 0.0,
      "mean_log10_prediction_ratio": 0.0
    },
    "default_pred_continuous": {
      "rows": 3687,
      "fraction": 0.025486468738119102,
      "PERM": 29.489167996346513,
      "deficit_contribution": 1.7970721155598823,
      "mean_log10_prediction_ratio": 1.0394165666495698
    },
    "continuous_pred_default": {
      "rows": 5805,
      "fraction": 0.04012719040541942,
      "PERM": 25.530065607668426,
      "deficit_contribution": 2.988269236840181,
      "mean_log10_prediction_ratio": -1.1525132829378726
    },
    "continuous_pred_continuous": {
      "rows": 43327,
      "fraction": 0.29949884215255934,
      "PERM": 80.60395242622567,
      "deficit_contribution": 5.809093790681368,
      "mean_log10_prediction_ratio": -0.03482849596617917
    },
    "continuous_underprediction": {
      "rows": 30402,
      "fraction": 0.21015449486745238,
      "PERM": 69.47659385569952,
      "deficit_contribution": 6.414630999889564,
      "mean_log10_prediction_ratio": -0.3871622606747688
    },
    "continuous_overprediction": {
      "rows": 18730,
      "fraction": 0.12947153769052638,
      "PERM": 81.59648004391983,
      "deficit_contribution": 2.3827320276319885,
      "mean_log10_prediction_ratio": 0.1906648797038616
    },
    "continuous_zero_utility": {
      "rows": 3211,
      "fraction": 0.022196108250095048,
      "PERM": 0.0,
      "deficit_contribution": 2.2196108250095046,
      "mean_log10_prediction_ratio": -1.5370934923589459
    }
  },
  "scope": "Frozen descriptive diagnosis only. Overlapping sign/tail groups are not additive. No fitting, thresholds, row/well selection or input features derived from validation labels."
}

## 探索重点

拆解已冻结有效PERM第五折预测失分来源，为下一项训练机制提供依据。

## 已探索内容与验证

144665行固定预测与真实自身标签描述性比较，核对全PERM分与恢复报告一致；不拟合、不修正、不筛选预测，不改变原模型。

## 收益

没有新增模型收益；取得失分组成证据，不能当成准确率提升。

## 不值得继续的点及证据

不从验证误差拟合偏置、阈值、行选择或后处理，避免把诊断变成验证侧拟合。现有门控/低LR80-14试验无收益，仅限配置/预算，后续需要实际训练新证据。
