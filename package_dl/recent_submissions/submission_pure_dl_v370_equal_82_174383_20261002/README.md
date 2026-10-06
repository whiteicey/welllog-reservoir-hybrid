# submission_pure_dl_v370_equal_82_174383_20261002 源码镜像

来源为同名本地冻结交付包；predict.py和模型架构依赖逐文件复制。未附模型、测试数据或逐行预测。

取得对应五个fold_*.pt后放入models，运行：
```bash
python predict.py --data_dir /path/to/data --output result.json --device cuda
python validate_delivery.py --data_dir /path/to/data
```

交付分数、区别和验证边界见仓库根README与docs/deep_learning/RECENT_SUBMISSIONS.md。验证JSON来自本地包运行，不代表本次全新环境复现。历史训练脚本保留原路径语义，不承诺空目录一键重训。
