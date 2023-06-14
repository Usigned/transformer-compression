# 资源受限环境下Transformer压缩方法研究

## 资源消耗预测

数据集：

- json格式

- tmp/*.json

测资源消耗

- 入口程序：profier.py
- 生成库：gen.py

## Transformer剪枝量化方法

> 若只需要剪枝可以切换至mvit分支

剪枝：mmsa.py

量化：quant_utils.py

vit: model.py

训练：train.py -> train_mvit

测试：eval.py

## RL部分

> 最终效果不好且代码较长，请谨慎尝试

入口：rl_train.py

环境：env.py

智能体：ddpg.py

