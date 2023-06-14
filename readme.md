# 资源受限环境下Transformer压缩方法研究

> link: https://gitee.com/aloha-qing/design-code
>
> 分支介绍：
>
> - master：合并了所有分支，最后一次合并时未检查，可能有Bug导致无法运行
> - huawei：用于华为云平台，代码较全
> - mvit：仅包含关于剪枝相关的内容
> - rl：包含第四章和强化学习相关内容

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

- QuantPruneEnv：限制智能体动作空间
- CombQuantPruneEnv：资源消耗和性能加权
- 详见二者奖励函数

智能体：ddpg.py

