# nCL-in-ReChorus
nCL算法旨在解决协同过滤的维度崩溃问题，我们在ReChorus轻量级推荐算法框架上实现了nCL推荐算法。

ReChorus中models内nCL.py、helpers内nCLRunnner.py是我们新增的代码文件，我们的代码在GPU下使用Pytorch实现。

## 运行指南
1.运行nCL模型代码首先需要配置ReChorus的环境，详细步骤请看ReChorus原址

2.安装POT库，使用IOPT方法实现会员矩阵更新，请执行以下命令：
```
pip install POT
```

3.请执行以下命令运行nCL模型：
进入src文件夹
```
cd src
```
调用nCL模型
```
python main.py --model_name nCL 
```
nCL模型在原有rechorus命令行参数上增加K、alpha、epsilon、iopt_iteration参数，
默认值设为--k 10 --alpha 0.5 --epsilon 0.001 --iopt_iteration 10 可以改变k和alpha调整模型参数
其他参数为ReChorus原有的，可根据ReChorus自行修改.

## Acknowledgements
论文链接 https://dl.acm.org/doi/10.1145/3616855.3635832

