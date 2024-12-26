# nCL-in-ReChorus
## 我们在rechorus轻量级推荐算法框架上实现了nCL推荐算法，且我们在cpu和gpu上都实现了nCL推荐算法。

rechorus中的nCL.py、nCLRunnner.py、nCL_g.py、CLRunner_g.py是我们新增的函数，前两个函数是在cpu上运行，后两个函数是在gpu下实现。

运行nCL模型需要安装POT库，请执行以下命令：
```
pip install POT
```

nCL模型在原有rechorus命令行参数上增加K、alpha、epsilon、iopt_iteration参数，
根据论文默认值设为--K 10 --alpha 1.0 --epsilon 0.001 --iopt_iteration 10

请执行以下命令运行cpu版本nCL模型：
```
python main.py --model_name nCL
```

请执行以下命令运行gpu版本nCL模型：
```
python main.py --model_name nCL_g 
```

## Acknowledgements
论文链接https://dl.acm.org/doi/10.1145/3616855.3635832

