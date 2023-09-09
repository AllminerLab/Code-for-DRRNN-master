
### 运行环境

```python
python 2.7.13
torchvision 0.1.8
pytorch 1.4.0
```

### 运行示例

- 运行预处理代码

```sh
python ./run.py \
-r ../data/Patio/Patio_Lawn_and_Garden_5.json \
-d ../data/Patio/input_data/ \
-rt ../data/Patio/preprocessed/rating \
-re ../data/Patio/preprocessed/review \
-o ../data/Patio/result/DRRNN \
-m 300 \
-p ../data/glove/glove.6B.300d.txt \
-f 32 \
-c True
```

- 运行模型代码

```
python ./run.py \
-r ../data/Patio/Patio_Lawn_and_Garden_5.json \
-d ../data/Patio/input_data/ \
-rt ../data/Patio/preprocessed/rating \
-re ../data/Patio/preprocessed/review \
-o ../data/Patio/result/DRRNN \
-m 300 \
-p ../data/glove/glove.6B.300d.txt \
-f 32 \
-c False
```

文献：
Wu-Dong Xi, Ling Huang, Chang-Dong Wang, Yin-Yu Zheng and Jian-Huang Lai. "Deep Rating and Review Neural Network for Item Recommendation", TNNLS2022
