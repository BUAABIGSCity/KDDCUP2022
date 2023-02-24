# Spatial-Temporal Graph Neural Network for Wind Power Forecasting in Baidu KDD CUP 2022

This repository contains the Baidu KDD CUP2022 solution by [aptx1231](https://github.com/aptx1231), [NickHan-cs](https://github.com/NickHan-cs).  

Our team name is **BUAA\_BIGSCity**, winning 11th place in 2490 teams. 

The overall structure of the codes refers to the [official baseline](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2022/wpf_baseline). The `submit.zip` here is our final submission file.

See more details in our paper [BUAA_BIGSCity: Spatial-Temporal Graph Neural Network for Wind Power Forecasting in Baidu KDD CUP 2022](https://arxiv.org/abs/2302.11159).

## Overview

Wind power is a rapidly growing source of clean energy. Accurate wind power forecasting is essential for grid stability and the security of supply. Therefore, organizers provide a wind power dataset containing historical data from 134 wind turbines and launch the [Baidu KDD Cup 2022](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction) to examine the limitations of current methods for wind power forecasting. The average of RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) is used as the evaluation score. 

We adopt two spatial-temporal graph neural network models, i.e., AGCRN and MTGNN, as our basic models. We train AGCRN by 5-fold cross-validation and additionally train MTGNN directly on the training and validation sets. Finally, we ensemble the two models based on the loss values of the validation set as our final submission. Using our method, our team **BUAA\_BIGSCity** achieves -45.36026 on the test set.

## Train

First place the [data](https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets) in the `data/` directory. Then run the script `gen_graph.py` to generate the geographic distance graph. After this, you can train the models.

### MTGNN

You can run the following command to train the MTGNN model on a 214-day training sets and a 31-day validation sets.

```shell
python main.py --model MTGNN --gpu_id 0 --random false --only_useful true --var_len 5 --graph_type geo --add_apt false --loss FilterHuberLoss --data_diff false
```

### AGCRN

We adopt a 5-fold cross-validation strategy for training AGCRN. You can run the following 5 commands in order to train the model.

```shell
python main_kfold.py --model AGCRN --gpu_id 3 --random false --only_useful true --var_len 6 --graph_type dtw --add_apt true --loss FilterHuberLoss --data_diff true --ind 0

python main_kfold.py --model AGCRN --gpu_id 0 --random false --only_useful true --var_len 6 --graph_type dtw --add_apt true --loss FilterHuberLoss --data_diff true --ind 1

python main_kfold.py --model AGCRN --gpu_id 0 --random false --only_useful true --var_len 6 --graph_type dtw --add_apt true --loss FilterHuberLoss --data_diff true --ind 2

python main_kfold.py --model AGCRN --gpu_id 0 --random false --only_useful true --var_len 6 --graph_type dtw --add_apt true --loss FilterHuberLoss --data_diff true --ind 3

python main_kfold.py --model AGCRN --gpu_id 0 --random false --only_useful true --var_len 6 --graph_type dtw --add_apt true --loss FilterHuberLoss --data_diff true --ind 4
```

The model will be saved in the `output/` directory during training, and the pre-trained model have been put in the `kfold_dtw_5_data_diff/` directory.

### Fusion

After training, we perform a weighted fusion of the prediction results of the 5 AGCRN models based on the reciprocals of valid losses. After obtaining the ensembled AGCRN model, we integrate the ensembled AGCRN model and MTGNN model again according to the ratio of **4:6** and obtain the final model prediction results. 

Using this method, we achieve **-45.36026** on the test set finally. The `submit.zip` here is our final submission file.

## Inference

You can run the `test.py` script to make inferences based on the model, only **single model** inferences are supported here. Here you need to specify the parameters `exp_id` and `best` to specify the directory of the optimal model.

```shell
python test.py --model MTGNN --gpu_id 0 --random false --only_useful true --var_len 5 --graph_type geo --add_apt false --data_diff false --exp_id 23577 --best 0

python test.py --model AGCRN --gpu_id 0 --random false --only_useful true --var_len 6 --graph_type dtw --add_apt true --data_diff true --exp_id 8204 --best 0
```

To construct the submission file (`submit.zip`), you can package all the files except `metrics.py` and `data/` into a zip file and submit it to the system for evaluation.

## Reference

```
# AGCRN
Bai L, Yao L, Li C, et al. Adaptive graph convolutional recurrent network for traffic forecasting[J]. Advances in neural information processing systems, 2020, 33: 17804-17815.

# MTGNN
Wu Z, Pan S, Long G, et al. Connecting the dots: Multivariate time series forecasting with graph neural networks[C]//Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2020: 753-763.
```
