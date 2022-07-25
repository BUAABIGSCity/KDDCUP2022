# BUAA\_BIGSCity：Spatial-Temporal Graph Neural Network for Wind Power Forecasting in Baidu KDD CUP 2022

python main.py --model MTGNN --gpu_id 3 --random false --only_useful true --var_len 5 --graph_type geo --add_apt false --loss FilterHuberLoss --epoch 1

| python main_new.py --model AGCRNv1 --gpu_id 0 --random false --only_useful true --var_len 6 --scale all --graph_type dtw --loss FilterHuberLoss --data_diff true --ind 0 --epoch 1 |
| ------------------------------------------------------------ |
| python main_new.py --model AGCRNv1 --gpu_id 0 --random false --only_useful true --var_len 6 --scale all --graph_type dtw --loss FilterHuberLoss --data_diff true --ind 1 |
| python main_new.py --model AGCRNv1 --gpu_id 1 --random false --only_useful true --var_len 6 --scale all --graph_type dtw --loss FilterHuberLoss --data_diff true --ind 2 |
| python main_new.py --model AGCRNv1 --gpu_id 2 --random false --only_useful true --var_len 6 --scale all --graph_type dtw --loss FilterHuberLoss --data_diff true --ind 3 |
| python main_new.py --model AGCRNv1 --gpu_id 0 --random false --only_useful true --var_len 6 --scale all --graph_type dtw --loss FilterHuberLoss --data_diff true --ind 4 |