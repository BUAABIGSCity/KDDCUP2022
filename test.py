import os
import glob
import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from easydict import EasyDict as edict
from MTGNN import MTGNN
from AGCRN import AGCRN
from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
from metrics import regressor_detailed_scores
from utils import load_model, get_logger, str2bool
from logging import getLogger


def predict(config, train_data):
    log = getLogger()
    name2id = {
        'weekday': 0,
        'time': 1,
        'Wspd': 2,
        'Wdir': 3,
        'Etmp': 4,
        'Itmp': 5,
        'Ndir': 6,
        'Pab1': 7,
        'Pab2': 8,
        'Pab3': 9,
        'Prtv': 10,
        'Patv': 11
    }
    select = config.select
    select_ind = [name2id[name] for name in select]

    with torch.no_grad():
        data_mean = torch.FloatTensor(train_data.data_mean).to(config.device)  # (1, 134, 1, 1)
        data_scale = torch.FloatTensor(train_data.data_scale).to(config.device)  # (1, 134, 1, 1)

        graph = train_data.graph  # (134, 134)

        if config.model == 'MTGNN':
            model = MTGNN(config=config, adj_mx=graph).to(config.device)
        elif config.model == 'AGCRN':
            model = AGCRN(config=config, adj_mx=graph).to(config.device)
        else:
            raise ValueError('Error config.model = {}'.format(config.model))

        output_path = config.output_path+config.exp_id+'_'+config.model
        load_model(os.path.join(output_path, "model_%d.pt" % config.best), model, log=log)

        model.eval()

        test_x = sorted(glob.glob(os.path.join("./data", "test_x", "*")))
        test_y = sorted(glob.glob(os.path.join("./data", "test_y", "*")))

        maes, rmses = [], []
        for i, (test_x_f, test_y_f) in enumerate(zip(test_x, test_y)):
            test_x_ds = TestPGL4WPFDataset(filename=test_x_f)  # (B,N,T,F)

            test_y_ds = TestPGL4WPFDataset(filename=test_y_f)  # (B,N,T,F)

            if config.only_useful:
                test_x = torch.FloatTensor(
                    test_x_ds.get_data()[:, :, -config.input_len:, select_ind]).to(config.device)
                test_y = torch.FloatTensor(
                    test_y_ds.get_data()[:, :, :config.output_len, select_ind]).to(config.device)
            else:
                test_x = torch.FloatTensor(
                    test_x_ds.get_data()[:, :, -config.input_len:, :]).to(config.device)
                test_y = torch.FloatTensor(
                    test_y_ds.get_data()[:, :, :config.output_len, :]).to(config.device)

            pred_y = model(test_x, None, data_mean, data_scale)  # (B,N,T)
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            pred_y = np.expand_dims(pred_y.cpu().numpy(), -1)  # (B,N,T,1)
            test_y = test_y[:, :, :, -1:].cpu().numpy()  # (B,N,T,F)

            pred_y = np.transpose(pred_y, [  # (N,B,T,1)
                1,
                0,
                2,
                3,
            ])
            test_y = np.transpose(test_y, [  # (N,B,T,F)
                1,
                0,
                2,
                3,
            ])
            test_y_df = test_y_ds.get_raw_df()

            _mae, _rmse = regressor_detailed_scores(
                pred_y, test_y, test_y_df, config.capacity, config.output_len)
            print('\n\tThe {}-th prediction for File {} -- '
                  'RMSE: {}, MAE: {}, Score: {}'.format(i, test_y_f, _rmse, _mae, (
                      _rmse + _mae) / 2))
            maes.append(_mae)
            rmses.append(_rmse)

        avg_mae = np.array(maes).mean()
        avg_rmse = np.array(rmses).mean()
        total_score = (avg_mae + avg_rmse) / 2

        print('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))
        print('--- Final Score --- \n\t{}'.format(total_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--model", type=str, default="MTGNN")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--input_len", type=int, default=144, help='input data len')
    parser.add_argument("--output_len", type=int, default=288, help='output data len')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_days", type=int, default=214)
    parser.add_argument("--val_days", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_id", type=str, default='55237')
    parser.add_argument("--best", type=int, default=0)
    parser.add_argument("--output_path", type=str, default='kfold_dtw_5_data_diff/')

    parser.add_argument("--random", type=str2bool, default=False, help='Whether shuffle num_nodes')
    parser.add_argument("--enhance", type=str2bool, default=True, help='Whether enhance the time dim')
    parser.add_argument("--only_useful", type=str2bool, default=True, help='Whether remove some feature')
    parser.add_argument("--var_len", type=int, default=5, help='Dimensionality of input features')
    parser.add_argument("--data_diff", type=str2bool, default=False, help='Whether to use data differential features')
    parser.add_argument("--add_apt", type=str2bool, default=False, help='Whether to use adaptive matrix')
    parser.add_argument("--binary", type=str2bool, default=True, help='Whether to set the adjacency matrix as binary')
    parser.add_argument("--graph_type", type=str, default="geo", help='graph type, dtw or geo')
    parser.add_argument("--dtw_topk", type=int, default=5, help='M dtw for dtw graph')
    parser.add_argument("--weight_adj_epsilon", type=float, default=0.8, help='epsilon for geo graph')
    parser.add_argument("--gsteps", type=int, default=1, help='Gradient Accumulation')
    parser.add_argument("--loss", type=str, default='FilterHuberLoss')
    parser.add_argument("--select", nargs='+', type=str,
                        default=['weekday', 'time', 'Wspd', 'Etmp', 'Itmp', 'Prtv', 'Patv'])

    args = parser.parse_args()
    dict_args = vars(args)

    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(dict_args)

    logger = get_logger(config)
    logger.info(config)

    size = [config.input_len, config.output_len]
    train_data = PGL4WPFDataset(
        config.data_path,
        filename=config.filename,
        size=[config.input_len, config.output_len],
        flag='train',
        total_days=config.total_days,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days,
        random=config.random,
        only_useful=config.only_useful,
        graph_type=config.graph_type,
        weight_adj_epsilon=config.weight_adj_epsilon,
        dtw_topk=config.dtw_topk,
        binary=config.binary,
    )

    gpu_id = config.gpu_id
    if gpu_id != -1:
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    config['device'] = device
    predict(config, train_data)  #, valid_data, test_data)
