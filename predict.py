import os
import torch
import torch.nn.functional as F
import numpy as np
from MTGNN import MTGNN
from AGCRN import AGCRN
from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset


def predict(settings, data_mean, data_scale, graph):
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
    select = settings['select']
    select_ind = [name2id[name] for name in select]

    with torch.no_grad():
        data_mean = torch.FloatTensor(data_mean).to(settings['device'])  # (1, 134, 1, 1)
        data_scale = torch.FloatTensor(data_scale).to(settings['device'])  # (1, 134, 1, 1)

        if settings['model'] == 'MTGNN':
            model = MTGNN(config=settings, adj_mx=graph).to(settings['device'])
        elif settings['model'] == 'AGCRN':
            model = AGCRN(config=settings, adj_mx=graph).to(settings['device'])
        else:
            model = MTGNN(config=settings).to(settings['device'])

        path_to_model = os.path.join(settings["checkpoints"], settings["checkpoints_in"],
                                     "model_{}.pt".format(settings['best']))
        model_state = torch.load(path_to_model, map_location=settings['device'])
        model.load_state_dict(model_state)

        model.eval()

        test_x_ds = TestPGL4WPFDataset(filename=settings['path_to_test_x'])  # (B,N,T,F)

        if settings['only_useful']:
            test_x = torch.FloatTensor(
                test_x_ds.get_data()[:, :, -settings['input_len']:, select_ind]).to(settings['device'])
        else:
            test_x = torch.FloatTensor(
                test_x_ds.get_data()[:, :, -settings['input_len']:, :]).to(settings['device'])

        print(test_x.shape, data_mean.shape, data_scale.shape)
        pred_y = model(test_x, None, data_mean, data_scale)  # (B,N,T)
        pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])  # (B,N,T)

        pred_y = np.expand_dims(pred_y.cpu().numpy(), -1)[0]  # (N,T,1)

    return pred_y


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions as a tensor \in R^{134 * 288 * 1}
    """
    # AGCRN model prediction (model_list)
    res = []
    weights = []
    for i in range(len(settings['model_list'])):
        # Select one of the models at a time
        di = settings['model_list'][i]
        settings.update(di)
        print(settings)
        print(settings['model'], settings['checkpoints_in'], settings['weight'])
        weights.append(settings['weight'])

        # load data_mean/scale
        data_mean = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         "npy/data_mean_{}.npy".format(settings['train_days'])))
        data_scale = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          "npy/data_scale_{}.npy".format(settings['train_days'])))
        print('load {}'.format(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "npy/data_mean_{}.npy".format(settings['train_days']))))
        # load graph
        if settings['graph_type'] == "geo":
            graph = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "npy/geo_graph.npy"))
            distances = graph.flatten()
            dist_std = distances.std()
            graph = np.exp(-np.square(graph / dist_std))
            graph[graph < settings['weight_adj_epsilon']] = 0
            if settings['binary']:
                graph[graph >= settings['weight_adj_epsilon']] = 1
            print(f"geo graph links: {graph.sum()}")
        elif settings['graph_type'] == "dtw":
            if settings['ind'] != -1:
                graph = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             "npy/dtw_graph_top{}_{}_{}.npy".format(
                                                 settings['dtw_topk'], settings['ind'], settings['K'])))
            else:
                graph = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             "npy/dtw_graph_top{}.npy".format(settings['dtw_topk'])))
            print(f"dtw graph links: {graph.sum()}")
        else:
            raise ValueError('Error graph_type = {}'.format(settings['graph_type']))

        # predict
        predictions = predict(settings, data_mean, data_scale, graph)  # (N,T,1)
        res.append(predictions)

    # Multi-model fusion
    print(weights)
    total = (1.0 / weights[0])
    predictions1 = (1.0 / weights[0]) * res[0]
    for i in range(1, len(res)):
        predictions1 += (1.0 / weights[i]) * res[i]
        total += (1.0 / weights[i])
    predictions1 = predictions1 / total
    print(predictions1.shape)

    # MTGNN model prediction (model_list2)
    res = []
    weights = []
    for i in range(len(settings['model_list2'])):
        # Select one of the models at a time
        di = settings['model_list2'][i]
        settings.update(di)
        print(settings)
        print(settings['model'], settings['checkpoints_in'], settings['weight'])
        weights.append(settings['weight'])

        # load data_mean/scale
        data_mean = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         "npy/data_mean_{}.npy".format(settings['train_days'])))
        data_scale = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          "npy/data_scale_{}.npy".format(settings['train_days'])))
        print('load {}'.format(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "npy/data_mean_{}.npy".format(settings['train_days']))))
        # load graph
        if settings['graph_type'] == "geo":
            graph = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "npy/geo_graph.npy"))
            distances = graph.flatten()
            dist_std = distances.std()
            graph = np.exp(-np.square(graph / dist_std))
            graph[graph < settings['weight_adj_epsilon']] = 0
            if settings['binary']:
                graph[graph >= settings['weight_adj_epsilon']] = 1
            print(f"geo graph links: {graph.sum()}")
        elif settings['graph_type'] == "dtw":
            if settings['ind'] != -1:
                graph = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             "npy/dtw_graph_top{}_{}_{}.npy".format(
                                                 settings['dtw_topk'], settings['ind'], settings['K'])))
            else:
                graph = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             "npy/dtw_graph_top{}.npy".format(settings['dtw_topk'])))
            print(f"dtw graph links: {graph.sum()}")
        else:
            raise ValueError('Error graph_type = {}'.format(settings['graph_type']))

        # predict
        predictions = predict(settings, data_mean, data_scale, graph)  # (N,T,1)
        res.append(predictions)

    # Multi-model fusion
    print(weights)
    total = (1.0 / weights[0])
    predictions2 = (1.0 / weights[0]) * res[0]
    for i in range(1, len(res)):
        predictions2 += (1.0 / weights[i]) * res[i]
        total += (1.0 / weights[i])
    predictions2 = predictions2 / total
    print(predictions2.shape)

    # AGCRN * 0.4 + MTGNN * 0.6
    predictions = predictions1 * 0.4 + predictions2 * 0.6
    return predictions
