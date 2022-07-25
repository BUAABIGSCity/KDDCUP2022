import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from logging import getLogger
from tqdm import tqdm
from wpf_dataset import time_dict
# from fastdtw import fastdtw


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class PGL4WPFDataset():

    def __init__(
            self,
            data_path,
            filename='wtbdata_245days.csv',
            size=None,
            capacity=134,
            day_len=24 * 6,
            total_days=245,
            random=False,
            only_useful=False,
            K=5,
            ind=4,
            num_workers=0,
            batch_size=32,
            pad_with_last_sample=False,
            graph_type='sem',
            weight_adj_epsilon=0.8,
            dtw_topk=5,
            binary=True,
    ):

        super().__init__()
        self.unit_size = day_len
        self.random = random
        self.only_useful = only_useful
        self.dtw_topk = dtw_topk
        self.K = K
        self.ind = ind
        self.binary = binary
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pad_with_last_sample = pad_with_last_sample
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]

        self.start_col = 0
        self.capacity = capacity

        self.data_path = data_path
        self.filename = filename
        self._logger = getLogger()

        self.graph_type = graph_type
        self.weight_adj_epsilon = weight_adj_epsilon

        self.total_size = self.unit_size * total_days
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))  # (t * n, f)
        df_data, raw_df_data = self.data_preprocess(df_raw)  # (t * n, f)

        self.df_data = df_data  # nan->0
        self.raw_df_data = raw_df_data  # contain nan

        x, y, data, raw_data = self.generate_input_data(self.df_data)
        x_train, y_train, x_val, y_val = self.split_train_val_test(x, y)

        self.build_scale(x_train)
        graph = self.build_graph_data(x_train)

        self._logger.info("x_train, y_train, x_val, y_val: {}, {}, {}, {}".format(x_train.shape,
                          y_train.shape, x_val.shape, y_val.shape))
        self._logger.info(f"graph: {graph}")
        self.graph = graph
        self.train_dataloader, self.eval_dataloader = self.gene_dataloader(x_train, y_train, x_val, y_val)
        self._logger.info("train / val: {}, {}".format(len(self.train_dataloader), len(self.eval_dataloader)))

    def gene_dataloader(self, x_train, y_train, x_val, y_val):
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        print('pad before', len(train_data), len(eval_data))

        if self.pad_with_last_sample:
            num_padding = (self.batch_size - (len(train_data) % self.batch_size)) % self.batch_size
            data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
            train_data = np.concatenate([train_data, data_padding], axis=0)
            num_padding = (self.batch_size - (len(eval_data) % self.batch_size)) % self.batch_size
            data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
            eval_data = np.concatenate([eval_data, data_padding], axis=0)
        print('pad', len(train_data), len(eval_data))

        train_dataset = ListDataset(train_data)
        eval_dataset = ListDataset(eval_data)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, drop_last=True,
                                      shuffle=True)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, drop_last=False,
                                     shuffle=False)
        return train_dataloader, eval_dataloader

    def data_preprocess(self, df_data):

        feature_name = [
            n for n in df_data.columns
            if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
            'TurbID' not in n
        ]
        # Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
        feature_name.append("Patv")

        new_df_data = df_data[feature_name]

        self._logger.info('adding time')
        t = df_data['Tmstamp'].apply(lambda x: time_dict[x])  # 计算这是第几个10min
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)  # 计算这是第几个7天
        new_df_data.insert(0, 'weekday', weekday)
        self._logger.info('adding time finish')

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        new_df_data = new_df_data.replace(
            to_replace=np.nan, value=0, inplace=False)

        return new_df_data, raw_df_data

    def generate_input_data(self, df_data):
        """

        Args:
            df_data(np.ndarray): shape: (len_time * 134, feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): (size, input_length, 134, feature_dim)
                y(np.ndarray): (size, output_length, 134, feature_dim)
        """
        cols_data = df_data.columns
        df_data = df_data[cols_data]
        raw_cols_data = self.raw_df_data.columns
        raw_df_data = self.raw_df_data

        data = df_data.values.astype('float32')  # (n=245*144*134, f)
        data = np.reshape(data, [self.capacity, self.total_size, len(cols_data)])  # (134, t, f), n = 134t
        # data = np.swapaxes(data, 0, 1)  # (t, 134, f)
        raw_data = raw_df_data.values.astype('float32')  # (n, f)
        raw_data = np.reshape(raw_data, [self.capacity, self.total_size, len(raw_cols_data)])  # (134, t, f), n = 134t

        num_samples = data.shape[1]  # t-dim
        # The length of the past time window for the prediction, depends on self.input_length
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_len + 1, 1, 1),)))
        # The length of future time window, depends on self.output_length
        y_offsets = np.sort(np.arange(1, self.output_len + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))  # input_len - 1
        max_t = abs(num_samples - abs(max(y_offsets)))  # n - output_len
        for t in tqdm(range(min_t, max_t), desc='split data'):  # total = max_t - min_t = n - output_len - input_len + 1
            x_t = data[:, t + x_offsets, :]
            y_t = data[:, t + y_offsets, :]
            x.append(x_t)  # (134, input_len, f)
            y.append(y_t)  # (134, output_len, f)
        print(len(x), x[0].shape)
        print(len(y), y[0].shape)
        # x = np.stack(x, axis=0)  # (max_t - min_t, 134, input_len, f)
        # y = np.stack(y, axis=0)  # (max_t - min_t, 134, output_len, f)
        return x, y, data, raw_data

    def split_train_val_test(self, x, y):
        """

        Args:
            x(np.ndarray): 输入数据 (num_samples, 134, input_len, feature_dim)
            y(np.ndarray): 输出数据 (num_samples, 134, output_len, feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, 134, input_len, feature_dim)
                y_train: (num_samples, 134, output_len, feature_dim)
                x_val: (num_samples, 134, input_len, feature_dim)
                y_val: (num_samples, 134, output_len, feature_dim)
        """
        unit_x_y_size = len(x) // self.K  # K次均分
        board = [0]
        for i in range(1, self.K):
            board.append(board[-1] + unit_x_y_size)
        board.append(len(x))
        print(board)

        # val
        x_val, y_val = x[board[self.ind]: board[self.ind+1]], y[board[self.ind]: board[self.ind+1]]
        print('val', board[self.ind], ':', board[self.ind+1])
        # train
        x_train, y_train = [], []
        for i in range(self.K):
            if i == self.ind:
                continue
            print('train', board[i], ':', board[i + 1])
            x_i = x[board[i]: board[i + 1]]
            y_i = y[board[i]: board[i + 1]]
            x_train += x_i
            y_train += y_i
        print(len(x_train), len(y_train), len(x_val), len(y_val))
        x_train = np.array(x_train)  # (b, n, t, f)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
        return x_train, y_train, x_val, y_val

    def build_scale(self, x_train):
        # x_train: (b, 134, t, f)
        self.data_mean = np.mean(
                    x_train[:, :, :, 2:],  # time & weekday 去掉了
                    axis=(0, 2, 3),
                    keepdims=True)  # (1, 134, 1, 1)
        self.data_scale = np.std(
                    x_train[:, :, :, 2:],  # time & weekday 去掉了
                    axis=(0, 2, 3),
                    keepdims=True)  # (1, 134, 1, 1)
        # np.save("npy/data_mean_{}_{}.npy".format(self.ind, self.K), self.data_mean)
        # np.save("npy/data_scale_{}_{}.npy".format(self.ind, self.K), self.data_scale)
        print('mean, scale, {}, {}'.format(self.data_mean.shape, self.data_scale.shape))

    def get_raw_df(self):
        return [self.raw_df_data]

    def build_graph_data(self, train_data):
        # x_train: (b, 134, t, f)
        origin_train_data = []
        for i in range(train_data.shape[0] - 1):  # Each data takes the result of the first step
            origin_train_data.append(train_data[i, :, 0, :])  # (134, f)
        for i in range(train_data[-1].shape[1]):  # The last data takes the result of all time steps
            origin_train_data.append(train_data[-1, :, i, :])  # (134, f)
        print(len(origin_train_data), origin_train_data[0].shape)
        origin_train_data = np.stack(origin_train_data)
        print('origin_train_data', origin_train_data.shape)  # (t, 134, f)

        if self.graph_type == "geo":
            graph = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "npy/geo_graph.npy"))
            distances = graph.flatten()
            dist_std = distances.std()
            graph = np.exp(-np.square(graph / dist_std))
            graph[graph < self.weight_adj_epsilon] = 0
            if self.binary:
                graph[graph >= self.weight_adj_epsilon] = 1
            self._logger.info(f"geo graph links: {graph.sum()}")
        elif self.graph_type == 'dtw':
            # df = origin_train_data[:, :, -1]  # (t, 134) 训练集的Patv
            # df = np.swapaxes(df, 0, 1)  # (134, t) 训练集的Patv
            # data_mean = np.mean(
            #     [df[:, self.unit_size * i: self.unit_size * (i + 1)]
            #      for i in range(df.shape[1] // self.unit_size)], axis=0)  # (134, 144)
            # dtw_distance = np.zeros((self.capacity, self.capacity))
            # for i in tqdm(range(self.capacity)):
            #     for j in range(i, self.capacity):
            #         dtw_distance[i][j], _ = fastdtw(data_mean[i, :], data_mean[j, :], radius=6)
            # for i in range(self.capacity):
            #     for j in range(i):
            #         dtw_distance[i][j] = dtw_distance[j][i]
            # np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
            #                      "npy/dtw_graph_{}_{}.npy".format(self.ind, self.K)), dtw_distance)
            dtw_distance = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                "npy/dtw_graph_{}_{}.npy".format(self.ind, self.K)))
            ind = np.argsort(dtw_distance)[:, 0:self.dtw_topk]  # (n, k)
            graph = np.zeros((self.capacity, self.capacity))
            for i in range(ind.shape[0]):
                for j in range(ind.shape[1]):
                    graph[i][ind[i][j]] = 1
                    graph[ind[i][j]][i] = 1
            # np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
            #                      "npy/dtw_graph_top{}_{}_{}.npy".format(self.dtw_topk, self.ind, self.K)), graph)
            graph = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "npy/dtw_graph_top{}_{}_{}.npy".format(self.dtw_topk, self.ind, self.K)))
            self._logger.info(f"dtw graph links: {graph.sum()}")
        else:
            raise ValueError('Error graph_type = {}'.format(self.graph_type))
        return graph
