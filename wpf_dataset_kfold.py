import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from logging import getLogger
from tqdm import tqdm
# from fastdtw import fastdtw


time_dict = {
 '00:00': 0, '00:10': 1, '00:20': 2, '00:30': 3, '00:40': 4, '00:50': 5, '01:00': 6, '01:10': 7, '01:20': 8, '01:30': 9,
 '01:40': 10, '01:50': 11, '02:00': 12, '02:10': 13, '02:20': 14, '02:30': 15, '02:40': 16, '02:50': 17, '03:00': 18,
 '03:10': 19, '03:20': 20, '03:30': 21, '03:40': 22, '03:50': 23, '04:00': 24, '04:10': 25, '04:20': 26, '04:30': 27,
 '04:40': 28, '04:50': 29, '05:00': 30, '05:10': 31, '05:20': 32, '05:30': 33, '05:40': 34, '05:50': 35, '06:00': 36,
 '06:10': 37, '06:20': 38, '06:30': 39, '06:40': 40, '06:50': 41, '07:00': 42, '07:10': 43, '07:20': 44, '07:30': 45,
 '07:40': 46, '07:50': 47, '08:00': 48, '08:10': 49, '08:20': 50, '08:30': 51, '08:40': 52, '08:50': 53, '09:00': 54,
 '09:10': 55, '09:20': 56, '09:30': 57, '09:40': 58, '09:50': 59, '10:00': 60, '10:10': 61, '10:20': 62, '10:30': 63,
 '10:40': 64, '10:50': 65, '11:00': 66, '11:10': 67, '11:20': 68, '11:30': 69, '11:40': 70, '11:50': 71, '12:00': 72,
 '12:10': 73, '12:20': 74, '12:30': 75, '12:40': 76, '12:50': 77, '13:00': 78, '13:10': 79, '13:20': 80, '13:30': 81,
 '13:40': 82, '13:50': 83, '14:00': 84, '14:10': 85, '14:20': 86, '14:30': 87, '14:40': 88, '14:50': 89, '15:00': 90,
 '15:10': 91, '15:20': 92, '15:30': 93, '15:40': 94, '15:50': 95, '16:00': 96, '16:10': 97, '16:20': 98, '16:30': 99,
 '16:40': 100, '16:50': 101, '17:00': 102, '17:10': 103, '17:20': 104, '17:30': 105, '17:40': 106, '17:50': 107,
 '18:00': 108, '18:10': 109, '18:20': 110, '18:30': 111, '18:40': 112, '18:50': 113, '19:00': 114, '19:10': 115,
 '19:20': 116, '19:30': 117, '19:40': 118, '19:50': 119, '20:00': 120, '20:10': 121, '20:20': 122, '20:30': 123,
 '20:40': 124, '20:50': 125, '21:00': 126, '21:10': 127, '21:20': 128, '21:30': 129, '21:40': 130, '21:50': 131,
 '22:00': 132, '22:10': 133, '22:20': 134, '22:30': 135, '22:40': 136, '22:50': 137, '23:00': 138, '23:10': 139,
 '23:20': 140, '23:30': 141, '23:40': 142, '23:50': 143}


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
        self.raw_df_data = raw_df_data  # 可能有nan

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
            df(np.ndarray): 数据数组，shape: (len_time * 134, feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(size, input_length, 134, feature_dim)
                y(np.ndarray): 模型输出数据，(size, output_length, 134, feature_dim)
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
        # 预测用的过去时间窗口长度 取决于self.input_length
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_len + 1, 1, 1),)))
        # 未来时间窗口长度 取决于self.output_length
        y_offsets = np.sort(np.arange(1, self.output_len + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))  # input_len - 1
        max_t = abs(num_samples - abs(max(y_offsets)))  # n - output_len
        for t in tqdm(range(min_t, max_t), desc='split data'):  # 总数 = max_t - min_t = n - output_len - input_len + 1
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
        划分训练集、测试集、验证集，并缓存数据集

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
