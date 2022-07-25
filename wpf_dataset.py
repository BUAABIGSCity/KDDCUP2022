import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from logging import getLogger
# from fastdtw import fastdtw


# Timestamp Fast Mapping
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


class PGL4WPFDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """

    def __init__(
            self,
            data_path,
            filename='wtbdata_245days.csv',
            flag='train',
            size=None,
            capacity=134,
            day_len=24 * 6,
            train_days=214,  # 15 days
            val_days=16,  # 3 days
            test_days=15,  # 6 days
            total_days=245,  # 30 days
            theta=0.9,
            random=False,
            only_useful=False,
            graph_type='sem',
            weight_adj_epsilon=0.8,
            dtw_topk=5,
            binary=True,
        ):

        super().__init__()
        self.unit_size = day_len
        self.train_days = train_days
        self.points_per_hour = day_len // 24
        self.random = random
        self.only_useful = only_useful
        self.dtw_topk = dtw_topk
        self.binary = binary
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]

        self.start_col = 0
        self.capacity = capacity
        self.theta = theta

        # initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.data_path = data_path
        self.filename = filename
        self.graph_type = graph_type
        self.weight_adj_epsilon = weight_adj_epsilon
        self._logger = getLogger()

        self.total_size = self.unit_size * total_days
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.test_size = test_days * self.unit_size
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        df_data, raw_df_data = self.data_preprocess(df_raw)
        print(df_data.shape, raw_df_data.shape)
        self.df_data = df_data  # nan->0
        self.raw_df_data = raw_df_data  # contain nan

        data_x, graph = self.build_graph_data(df_data)
        self._logger.info(f"data_shape: {data_x.shape}")
        self._logger.info(f"graph: {graph}")
        self.data_x = data_x  # (134, t, f)
        self.graph = graph

    def __getitem__(self, index):
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[:, s_begin:s_end, :]
        seq_y = self.data_x[:, r_begin:r_end, :]

        if self.flag == "train":
            if self.random:
                perm = np.arange(0, seq_x.shape[0])
                np.random.shuffle(perm)
                return seq_x[perm].astype('float32'), seq_y[perm].astype('float32')
            else:
                return seq_x.astype('float32'), seq_y.astype('float32')
        else:
            return seq_x.astype('float32'), seq_y.astype('float32')

    def __len__(self):
        return self.data_x.shape[1] - self.input_len - self.output_len + 1

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
        t = df_data['Tmstamp'].apply(lambda x: time_dict[x])
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)
        new_df_data.insert(0, 'weekday', weekday)

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        new_df_data = new_df_data.replace(
            to_replace=np.nan, value=0, inplace=False)
        return new_df_data, raw_df_data

    def get_raw_df(self):
        return self.raw_df

    def build_graph_data(self, df_data):
        cols_data = df_data.columns
        df_data = df_data[cols_data]
        raw_cols_data = self.raw_df_data.columns
        raw_df_data = self.raw_df_data

        data = df_data.values  # (n, f)
        data = np.reshape(data,  # (134, t, f), n = 134t
                          [self.capacity, self.total_size, len(cols_data)])
        raw_data = raw_df_data.values  # (n, f)
        raw_data = np.reshape(  # (134, t, f), n = 134t
            raw_data, [self.capacity, self.total_size, len(raw_cols_data)])

        border1s = [
            0, self.train_size - self.input_len,
            self.train_size + self.val_size - self.input_len
        ]
        border2s = [
            self.train_size, self.train_size + self.val_size,
            self.train_size + self.val_size + self.test_size
        ]

        self.data_mean = np.expand_dims(  # (1, 134, 1, 1)
                np.mean(
                    data[:, border1s[0]:border2s[0], 2:],
                    axis=(1, 2),
                    keepdims=True),
                0)
        self.data_scale = np.expand_dims(  # (1, 134, 1, 1)
                np.std(data[:, border1s[0]:border2s[0], 2:],
                       axis=(1, 2),
                       keepdims=True),
                0)
        # np.save("data_mean_{}.npy".format(self.train_days), self.data_mean)
        # np.save("data_scale_{}.npy".format(self.train_days), self.data_scale)

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.raw_df = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(   # (134, t, f) --> (len, f)
                    data=raw_data[turb_id, border1 + self.input_len:border2],
                    columns=raw_cols_data))

        data_x = data[:, border1:border2, :]
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
            # df = data[:, border1s[0]:border2s[0], -1]  # (134, t) Patv
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
            # np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "npy/dtw_graph.npy"), dtw_distance)
            dtw_distance = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "npy/dtw_graph.npy"))
            ind = np.argsort(dtw_distance)[:, 0:self.dtw_topk]  # (n, k)
            graph = np.zeros((self.capacity, self.capacity))
            for i in range(ind.shape[0]):
                for j in range(ind.shape[1]):
                    graph[i][ind[i][j]] = 1
                    graph[ind[i][j]][i] = 1
            np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "dtw_graph_top{}.npy".format(self.dtw_topk)), graph)
            self._logger.info(f"dtw graph links: {graph.sum()}")
        else:
            raise ValueError('Error graph_type = {}'.format(self.graph_type))
        return data_x, graph


class TestPGL4WPFDataset(Dataset):
    """
    Desc: Data preprocessing,
    """

    def __init__(self, filename, capacity=134, day_len=24 * 6, only_useful=False):

        super().__init__()
        self.unit_size = day_len
        self.only_useful = only_useful
        self.start_col = 0
        self.capacity = capacity
        self.filename = filename
        self._logger = getLogger()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.filename)
        df_data, raw_df_data = self.data_preprocess(df_raw)
        self.df_data = df_data
        self.raw_df_data = raw_df_data

        data_x = self.build_graph_data(df_data)
        self.data_x = data_x

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
        t = df_data['Tmstamp'].apply(lambda x: time_dict[x])
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)
        new_df_data.insert(0, 'weekday', weekday)

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        new_df_data = new_df_data.replace(to_replace=np.nan, value=0)

        return new_df_data, raw_df_data

    def get_raw_df(self):
        return self.raw_df

    def build_graph_data(self, df_data):
        cols_data = df_data.columns
        df_data = df_data[cols_data]
        raw_cols_data = self.raw_df_data.columns
        raw_df_data = self.raw_df_data
        data = df_data.values
        raw_data = raw_df_data.values

        data = np.reshape(data, [self.capacity, -1, len(cols_data)])
        raw_data = np.reshape(raw_data, [self.capacity, -1, len(raw_cols_data)])

        data_x = data[:, :, :]

        self.raw_df = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_data[turb_id], columns=raw_cols_data))
        return np.expand_dims(data_x, [0])

    def get_data(self):
        return self.data_x
