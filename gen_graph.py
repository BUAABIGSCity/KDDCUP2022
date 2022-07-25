import os
import math
import numpy as np
import pandas as pd


if __name__ == "__main__":
    data_path = "./data"
    geo_file = "sdwpf_baidukddcup2022_turb_location.CSV"
    geo = pd.read_csv(os.path.join(data_path, geo_file))
    coord_list = []
    for row in geo.values:
        coord_list.append((row[1], row[2]))
    num_nodes = len(coord_list)
    graph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            dist = math.sqrt((coord_list[i][0] - coord_list[j][0]) ** 2 + (coord_list[i][1] - coord_list[j][1]) ** 2)
            graph[i][j] = dist
            graph[j][i] = dist
    np.save("npy/geo_graph.npy", graph)