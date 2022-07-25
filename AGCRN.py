import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger

# 跟AGCRN比，加入预定义的图（graph_type = {geo, dtw, sem, geo_dtw, geo_sem}）
# 使用参数add_apt控制是否使用可学习的图


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


class VWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, dropout, add_apt):
        super(VWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.dropout_rate = dropout
        self.add_apt = add_apt
        if self.dropout_rate != 0:
            self.dropout = nn.Dropout(p=dropout)
        if self.add_apt:
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k + 1, dim_in, dim_out))
        else:
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings, lap):
        """
        GCN

        Args:
            x(torch.tensor): (B, N, C)
            node_embeddings(torch.tensor): (N, D)

        Returns:
            torch.tensor: (B, N, output_dim)
        """
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        if self.dropout_rate != 0:
            node_embeddings = self.dropout(node_embeddings)
        support_set = [torch.eye(node_num).to(x.device), lap]
        # default cheb_k = 3 迭代计算邻接矩阵的各阶切比雪夫多项式Tk的结果
        # Tk(L) = 2 * L * Tk-1(L) - Tk-2(L)
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * lap, support_set[-1]) - support_set[-2])
        if self.add_apt:
            supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
            support_set.append(supports)
        supports = torch.stack(support_set, dim=0)  # (k, n, n)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                         # N, dim_out
        # supports = (k, n, n), x = (B, n, dim_in) --> (b, k, n, dim_in)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, dropout, add_apt):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.graph_gate = VWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim, dropout, add_apt)
        self.graph_update = VWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim, dropout, add_apt)

    def forward(self, x, state, node_embeddings, graph_lap=None):
        """
        变型的GRU

        Args:
            x(torch.tensor): (B, num_nodes, input_dim)
            state(torch.tensor): (B, num_nodes, hidden_dim)
            node_embeddings(torch.tensor): (num_nodes, D)

        Returns:
            torch.tensor: (B, num_nodes, hidden_dim)

        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)  # B, num_nodes, input_dim + hidden_dim
        z_r = torch.sigmoid(self.graph_gate(input_and_state, node_embeddings, graph_lap))  # B, num_nodes, 2 * hidden_dim
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)  # B, num_nodes, hidden_dim
        candidate = torch.cat((x, z*state), dim=-1)  # B, num_nodes, input_dim + hidden_dim
        hc = torch.tanh(self.graph_update(candidate, node_embeddings, graph_lap))  # B, num_nodes, hidden_dim
        h = r*state + (1-r)*hc  # B, num_nodes, hidden_dim
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, config):
        super(AVWDCRNN, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_dim = config['feature_dim']
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        self.dropout = config.get('dropout', 0)
        self.add_apt = config.get('add_apt', True)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.feature_dim,
                                          self.hidden_dim, self.cheb_k, self.embed_dim, self.dropout, self.add_apt))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim, self.dropout, self.add_apt))

    def forward(self, x, init_state, node_embeddings, graph_lap=None):
        """
        多层GRU

        Args:
            x(torch.tensor): (B, T, N, D)
            init_state(torch.tensor): (num_layers, B, N, hidden_dim)
            node_embeddings(torch.tensor): (N, D)

        Returns:
            tuple: tuple contains:
                current_inputs: the outputs of last layer, (B, T, N, hidden_dim) \n
                output_hidden: the last state for each layer, (num_layers, B, N, hidden_dim)
        """
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.feature_dim
        seq_length = x.shape[1]  # input_window
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):  # 多层GRU
            state = init_state[i]  # (B, N, hidden_dim)
            inner_states = []
            for t in range(seq_length):  # 多个时间步 输入GRU循环
                # (B, N, D) + (B, N, hidden_dim) + (N, D)  -->  (B, N, hidden_dim)
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, graph_lap)
                inner_states.append(state)  # (B, N, hidden_dim)
            output_hidden.append(state)  # the last state
            current_inputs = torch.stack(inner_states, dim=1)  # (B, T, N, hidden_dim)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, config, adj_mx=None):
        super().__init__()

        self.adj_mx = adj_mx
        self.feature_dim = config["var_len"]
        self.input_window = config["input_len"]
        self.output_window = config["output_len"]
        self.num_nodes = config["capacity"]
        self.data_diff = config["data_diff"]
        self.output_dim = 1
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim

        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.time_dim = config.get('time_dim', 0)
        self.dropout_rate = config.get('dropout', 0)
        self.device = config.get('device', torch.device('cpu'))
        if self.time_dim != 0:
            self.time_emb = nn.Embedding(144, self.time_dim)
            config['feature_dim'] = self.feature_dim + self.time_dim

        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(config)
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        if self.dropout_rate != 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self._logger = getLogger()
        self._init_parameters()

        self.graph_lap = torch.tensor(calculate_normalized_laplacian(self.adj_mx).astype(
            np.float32).todense()).to(self.device)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch_x, batch_y, data_mean, data_scale):
        inputs = batch_x  # (batch_size, num_nodes, input_window, feature_dim)
        bz, id_len, input_len, var_len = inputs.shape

        inputs = inputs[:, :, :, 2:]
        inputs = (inputs - data_mean) / data_scale

        if self.data_diff:
            inputs_diff = inputs[:, :, 1:, -1:] - inputs[:, :, :-1, -1:]
            inputs_diff = torch.cat((torch.zeros(bz, id_len, 1, 1).to(inputs.device), inputs_diff), 2)
            inputs = torch.cat((inputs, inputs_diff), 3)

        if self.time_dim != 0:
            time_emb = self.time_emb(batch_x[:, :, :, 1].long())  # (batch_size, num_nodes, input_window, hid_dim)
            inputs = torch.cat([inputs, time_emb], axis=3)  # (batch_size, num_nodes, input_window, hid_dim+feature_dim)

        inputs = inputs.permute((0, 2, 1, 3))  # (batch_size, input_window, num_nodes, feature_dim)

        init_state = self.encoder.init_hidden(inputs.shape[0])
        # graph_lap = torch.tensor(calculate_normalized_laplacian(graph).astype(np.float32).todense()).to(inputs.device)
        output, _ = self.encoder(inputs, init_state, self.node_embeddings, self.graph_lap)  # B, T, N, hidden
        output = output[:, -1:, :, :]                                       # B, 1, N, hidden
        #  (B, 1, N, hidden_dim)
        if self.dropout_rate != 0:
            output = self.dropout(output)

        # CNN based predictor, kernel = (1, hidden_dim)
        output = self.end_conv(output)                           # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)                      # B, T, N, C
        return output[..., 0].permute((0, 2, 1))  # (B,N,T)