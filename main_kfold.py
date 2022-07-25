import argparse
import yaml
import numpy as np
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import random
import loss as loss_factory
from wpf_dataset_kfold import PGL4WPFDataset
from MTGNN import MTGNN
from AGCRN import AGCRN
from utils import save_model, _create_if_not_exist, get_logger, str2bool, ensure_dir
from logging import getLogger
from tqdm import tqdm
from main import set_seed, data_augment, build_optimizer, build_lr_scheduler


def train_and_evaluate(config, dataset):
    # weekday,time,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
    # 0       1    2    3    4    5    6    7    8    9    10   11
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

    log = getLogger()

    data_mean = torch.FloatTensor(dataset.data_mean).to(config.device)  # (1, 134, 1, 1)
    data_scale = torch.FloatTensor(dataset.data_scale).to(config.device)  # (1, 134, 1, 1)

    graph = dataset.graph  # (134, 134)

    train_data_loader = dataset.train_dataloader
    valid_data_loader = dataset.eval_dataloader

    if config.model == 'MTGNN':
        model = MTGNN(config=config, adj_mx=graph).to(config.device)
    elif config.model == 'AGCRN':
        model = AGCRN(config=config, adj_mx=graph).to(config.device)
    else:
        raise ValueError('Error config.model = {}'.format(config.model))

    log.info(model)
    for name, param in model.named_parameters():
        log.info(str(name) + '\t' + str(param.shape) + '\t' +
                          str(param.device) + '\t' + str(param.requires_grad))
    total_num = sum([param.nelement() for param in model.parameters()])
    log.info('Total parameter numbers: {}'.format(total_num))

    loss_fn = getattr(loss_factory, config.loss)()

    opt = build_optimizer(config, log, model)
    grad_accmu_steps = config.gsteps
    opt.zero_grad()
    lr_scheduler = build_lr_scheduler(config, log, opt)
    _create_if_not_exist(config.output_path)
    global_step = 0

    best_score = np.inf
    patient = 0

    col_names = dict(
        [(v, k) for k, v in enumerate(dataset.get_raw_df()[0].columns)])

    valid_records = []

    for epoch in range(config.epoch):
        model.train()
        losses = []
        for batch_x, batch_y in tqdm(train_data_loader, 'train'):
            if config.enhance:
                batch_x, batch_y = data_augment(batch_x, batch_y)

            # weekday,time,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
            # 0       1    2    3    4    5    6    7    8    9    10   11
            if config.only_useful:
                batch_x = batch_x[:, :, :, select_ind]
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            input_y = batch_y  # (B,N,T,F)
            batch_y = batch_y[:, :, :, -1]  # (B,N,T)
            batch_y = (batch_y - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]

            pred_y = model(batch_x, input_y, data_mean, data_scale)  # (B,N,T)
            loss = loss_fn(pred_y, batch_y, input_y, col_names)
            loss = loss / grad_accmu_steps
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            if global_step % grad_accmu_steps == 0:
                opt.step()
                opt.zero_grad()
            global_step += 1
            losses.append(loss.item())
            if global_step % config.log_per_steps == 0:
                log.info("Step %s Train Loss: %s" % (global_step, loss.item()))
        log.info("Epoch=%s, exp_id=%s, Train Loss: %s" % (epoch, config.exp_id, np.mean(losses)))

        valid_r = evaluate(
                valid_data_loader,
                dataset.get_raw_df(),
                model,
                loss_fn,
                config,
                data_mean,
                data_scale,
                tag="val",
                select_ind=select_ind)
        valid_records.append(valid_r)

        log.info("Epoch={}, exp_id={}, Valid ".format(epoch, config.exp_id) + str(dict(valid_r)))

        if lr_scheduler is not None:
            if config.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler.step(valid_r['loss'])
            else:
                lr_scheduler.step()

        best_score = min(valid_r['loss'], best_score)

        if best_score == valid_r['loss']:
            patient = 0
            save_model(config.output_path+config.exp_id+'_'+config.model, model, opt=opt, steps=epoch, log=log)
        else:
            patient += 1
            if patient > config.patient:
                break

    best_epochs = min(enumerate(valid_records), key=lambda x: x[1]["loss"])[0]
    log.info("Best valid Epoch %s" % best_epochs)
    log.info("Best valid score %s" % valid_records[best_epochs])


def evaluate(valid_data_loader,
             valid_raw_df,
             model,
             loss_fn,
             config,
             data_mean,
             data_scale,
             tag="train",
             select_ind=None):
    with torch.no_grad():
        col_names = dict([(v, k) for k, v in enumerate(valid_raw_df[0].columns)])
        model.eval()
        step = 0
        pred_batch = []
        gold_batch = []
        input_batch = []
        losses = []
        for batch_x, batch_y in tqdm(valid_data_loader, tag):
            # weekday,time,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,Patv
            # 0       1    2    3    4    5    6    7    8    9    10   11
            if config.only_useful:
                batch_x = batch_x[:, :, :, select_ind]
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            pred_y = model(batch_x, batch_y, data_mean, data_scale)

            scaled_batch_y = batch_y[:, :, :, -1]  # (B,N,T)
            scaled_batch_y = (scaled_batch_y - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1]
            loss = loss_fn(pred_y, scaled_batch_y, batch_y, col_names)
            losses.append(loss.item())

            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
            pred_y = pred_y.cpu().numpy()  # (B,N,T)

            batch_y = batch_y[:, :, :, -1].cpu().numpy()  # (B,N,T)
            input_batch.append(batch_x[:, :, :, -1].cpu().numpy())  # (B,N,T)
            pred_batch.append(pred_y)
            gold_batch.append(batch_y)

            step += 1
        model.train()

        output_metric = {
            'loss': np.mean(losses),
        }

        return output_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--model", type=str, default="AGCRN")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--input_len", type=int, default=144, help='input data len')
    parser.add_argument("--output_len", type=int, default=288, help='output data len')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_days", type=int, default=214)
    parser.add_argument("--val_days", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--K", type=int, default=5, help='K-fold')
    parser.add_argument("--ind", type=int, default=0, help='selected fold for validation set')
    parser.add_argument("--random", type=str2bool, default=False, help='Whether shuffle num_nodes')
    parser.add_argument("--enhance", type=str2bool, default=True, help='Whether enhance the time dim')
    parser.add_argument("--only_useful", type=str2bool, default=False, help='Whether remove some feature')
    parser.add_argument("--var_len", type=int, default=6, help='Dimensionality of input features')
    parser.add_argument("--data_diff", type=str2bool, default=True, help='Whether to use data differential features')
    parser.add_argument("--add_apt", type=str2bool, default=True, help='Whether to use adaptive matrix')
    parser.add_argument("--binary", type=str2bool, default=True, help='Whether to set the adjacency matrix as binary')
    parser.add_argument("--pad", type=str2bool, default=False, help='pad with last sample')
    parser.add_argument("--graph_type", type=str, default="dtw", help='graph type, dtw or geo')
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

    exp_id = config.get('exp_id', None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = str(exp_id)

    logger = get_logger(config)
    logger.info(config)
    set_seed(config.seed)
    ensure_dir(config.output_path)

    dataset = PGL4WPFDataset(
        data_path=config.data_path,
        filename=config.filename,
        size=[config.input_len, config.output_len],
        total_days=config.total_days,
        random=config.random,
        only_useful=config.only_useful,
        K=config.K,
        ind=config.ind,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pad_with_last_sample=config.pad,
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
    train_and_evaluate(config, dataset)
