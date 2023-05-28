import torch
import numpy as np
import pandas as pd
import pickle
import os
import networkx as nx
from pyproj import Transformer
from sklearn.metrics import precision_score, recall_score, f1_score

from constants import *
from dataloader import DataLoader


def dataset_collate(trips):
    trips_collate = []
    for trip in trips:
        trip_collate = []
        trip = trip.split(';')
        for loc in trip:
            idx, lon, lat, time = loc.split(',')
            trip_collate.append([int(idx), float(lon), float(lat), int(time)])
        trips_collate.append(trip_collate)
    return trips_collate


def load_dataset(args, data_format):
    data_path = os.path.join('../data', 'T-drive')
    if data_format is 'csv':
        adj_path = os.path.join(data_path, 'graph_A.csv')
        train_path = os.path.join(data_path, 'traj_train.csv')
        val_path = os.path.join(data_path, "traj_val.csv")
        test_path = os.path.join(data_path, 'traj_test.csv')

        lbs_train = pd.read_csv(train_path)
        lbs_val = pd.read_csv(val_path, converters={'trips_sparse': eval, 'num_labels': eval})
        lbs_test = pd.read_csv(test_path, converters={'trips_sparse': eval, 'num_labels': eval})
        id2loc = pickle.load(open(os.path.join(data_path, 'grid2center_Beijing.pickle'), 'rb'))
        print("train data size {}, val data size {}, test data size {}, cell tower num {}"\
              .format(len(lbs_train), len(lbs_val), len(lbs_test), len(id2loc)))

    coor_transformer = Transformer.from_crs("epsg:4326", "epsg:4575", always_xy=True)

    def data_to_input(trips):
        trips_input = []
        for trip in trips:
            res = []
            time_min = trip[0][-1]
            for (loc, lon, lat, time) in trip:
                coords = id2loc[loc]
                res.append((int(loc)+TOTAL_SPE_TOKEN, time-time_min, coords[0], coords[1]))
            trips_input.append(res)
        return trips_input


    loc_size = len(id2loc)
    # id2loc = {id: to3414.transform(loc[0], loc[1]) for loc,id in loc2id.items()}

    adj_pd = pd.read_csv(adj_path)
    adj_pd = adj_pd.add({'src': TOTAL_SPE_TOKEN, 'dst': TOTAL_SPE_TOKEN, 'weight': 0})
    G = nx.DiGraph()
    G.add_nodes_from(list(range(loc_size + TOTAL_SPE_TOKEN)))
    src, dst, weights = adj_pd['src'].values.tolist(), adj_pd['dst'].values.tolist(), adj_pd['weight'].values.tolist()
    G.add_weighted_edges_from(zip(src, dst, weights))
    adj_graph = nx.to_numpy_array(G)

    train_traj = dataset_collate(lbs_train['trips_new'].values.tolist())
    val_traj = lbs_val['trips_sparse'].values.tolist()
    val_tgt = lbs_val['num_labels'].values.tolist()
    test_traj = lbs_test['trips_sparse'].values.tolist()
    test_tgt = lbs_test['num_labels'].values.tolist()


    max_len = 0
    for i in train_traj:
        length = len(i)
        if length > max_len:
            max_len = length

    print("train num {}, val num {}, test num {}, target {}, " \
          .format(len(train_traj), len(val_traj), len(test_traj), len(test_tgt)))


    train_input = data_to_input(train_traj)
    val_input = data_to_input(val_traj)
    val_target = val_tgt
    test_input = data_to_input(test_traj)
    test_target = test_tgt

    return train_input, val_input, val_target, test_input, test_target, loc_size, id2loc, max_len, adj_graph

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')

def get_dataloader(data, batch_size, max_seq_len, drop_num, ratio):
    dataloader = DataLoader(data, batch_size, max_seq_len, drop_num, ratio)

    return dataloader


def loss_func(pred, true, mask, func):
    mask = mask.long().view(-1)
    loss_ = func(pred, true) * mask

    return loss_.mean()


def pad_array(a, max_length, max_time=5000, PAD=0):
    """
    a (array[int32])
    """
    if len(a.shape) == 2: ## input seq (loc id, timestamp)
        arr_np = np.array([(PAD, max_time)] * (max_length - len(a)))
        if len(arr_np) != 0:
            res = np.concatenate((a, arr_np))
        else:
            res = a
    elif len(a.shape) == 1: ## label seq (0 or 1 for tagging)
        arr_np = np.array([PAD] * (max_length - len(a)))
        res = np.concatenate((a, arr_np))
    # print(a.shape, arr_np.shape)
    # print(a, arr_np)

    return res


def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(np.array(a[i]), max_length) for i in range(len(a))]
    a = np.stack(a)
    # print(a.shape, a)
    return torch.LongTensor(a)

def get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id):
    batch_size = src_token_ids_batch.shape[0]

    # src_mask shape = (B, 1, 1, S) check out attention function in transformer_model.py where masks are applied
    # src_mask only masks pad tokens as we want to ignore their representations (no information in there...)
    src_mask = (src_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)
    num_src_tokens = torch.sum(src_mask.long())

    return src_mask, num_src_tokens


def get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id):
    batch_size = trg_token_ids_batch.shape[0]
    device = trg_token_ids_batch.device

    # Same as src_mask but we additionally want to mask tokens from looking forward into the future tokens
    # Note: wherever the mask value is true we want to attend to that token, otherwise we mask (ignore) it.
    sequence_length = trg_token_ids_batch.shape[1]  # trg_token_ids shape = (B, T) where T max trg token-sequence length
    trg_padding_mask = (trg_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)  # shape = (B, 1, 1, T)
    trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device) == 1).transpose(2, 3)

    # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain target token)
    trg_mask = trg_padding_mask & trg_no_look_forward_mask  # final shape = (B, 1, T, T)
    num_trg_tokens = torch.sum(trg_padding_mask.long())

    return trg_mask, num_trg_tokens


def get_masks_and_count_tokens(src_token_ids_batch, pad_token_id):
    src_mask, num_src_tokens = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
    # trg_mask, num_trg_tokens = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)

    return src_mask, num_src_tokens


def evaluation(preds, targets, lengths):
    label_arr, pred_arr = [], []

    for pred, label, length in zip(preds, targets, lengths):

        pred, label = pred[:length], label[:length]
        label_arr.extend(label)
        pred_arr.extend(pred)

    assert len(label_arr) == len(pred_arr)

    pre = precision_score(label_arr, pred_arr)
    rec = recall_score(label_arr, pred_arr)
    f1 = f1_score(label_arr, pred_arr)

    print("average precision {}, average recall {}, average f1 {}". \
          format(pre, rec, f1))
    return pre, rec, f1


def evaluation_multiclass(preds, targets, lengths):
    label_arr, pred_arr = [], []

    for pred, label, length in zip(preds, targets, lengths):
        pred, label = pred[:length], label[:length]
        label_arr.extend(label)
        pred_arr.extend(pred)

    assert len(label_arr) == len(pred_arr)

    pre_ = precision_score(label_arr, pred_arr, average=None)
    rec_ = recall_score(label_arr, pred_arr, average=None)

    print("precision for each class {}".format(pre_))
    print("recall for each class {}".format(rec_))

    pre = precision_score(label_arr, pred_arr, average='micro')
    pre_weighted = precision_score(label_arr, pred_arr, average='weighted')
    rec = recall_score(label_arr, pred_arr, average='micro')
    rec_weighted = recall_score(label_arr, pred_arr, average='weighted')
    f1_micro = f1_score(label_arr, pred_arr, average='micro')
    f1_macro = f1_score(label_arr, pred_arr, average='macro')

    print("average precision {:.4f}, weighted {:.4f};  average recall {:.4f}, weighted {:.4f}; average f1-micro {:.4f}, average f1-macro {:.4f}". \
          format(pre, pre_weighted, rec, rec_weighted, f1_micro, f1_macro))
    return pre, rec, f1_micro, f1_macro



def validation(dataset, model, A, device):
    preds, labels, lengths = [], [], []
    for i, batch_data in enumerate(dataset):
        with torch.no_grad():
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_enc_loc, batch_enc_time, batch_enc_coor, batch_lengths, batch_target = batch_data

            src_mask, _ = get_masks_and_count_tokens_src(batch_enc_loc, PAD_TOKEN)

            outputs = model(batch_enc_loc, batch_enc_time, batch_enc_coor, src_mask, A, 'tagging')

            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            label = batch_target.cpu().numpy()
            length = batch_lengths.cpu().numpy()

            preds.extend(pred)
            labels.extend(label)
            lengths.extend(length)

    return preds, labels, lengths