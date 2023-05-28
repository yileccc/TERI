import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import os
import pickle
from pyproj import Transformer
from constants import *


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
        print("train data size {}, test data size {}, cell tower num {}".format(len(lbs_train), len(lbs_test),
                                                                                len(id2loc)))

    coor_transformer = Transformer.from_crs("epsg:4326", "epsg:4575", always_xy=True)

    def data_to_input(trips):
        trips_input = []
        for trip in trips:
            res = []
            time_min = trip[0][-1]
            for (loc, lon, lat, time) in trip:
                if loc == 'BLK':
                    res.append((BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT))
                else:
                    coords = id2loc[loc]
                    res.append((int(loc)+TOTAL_SPE_TOKEN, time-time_min, coords[0], coords[1]))
            trips_input.append(res)
        return trips_input

    loc_size = len(id2loc)
    adj_pd = pd.read_csv(adj_path)
    adj_pd = adj_pd.add({'src': TOTAL_SPE_TOKEN, 'dst': TOTAL_SPE_TOKEN, 'weight': 0})
    G = nx.DiGraph()
    G.add_nodes_from(list(range(loc_size+TOTAL_SPE_TOKEN)))
    src, dst, weights = adj_pd['src'].values.tolist(), adj_pd['dst'].values.tolist(), adj_pd['weight'].values.tolist()
    G.add_weighted_edges_from(zip(src, dst, weights))
    adj_graph = nx.to_numpy_array(G)


    train_traj = dataset_collate(lbs_train['trips_new'].values.tolist())
    val_traj, val_num_labels = lbs_val['trips_sparse'].values.tolist(), lbs_val['num_labels'].values.tolist()
    test_traj, test_num_labels = lbs_test['trips_sparse'].values.tolist(), lbs_test['num_labels'].values.tolist()
    test_tgt = dataset_collate(lbs_test['trips_new'].values.tolist())
    val_tgt = dataset_collate(lbs_val['trips_new'].values.tolist())


    max_len = 0
    for i in train_traj:
        length = len(i)
        if length > max_len:
            max_len = length
    print("train num {}, val num {}, test num {}, target {}" \
          .format(len(train_traj), len(val_traj), len(test_traj), len(test_tgt)))

    # + special tokens: PAD, BOS, EOS, NUL, BLK
    train_input = data_to_input(train_traj)
    val_input = data_to_input(val_traj)
    val_target = val_tgt
    test_input = data_to_input(test_traj)
    test_target = test_tgt

    return train_input, val_input, val_num_labels, val_target, test_input, test_num_labels, test_target, loc_size, id2loc, max_len, adj_graph



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


def loss_func(pred, true, func):
    mask = (true != 0).long()
    loss_ = func(pred, true) * mask

    return loss_.mean()


def pad_array(a, max_length, max_time=5000, PAD=0):
    """
    a (array[int32])
    """
    if len(a[0]) == 2: ## input seq (loc id, timestamp)
        arr_np = np.array([(PAD, max_time)] * (max_length - len(a)))
        if len(arr_np) != 0:
            res = np.concatenate((a, arr_np))
        else:
            res = a
    elif len(a[0]) == 1: ## label seq (0 or 1 for tagging)
        arr_np = np.array([PAD] * (max_length - len(a)))
        res = np.concatenate((a, arr_np))
    # print(a.shape, arr_np.shape)
    # print(a, arr_np)

    return res


def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a)
    #     print(a.shape, a)
    return torch.LongTensor(a)

def get_test_blk_indices(t):
    res = [i for i,x in enumerate(t) if x[0]==BLK_TOKEN]
    if len(res) == 0: res = [1]
    return res

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


def get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, pad_token_id):
    src_mask, num_src_tokens = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
    trg_mask, num_trg_tokens = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)

    return src_mask, trg_mask, num_src_tokens, num_trg_tokens



def project2D_enriched(updates, func):
    updates2D = []
    for update in updates:
        lon = update[0]
        lat = update[1]
        (x, y) = func.transform(np.float(lon), np.float(lat))
        updates2D.append([x, y])
    return (updates2D)


def validation(dataset, model, A, device, sample=False):
    preds = []
    for i, batch_data in enumerate(dataset):
        with torch.no_grad():
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_loc, batch_time, batch_coor, batch_lengths, batch_masked_pos, batch_masked_pos_lengths = batch_data

            lengths = batch_lengths.cpu().numpy()
            masked_pos = batch_masked_pos.cpu().numpy()
            masked_pos_lengths = batch_masked_pos_lengths.cpu().numpy()

            batch_pred_inputs = torch.tensor([BLK_TOKEN] * batch_loc.size(0), dtype=torch.long,
                                             device=device).unsqueeze(1)

            for idx in range(batch_masked_pos.shape[1]):
                attn_mask, _ = get_masks_and_count_tokens_trg(torch.cat([batch_loc, batch_pred_inputs], dim=1),
                                                              PAD_TOKEN)
                batch_masked_pos_cur = batch_masked_pos[:, :idx + 1]

                assert batch_pred_inputs.shape[1] == batch_masked_pos_cur.shape[1], "decoding step not correct"

                trg_probs = model(batch_loc, batch_time, batch_coor, attn_mask, A, 'recovery',
                                           batch_masked_pos_cur, batch_pred_inputs)

                last_words_batch = trg_probs[:, idx]  # B x vocab_size

                if sample:
                    pred_locs = torch.multinomial(last_words_batch, num_samples=1)
                else:
                    pred_locs = torch.argmax(last_words_batch, dim=-1)

                batch_pred_inputs = torch.cat([batch_pred_inputs, pred_locs.unsqueeze(1)], dim=1)

            output_pred_locs = batch_pred_inputs[:, 1:].cpu().numpy()  # remove the first blk token
            output_locs = batch_loc.cpu().numpy()
            batch_preds_post = []
            for idx, (pred, masked_p, length, masked_pos_length) in enumerate(
                    zip(output_pred_locs, masked_pos, lengths, masked_pos_lengths)):
                masked_p = masked_p[:masked_pos_length]
                output_locs[idx, masked_p] = pred[:masked_pos_length]
                batch_preds_post.append(output_locs[idx, :length])

            preds.extend(batch_preds_post)
    return preds



def evaluation(inputs, preds, truths, id2loc, maxlen):
    recall_total, precision_total, recovery_total, micro_precision_total = [], [], [], []

    for drop, pred, label in zip(inputs, preds, truths):
        label = [l[0] for l in label]
        # print("input {}, label {}".format(drop, label))

        pred = [p-TOTAL_SPE_TOKEN for p in pred if p>=TOTAL_SPE_TOKEN]

        recall = len(set(pred).intersection(set(label))) / len(label)
        precision = len(set(pred).intersection(set(label))) / len(pred)

        drop = [p[0]-TOTAL_SPE_TOKEN for p in drop if p[0]>=TOTAL_SPE_TOKEN]

        expected = set(label) - set(drop)
        if len(expected) > 0:
            recovery = len(set(pred).intersection(expected)) / len(expected)
        else:
            recovery = 1

        pred_missing = [loc for loc in pred if loc not in drop]
        if len(pred_missing) > 0:
            micro_prec = len(set(pred_missing).intersection(expected)) / len(pred_missing)
        else:
            micro_prec = 0

        recall_total.append(recall)
        recovery_total.append(recovery)
        precision_total.append(precision)
        micro_precision_total.append(micro_prec)

    print("average recall {}, average precision {}, average recovery {}, average micro-precision {}". \
          format(np.mean(recall_total), np.mean(precision_total), np.mean(recovery_total), np.mean(micro_precision_total)))
    prec, recall, recovery, micro_precision = np.mean(precision_total), np.mean(recall_total), np.mean(recovery_total), np.mean(
        micro_precision_total)

    return prec, recall, recovery, micro_precision