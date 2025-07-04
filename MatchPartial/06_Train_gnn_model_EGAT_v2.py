from .utilize_func import *
from .model_sim_EGAT_v2_h8 import *
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from .parameters import *
from .eval_prediction_func import *
from .mask_head import *
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import random
from tqdm import tqdm
from pathlib import Path
import copy


def save_best_model(model, optimizer, train_history, train_metrics, valid_history, valid_metrics, model_path):
    def should_save(metric, history, idx, comparison_func, smoothing_sigma=10):
        history_array = np.array(history)
        if history_array.ndim == 1:
            smoothed_history = gaussian_filter1d(history_array, sigma=smoothing_sigma)
        elif history_array.ndim == 2:
            smoothed_history = gaussian_filter1d(history_array[:, idx], sigma=smoothing_sigma)
        else:
            raise ValueError(f"Unexpected history dimensions: {history_array.ndim}")
        return comparison_func(metric, smoothed_history).any()

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    save_criteria = [
        (train_metrics[1], train_history, 1, np.less, 'loss_train.pt'),
        (train_metrics[2], train_history, 2, np.greater, 'acc_train.pt'),
        (train_metrics[3], train_history, 3, np.greater, 'precision_train.pt'),
        (train_metrics[4], train_history, 4, np.greater, 'recall_train.pt'),
        (valid_metrics[1], valid_history, 1, np.less, 'loss_valid.pt'),
        (valid_metrics[2], valid_history, 2, np.greater, 'acc_valid.pt'),
        (valid_metrics[3], valid_history, 3, np.greater, 'precision_valid.pt'),
        (valid_metrics[4], valid_history, 4, np.greater, 'recall_valid.pt'),
    ]

    for metric, history, idx, comp_func, filename in save_criteria:
        if should_save(metric, history, idx, comp_func):
            torch.save(state, Path(model_path) / filename)

    torch.save(state, Path(model_path) / 'last_epoch.pt')


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif hasattr(m, 'weight') and m.weight is not None and isinstance(m.weight, torch.nn.Parameter):
        nn.init.xavier_uniform_(m.weight)


def generate_test_t1_t2(t_max, t_ref_list):
    result_array = np.empty((0, 2), int)
    for t_ref in t_ref_list:
        t_list = np.arange(0, t_max, t_max // 30 if t_max > 30 else t_max // 5)
        result = np.column_stack((np.full_like(t_list, t_ref), t_list))
        result_array = np.concatenate((result_array, result), axis=0)
    return result_array


def generate_random_pair_t1_t2(t_max):
    t_list = np.arange(0, t_max)
    np.random.seed(42)
    np.random.shuffle(t_list)
    t_orig = t_list[:len(t_list) // 2 * 2].reshape(-1, 2)

    random.seed(42)
    range_array = np.tile(np.arange(370, 390), 3)
    if t_max > len(range_array):
        random_selection = random.sample(range(0, t_max), len(range_array))
    else:
        random_selection = random.sample(range(0, t_max), 5)

    min_len = min(len(random_selection), len(range_array))
    result = np.column_stack((
        np.array(random_selection[:min_len]),
        np.array(range_array[:min_len])
    ))

    return np.concatenate((t_orig, result), axis=0)


def compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio):
    all_match_scores1 = model(data1, data2)
    mask1 = model.mask1 if method == 3 else None
    mask2 = model.mask2 if method == 3 else None

    edge_label1 = get_edge_label(data1, data2, nearby_search_num, norm_scale, with_AM, device, mask1, mask2)
    class_weights1 = get_class_weights(edge_label1).to(device) * torch.tensor([1, ratio]).to(device)

    criteria1 = nn.CrossEntropyLoss(weight=class_weights1, reduction='mean')
    loss1 = criteria1(all_match_scores1, edge_label1.to(torch.long))
    return loss1, all_match_scores1, edge_label1


def update_ratio(epoch, total_epochs):
    initial_ratio = 1
    final_ratio = 0.05
    return initial_ratio - ((initial_ratio - final_ratio) * (epoch / total_epochs))


if __name__ == "__main__":
    with_AM = False
    ratio = float(ratio)
    nearby_search_num = int(nearby_search_num)
    os.makedirs(model_path, exist_ok=True)
    np.savetxt(os.path.join(model_path, 'train_history.txt'), [])

    print(ratio)
    print(nearby_search_num)
    print(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    model = NodeLevelGNN(nearby_search_num, norm_scale, with_AM, device).to(device)
    model.apply(init_weights)
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    train_history = []
    valid_history = []
    t1_t2_pair = generate_random_pair_t1_t2(t_max)
    pair = t1_t2_pair

    t_ref_list = list(t_initial_list)
    t = 10
    test_t1_t2 = generate_test_t1_t2(t_max, t_ref_list)[t:t + 15]

    print("graph_path", graph_path)
    print("device", device)

    for epoch in range(500):
        loss_train = tn_all = fp_all = fn_all = tp_all = balanced_acc_all = 0
        model.train()

        for t1, t2 in pair:
            optimizer.zero_grad()

            t1 = np.clip(int(t1) + random.choice([-1, 0, 1, 0]), 0, t_max - 1)
            t2 = np.clip(int(t2) + random.choice([-1, 0, 1, 0]), 0, t_max - 1)

            if method == 3:
                model.mask1 = extend_mask_along_central(graph_path.parent / 'data.h5', t1, ch, centroid_ref, extension_length)
                model.mask2 = extend_mask_along_central(graph_path.parent / 'data.h5', t2, ch, centroid_ref, extension_length)

            data1, data2 = get_pair_data(graph_path, int(t1), int(t2), device)
            gamma = 2
            loss, all_match_scores1, edge_label1 = compute_loss_data12(
                model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio
            )

            all_match_scores_m = torch.argmax(all_match_scores1, dim=1)
            if torch.sum(edge_label1) + torch.sum(all_match_scores_m) == 0:
                tn, fp, fn, tp = edge_label1.shape[0], 0, 0, 0
            else:
                tn, fp, fn, tp = confusion_matrix(edge_label1.cpu(), all_match_scores_m.cpu()).ravel()

            tn_all += tn
            fp_all += fp
            fn_all += fn
            tp_all += tp

            loss.backward()
            optimizer.step()
            loss_train += loss.item()

            print(epoch, [t1, t2], loss.item(), fp, tp)

        loss_train /= len(pair)
        balanced_acc_all /= len(pair)

        if epoch % 1 == 0:
            model.eval()
            train_metrics = [int(epoch), loss_train] + compute_measure_matrix(tn_all, fp_all, fn_all, tp_all)
            valid_metrics = [int(epoch)] + eval_model(model, graph_path, test_t1_t2, device, nearby_search_num, norm_scale, with_AM, ratio)

            train_metrics = [round(m, 3) for m in train_metrics]
            valid_metrics = [round(m, 3) for m in valid_metrics]

            print(train_metrics)
            print(valid_metrics)

            save_best_model(model, optimizer, train_history, train_metrics, valid_history, valid_metrics, model_path)
            train_history.append(train_metrics)
            valid_history.append(valid_metrics)

            np.savetxt(Path(model_path) / 'train_history.txt', train_history)
            np.savetxt(Path(model_path) / 'valid_history.txt', valid_history)