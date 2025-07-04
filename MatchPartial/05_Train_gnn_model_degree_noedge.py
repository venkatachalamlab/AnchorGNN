

from .utilize_func import *
from .model_sim_degree_noedge import *
# from GNN_model_orig import *

from sklearn.metrics import confusion_matrix
import sys
import random
from tqdm import tqdm
import os
from sklearn.utils.class_weight import compute_class_weight
from .parameters import *
import copy
from .eval_prediction_func import *
from .mask_head import *
from scipy.ndimage import gaussian_filter1d


def read_args():
    with_AM = sys.argv[1] == 'True'
    t_interval = sys.argv[2]
    ratio = sys.argv[3]
    nearby_search_num = sys.argv[4]
    model_path = sys.argv[5]
    
    return with_AM, t_interval, ratio, nearby_search_num, model_path




def save_best_model(model, optimizer, train_history, train_metrics, valid_history, valid_metrics, model_path):
    """
    Saves the model state based on best performance in different metrics and saves the last epoch.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        train_history (list): List of training metrics over epochs.
        train_metrics (list): Current training metrics [metric_0, loss, acc, precision, recall].
        valid_history (list): List of validation metrics over epochs.
        valid_metrics (list): Current validation metrics [metric_0, loss, acc, precision, recall].
        model_path (str): Path to save the model files.
    """
    def should_save(metric, history, idx, comparison_func, smoothing_sigma=10):
        """Determine if the model should be saved for a specific metric."""
        history_array = np.array(history)

        # Handle 1D or 2D history
        if history_array.ndim == 1:
            smoothed_history = gaussian_filter1d(history_array, sigma=smoothing_sigma)
        elif history_array.ndim == 2:
            smoothed_history = gaussian_filter1d(history_array[:, idx], sigma=smoothing_sigma)
        else:
            raise ValueError(f"Unexpected history dimensions: {history_array.ndim}")

        # Use np.any() or np.all() based on desired logic
        return comparison_func(metric, smoothed_history).any()

    # Save model state
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    # Define metrics and comparison functions for saving
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

    # Loop through criteria and save if conditions are met
    for metric, history, idx, comp_func, filename in save_criteria:
        if should_save(metric, history, idx, comp_func):
            torch.save(state, model_path + filename)

    # Save the last epoch model
    torch.save(state, model_path + 'last_epoch.pt')



with_AM, t_interval, ratio, nearby_search_num, model_path = read_args()
ratio = float(ratio)
nearby_search_num = int(nearby_search_num)
os.makedirs(model_path, exist_ok=True)
np.savetxt(os.path.join(model_path, 'train_history.txt'), [])
print(with_AM)
print(t_interval)
print(ratio)
print(nearby_search_num)
print(model_path)





def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")



model = NodeLevelGNN(nearby_search_num, norm_scale, with_AM, device).to(device)


model.apply(init_weights)
model.train()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3)






def generate_test_t1_t2(t_end,t_ref_list):
    result_array = np.empty((0, 2), int)
    for t_ref in t_ref_list:
        
        t_list = np.arange(0, t_end, t_end // 30)
        first_column = np.full_like(t_list, t_ref).reshape(-1, 1)  # Create a column of 444
        result = np.concatenate((first_column, t_list.reshape(-1, 1)), axis=1)
        result_array = np.concatenate((result_array, result), axis=0)
    return result_array


def generate_random_pair_t1_t2(t_end):
    t_list = np.arange(0, t_end, 1)
    np.random.seed(42)
    np.random.shuffle(t_list)
    t_orig = t_list.reshape(-1, 2)


    random.seed(42)  # E
    range_array = np.tile(np.arange(370, 390), 3)  # Repeat to match 60 values
    random_selection = random.sample(range(0, 1060), len(range_array))
    result = np.column_stack((random_selection, range_array))
    t_train = np.concatenate((t_orig,result),axis = 0)
    return t_train





def compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio):

    
    all_match_scores1 = model(data1, data2)

    if method ==3:
        mask1 = model.mask1
        mask2 = model.mask2
    else:
        mask1 = None
        mask2 = None
    
    edge_label1 = get_edge_label(data1,data2,nearby_search_num,norm_scale,with_AM,device, mask1, mask2)
    
    # edge_label1 = get_edge_label(data1,data2,nearby_search_num,norm_scale,with_AM,device)


    ## Method 1: cross entropy loss
    # class_weights1 = torch.tensor([1,1],dtype=torch.float32).to(device)
    class_weights1 = get_class_weights(edge_label1) 
    class_weights1 = (class_weights1 * torch.tensor([1,ratio])).to(device)
    criteria1 = nn.CrossEntropyLoss(weight = class_weights1,reduction='mean')
    loss1 = criteria1(all_match_scores1,edge_label1.to(torch.long))

    
    # # Method 2: focal loss
    # # class_weights1 = compute_class_weight(class_weight="balanced", classes=np.unique(edge_label1.cpu().numpy()), y=edge_label1.cpu().numpy())
    # # class_weights1 = torch.tensor(class_weights1,dtype=torch.torch.float32).to(device)
    # # class_weights1 = (class_weights1 * torch.tensor([1,ratio]).to(device))
    # class_weights1 = torch.tensor([1,1],dtype=torch.float32).to(device)
    # criteria1 = FocalLoss(alpha=class_weights1, gamma=gamma, reduction='mean')
    # loss1 = criteria1(all_match_scores1,edge_label1.to(torch.long))

    return loss1, all_match_scores1, edge_label1




optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3)
t_end = t_max
optimizer.zero_grad()  
train_history = []    
valid_history = []
t1_t2_pair = generate_random_pair_t1_t2(t_end)
pair = t1_t2_pair


t_ref_list = [444, 463]
t = 10
test_t1_t2 = generate_test_t1_t2(t_end,[444, 463])[t:t+15]
# test_t1_t2 = generate_pair_t1_t2_orig(t_end,1)[0:10] ## To test on 'graph2'


total_epochs = 250 

def update_gamma(epoch, total_epochs):
    # Linearly increase gamma from 0 to 2
    initial_gamma = 0
    final_gamma = 4
    gamma = initial_gamma + ((final_gamma - initial_gamma) * (epoch / total_epochs))
    return gamma




def update_ratio(epoch, total_epochs):
    initial_ratio = 1
    final_ratio = 0.05
    ratio = initial_ratio - ((initial_ratio - final_ratio) * (epoch / total_epochs))
    return ratio





print("graph_path",graph_path)


# for epoch in range(0, total_epochs+1): 
print("device",device)
for epoch in range(500):
    [loss_train, tn_all, fp_all, fn_all, tp_all, balanced_acc_all] = [0, 0, 0, 0, 0, 0 ]


    
    num = 0
    idx = epoch%5

    model.train()
    # np.random.shuffle(pair_tot)
    for t1,t2 in pair:
       
        optimizer.zero_grad()


        # ratio = update_ratio(epoch, total_epochs)
        # gamma = update_gamma(epoch, total_epochs)
        # ratio = 0.2
        gamma = 2

        
        t1 = np.clip(int(t1) + random.choice([-1,0,1,0]), 0, t_max - 1)
        t2 = np.clip(int(t2) + random.choice([-1,0,1,0]), 0, t_max - 1)
        
        # model.mask1 = produce_head_mask(graph_path.parent/('data.h5'), t1, ch)
        # model.mask2 = produce_head_mask(graph_path.parent/('data.h5'), t2, ch)
        model.mask1 = extend_mask_along_central(graph_path.parent/('data.h5'), t1, ch, centroid_ref, extension_length)
        model.mask2 = extend_mask_along_central(graph_path.parent/('data.h5'), t2, ch, centroid_ref, extension_length)
        
        
        



        # #### Method 2: add nearby frame training to cancel out the inner similarity
        # data1, data2 = get_pair_data(graph_path, int(t1), np.clip(int(t1+random.choice([-1,1])),0, t_max - 1),device)
        # loss11, all_match_scores11, edge_label11 = compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio)
        # data1, data2 = get_pair_data(graph_path, int(t1), np.clip(int(t1+random.choice([-5,5])),0, t_max - 1),device)
        # loss22, all_match_scores22, edge_label22 = compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio)
        # data1, data2 = get_pair_data(graph_path, int(t1),int(t2),device)
        # loss1, all_match_scores1, edge_label1 = compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio)



        # #### Method 2: add nearby frame training to cancel out the inner similarity
        # data1, data2 = get_pair_data(graph_path, int(t1), np.clip(int(t1+random.choice([-1,1])),0, t_max - 1),device)
        # loss11, all_match_scores11, edge_label11 = compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio)
        # data1, data2 = get_pair_data(graph_path, int(t1), np.clip(int(t1+random.choice([-1,1])),0, t_max - 1),device)
        # loss22, all_match_scores22, edge_label22 = compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio)
        # data1, data2 = get_pair_data(graph_path, int(t1),int(t2),device)
        # loss1, all_match_scores1, edge_label1 = compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio)
        

        ### Method 1: random pair
        data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)
        loss1, all_match_scores1, edge_label1 = compute_loss_data12(model, data1, data2, nearby_search_num, norm_scale, with_AM, device, gamma, ratio)


        # loss = loss1 + (loss11 + loss22) * 1  ## inner similarity loss
        loss = loss1
        
    


        all_match_scores_m = torch.argmax(all_match_scores1,dim = 1)
        # all_match_scores_m = torch.sigmoid(all_match_scores1) > 0.5
        # all_match_scores_m = (all_match_scores1) > 0.5
        
        if torch.sum(edge_label1) + torch.sum( all_match_scores_m) == 0:
            [tn, fp, fn, tp] = [edge_label1.shape[0], 0, 0, 0]
        else:
            tn, fp, fn, tp = confusion_matrix(edge_label1.detach().cpu(), all_match_scores_m.detach().cpu()).ravel()

        # tn, fp, fn, tp = confusion_matrix(edge_label1.detach().cpu(), all_match_scores_m.detach().cpu()).ravel()
        # balanced_acc = balanced_accuracy_score(all_match_scores_m.detach().numpy(),edge_label1.to(torch.long).detach().numpy())
        tn_all += tn
        fp_all += fp
        fn_all += fn 
        tp_all += tp
        # balanced_acc_all += balanced_acc
        num += 1

        # loss.requires_grad = True
        
        
        loss.backward()  

        # Clip gradients to avoid exploding gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients by norm


        optimizer.step()  

        loss_train += loss.item()


    loss_train = loss_train/num
    balanced_acc_all = balanced_acc_all/num


    if epoch % 1 == 0 :
        model.eval()
        train_metrics = [int(epoch), loss_train] + compute_measure_matrix(tn_all, fp_all, fn_all, tp_all)
        valid_metrics = [int(epoch)]+ eval_model(model,graph_path,test_t1_t2,device,nearby_search_num,norm_scale,with_AM,ratio)

        train_metrics = [round(num, 3) for num in train_metrics]
        valid_metrics = [round(num, 3) for num in valid_metrics]

        print(train_metrics)
        print(valid_metrics)
                
        save_best_model(model,optimizer,train_history,train_metrics, valid_history,valid_metrics, model_path)
        train_history.append(train_metrics)
        valid_history.append(valid_metrics)
        np.savetxt(model_path+'train_history.txt',train_history)
        np.savetxt(model_path+'valid_history.txt',valid_history)   





edge_label1_tot, all_match_scores_m_tot = evaluate_prediction_results(444, model, graph_path, method, t_max, nearby_search_num, norm_scale, with_AM, device, 0.6)
edge_label1_tot, all_match_scores_m_tot = evaluate_prediction_results(444, model, graph_path, method, t_max, nearby_search_num, norm_scale, with_AM, device, 0.7)














