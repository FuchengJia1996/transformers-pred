
import os
import torch
import numpy as np
from tqdm import tqdm
import json
import math
import matplotlib.pyplot as plt
import time
class ActivationModule:
    def __init__(self) :
        self.visualize_strategy = 1
        self.activations = None
        self.histograms = None
        self.file_path = None
        self.num_layers = 0
        self.num_weights = 0
    def set_visuable_strategy(self, s=1) :
        self.visualize_strategy = s
    def set_init_dict(self , num_layers , num_weights, file_path) :
        self.num_layers = num_layers
        self.num_weights = num_weights
        self.activations = [[[] for id in range(num_weights)] for _ in range(num_layers)]
        self.histograms= [[{} for id in range(num_weights)] for _ in range(num_layers)]
        self.file_path = file_path
        
    def fd_th(self, weight, sp) :
        tensor = weight.flatten().cpu().numpy()  # 将张量转换为 NumPy 数组
        threshold = np.percentile(tensor, q=sp*100)  # 计算分位数
        return threshold.item()        

    def fd_th_by_histogram(self, histogram, sp) :
        bin_centers = histogram["bin_centers"]
        counts = histogram["histogram"]  # 使用 histogram 键

        # 计算累积分布
        cumulative_counts = torch.cumsum(counts, dim=0)
        total_count = cumulative_counts[-1]

        # 找到满足稀疏性比例 sp 的阈值
        target_count = (1 - sp) * total_count
        idx = torch.searchsorted(cumulative_counts, target_count)

        if idx == 0:
            threshold = bin_centers[0]
        elif idx == len(bin_centers):
            threshold = bin_centers[-1]
        else:
            lower_count = cumulative_counts[idx - 1]
            upper_count = cumulative_counts[idx]
            lower_value = bin_centers[idx - 1]
            upper_value = bin_centers[idx]

            fraction = (target_count - lower_count) / (upper_count - lower_count)
            threshold = lower_value + fraction * (upper_value - lower_value)
        return threshold.item()

    def cal_histograms(self, activations):
        flattened_activations = activations.flatten().detach().to('cuda')

        acts = torch.sort(flattened_activations)[0]

        lower_bound = acts[int(0.01 * len(acts))]
        upper_bound = acts[int(0.99 * len(acts))]
        filtered_activations = flattened_activations[
            (flattened_activations >= lower_bound) & (flattened_activations <= upper_bound)
        ]

        bins = 1000

        histogram, bin_edges = np.histogram(filtered_activations.cpu().numpy(), bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return histogram, bin_edges, bin_centers

    def visualize_histogram(self, layer_idx, weight_idx):
        histograms_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}", "histograms.pt")

        # 加载直方图数据
        histogram_data = torch.load(histograms_path, map_location='cpu')
        histogram, bin_edges, bin_centers = histogram_data['histogram'], histogram_data['bin_edges'], histogram_data['bin_centers']

        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, histogram, width=(bin_edges[-1] - bin_edges[0]) / len(bin_edges), alpha=0.7, color='blue')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(f"Histogram for Layer {layer_idx}, Weight {weight_idx}")

        # 先保存再显示
        plt.savefig(os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}", "histograms.png"))
        plt.show()
        plt.close() 

    
    def save_layer_weight(self, layer_idx, weight_idx) :
        if not self.activations[layer_idx][weight_idx]:
            print('None', self.activations[layer_idx][weight_idx])
            return 
        else :   
            layer_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}")
            os.makedirs(layer_path, exist_ok=True)

            activations_path = os.path.join(layer_path, "activations.pt")
            combine_activations = torch.cat(self.activations[layer_idx][weight_idx], dim=0)
            torch.save(combine_activations, activations_path)

            histograms_path = os.path.join(layer_path, "histograms.pt")
            histogram, bin_edges, bin_centers = self.cal_histograms(combine_activations)
            # print({'histogram': histogram, 'bin_edges': bin_edges, 'bin_centers': bin_centers})
            torch.save({'histogram': histogram, 'bin_edges': bin_edges, 'bin_centers': bin_centers}, histograms_path)

    def clear_layer_weight(self, layer_idx, weight_idx) :
        print('clear', layer_idx, weight_idx)
        self.activations[layer_idx][weight_idx] = []
        
    def grab_activations(self, x, layer_idx, weight_idx):
        if x.size(1) > 1:  # Check if seq_len > 1
            print('grab ', layer_idx, weight_idx, x.size())
            self.activations[layer_idx][weight_idx].append(x.detach().squeeze(0).cpu().float())
            
            self.save_layer_weight(layer_idx, weight_idx)
            self.clear_layer_weight(layer_idx, weight_idx)
            # 可视化直方图
            if self.visualize_strategy == 1 :
                start_time = time.time()
                self.visualize_histogram(layer_idx, weight_idx)
                visualize_time = time.time() - start_time  
                print(f"visualize_histogram ({layer_idx}, {weight_idx}) took {visualize_time:.6f} seconds")
    
    def save_activations(self):
        for layer_idx in range(self.num_layers):
            for weight_idx in range(self.num_weights) :
                self.save_layer_weight(layer_idx, weight_idx)

    def load_activations(self):
        self.activations = [[[] for _ in range(self.num_weights)] for _ in range(self.num_layers)]
        self.histograms = [[{} for _ in range(self.num_weights)] for _ in range(self.num_layers)]

        for layer_idx in range(self.num_layers):
            for weight_idx in range(self.num_weights):
                layer_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}")
                
                activations_path = os.path.join(layer_path, "activations.pt")
                if os.path.exists(activations_path):
                    self.activations[layer_idx][weight_idx] = torch.load(activations_path)
                    
                histograms_path = os.path.join(layer_path, "histograms.pt")
                if os.path.exists(histograms_path):
                    self.histograms[layer_idx][weight_idx] = torch.load(histograms_path)
    
    def find_threshold(self, sp, output_path):
        thresholds = {}
        for layer_idx in range(self.num_layers):
            layer_thresholds = {}
            for weight_idx in range(self.num_weights):
                layer_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}")
                activations_path = os.path.join(layer_path, "activations.pt")
                
                if not os.path.exists(activations_path):
                    continue

                weight = torch.load(activations_path)
                threshold = self.fd_th(torch.abs(weight), sp)
                layer_thresholds[weight_idx]=threshold
                print('find_threshold:  ', layer_idx, weight_idx, threshold)                
                
                del weight
                
                
            thresholds[layer_idx]=layer_thresholds

        # Save thresholds to output_path
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"sparse-{sp}.json")
        with open(save_path,'w') as f:
            json.dump(thresholds, f)
        # print(thresholds)
        # torch.save(thresholds, )
        return thresholds
        
class STEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for the backward pass.
    The forward pass multiplies the input by a mask, and the backward pass
    simply passes the gradient through without modification.
    """
    @staticmethod
    def forward(ctx, input, mask):
        # Forward pass: apply the mask to the input
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Return gradient for input and None for mask

class WeightPredictor(object):
    def __init__(self, model_name='Meta-Llama-3-8B', dtype=torch.float32, device=torch.device("cuda:0"), D=1024):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.sparsity_strategy = 'Dynamic'
        
        self.num_layers = 0
        self.num_weights = 0
        self.weight_counters = []
        self.weight_maps = {}

        self.DO_CAL_ACTIVATIONS = False
        self.activations = ActivationModule()

        self.sparse_infer = 1
        self.sparse_params = [0,0]
        
        self.reset()

    def set_sparse_infer(self, s=1) :
        self.sparse_infer = s
    def is_sparse_infer(self) -> bool:
        return self.sparse_infer
    def reset(self) :
        print('Init Reset')
        self.attn_sp = 0.0
        self.mlp_sp = 0.0
        self.w_p = 0.0
        self.threshold = [[0.0] * self.weight_counters[_] for _ in range(self.num_layers)]
        self.do_pre_prediction = 0

        # CROSS_LAYER_TEST
        self.preds = [[None for _ in range(self.weight_counters[layer_id])] for layer_id in range(self.num_layers)]
        self.wmetrics = [[None for _ in range(self.weight_counters[layer_id])] for layer_id in range(self.num_layers)]
        self.similarity_results = [[] * self.weight_counters[_] for _ in range(self.num_layers)]

    def set_cal_activations(self, file_path) :
        print('set activations', self.num_layers,self.num_weights)
        self.DO_CAL_ACTIVATIONS = True
        self.activations.set_init_dict(self.num_layers, self.num_weights, file_path)
        
    def to_fp16(self):
        self.dtype = torch.float16

    def to_bf16(self):
        self.dtype = torch.bfloat16
        
    def set_sparsity_threshold(self, file_path=None) :
        print('set_sparsity_threshold')
        if file_path == None :
            file_path = os.environ.get('THRESHOLD_PATH',None)
        if file_path == None : 
            print('there is none this file.')
        #     file_path = f'./threshold/{self.model_name}/{self.model_name}-{self.attn_sp}.txt'
        print('threshold_path', file_path)
        self.threshold = [[0.0] * self.num_weights for _ in range(self.num_layers)] 
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                sparsity_all_dict = json.load(f)
                
            for i in range(self.num_layers):
                layer_key = f"{i}"
                if layer_key in sparsity_all_dict:
                    layer_thresholds = sparsity_all_dict[layer_key]
                    for j in range(self.num_weights) :
                        self.threshold[i][j] = layer_thresholds.get(f"{j}", 0.0)
                        
            self.sparsity_strategy = 'Static'
            
            # print(self.threshold)
        else:
            self.sparsity_strategy = 'Dynamic'
        print('sparsity_strategy : ', self.sparsity_strategy)
         
    def score_to_mask(self, x, sp, thres=0.0, ilayer=-1):
            # Dynamic TOP-K
        b = thres
        if len(x.shape) == 2:
            thres = x.sort(dim=-1).values[:, int(x.shape[-1] * sp)].view(x.shape[0], 1)
        elif len(x.shape) == 3:
            thres = x.sort(dim=-1).values[:, :, int(x.shape[-1] * sp)].view(x.shape[0], x.shape[1], 1)
        else:
            raise ValueError("Length of x shape must be 2 or 3")
        a = thres

        # choose threshold
        if self.sparsity_strategy == 'Dynamic':
            thres = a
        elif self.sparsity_strategy == 'Static':
            thres = b
        elif self.sparsity_strategy == 'Mixmin':
            b_tensor = torch.tensor(b, device=a.device, dtype=a.dtype)
            thres = torch.minimum(a, b_tensor)
        elif self.sparsity_strategy == 'Mixmax':
            b_tensor = torch.tensor(b, device=a.device, dtype=a.dtype)
            thres = torch.maximum(a, b_tensor)
        else:
            thres = b

        # all activation in layer 0
        r = os.environ.get('ACTIVATE_LAYER' , '0') 
        if ilayer >= 0 and ilayer <= int(r): 
            # print('YES')
            mask = x >= 0 
        else :
            mask =  x >= thres
        mask = mask.to(torch.int64)

        # compute sparse param C
        if mask.sum() > 0:  
            sum_all = x.sum()
            sum_masked = (x * mask).sum() 
            C = sum_all / sum_masked
        else:
            C = 1.0 

        return mask, C

    def predict_by_x_thres(self, ilayer, iweight, x):
        # print('predict' , ilayer , iweight)
        sp = self.attn_sp
        # Prediction.
        x = x.abs()
        threshold = self.threshold[ilayer][iweight]
        preds, C = self.score_to_mask(x, sp, threshold, ilayer)
       
        # predictor
        if self.do_pre_prediction:
            self.preds[ilayer][iweight] = preds
            if ilayer > 0:
                prev_preds = self.preds[ilayer - 1][iweight]
                if prev_preds is not None:
                    device = preds.device
                    prev_preds = prev_preds.to(device)
                    current_ones = preds.sum().item()
                    if current_ones > 0:
                        common_ones = (preds & prev_preds).sum().item()
                        similarity = common_ones / current_ones
                        # print(f'ilayer {ilayer} iweight {iweight} similarity {similarity}')
                        self.similarity_results[ilayer][iweight].append(similarity)
                    self.preds[ilayer - 1][iweight] = None # Clear
                else :
                    pass

        return preds, C

    def apply_pred(self, x, pred=None):
        return x if pred is None else x * pred.to(x.dtype).to(x.device)
    
    def generate_pred(self, ilayer, iweight, x) :
        # print('grab ', ilayer, iweight, x.size())
        if self.DO_CAL_ACTIVATIONS == True:
            self.activations.grab_activations(x, ilayer, iweight)
        if self.is_sparse_infer() == False:
            return x
        else :
            pred, C = self.predict_by_x_thres(ilayer, iweight, x)
            if os.environ.get('DEBUG_CROSSLAYER','0') != '0' :
                # 统计 pred 中零值的数量和总元素数量
                total_elements = pred.numel()
                zero_elements = (pred == 0).sum().item()

                # update self.sparse_params
                self.sparse_params[0] += total_elements
                self.sparse_params[1] += zero_elements

                # zero_ratio = zero_elements / total_elements
                # print(f"Layer {ilayer}, Weight {iweight}: Zero ratio in pred = {zero_ratio:.4f}")


            if os.environ.get('BACKWARD_STRATEGY','0') != '0' :
                return self.apply_pred(x, pred)
            else : 
                return STEFunction.apply(x, pred)

    def set_sp_config(self, attn_sp, mlp_sp, w_p):
        self.attn_sp = attn_sp
        self.mlp_sp = mlp_sp
        self.w_p = w_p
        print(f"Set sparsity: attn {self.attn_sp}, mlp {self.mlp_sp}, w {self.w_p}")

    def set_do_pre_prediction(self, do_pre_prediction):
        self.do_pre_prediction = do_pre_prediction
        print(f"Set pre-prediction: {self.do_pre_prediction}")

    def set_sparsity_strategy(self, method: str) :
        self.sparsity_strategy = method
        print('sparsity_strategy: ', method)

    def set_layers_and_weights(self, num_layers , weight_counters, weight_map) :
        self.num_layers = num_layers
        self.num_weights = weight_counters[0]
        self.weight_counters = weight_counters
        self.weight_map = weight_map
        print('num_layers', self.num_layers)
        print('weight_counters', self.weight_counters)
        self.reset()

global_weight_preditor = None

def is_sparse_infer():
    # return os.environ.get("ENABLE_SPARSE_INFER", "0") == "1"
    return False

def _init_weight_predictor(model_name=None):
    global global_weight_preditor
    if global_weight_preditor is not None:
        raise KeyError('global_weight_preditor')
    
    dtype = torch.float32
    local_rank = os.environ.get("LOCAL_RANK","-1")

    if local_rank != "-1":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
    
    print("Create and load preditor...")
    print("Local device:", device)
    # print("Checkpoint dir:", checkpoint_dir)
    global_weight_preditor = WeightPredictor(model_name, dtype=dtype, device=device,)
    global_weight_preditor.to_bf16()
    return global_weight_preditor

global_weight_preditor = _init_weight_predictor()