
import os
import torch
import numpy as np
from tqdm import tqdm

MODEL_CONFIGS = {
    "Llama-2-7b-chat-hf": {
        "num_layers": 32, "num_weights": 7,
        "pred_sizes": [
            [4096, 4096], [4096, 4096], [4096, 4096], [4096, 4096],
            [4096, 4096], [4096, 4096], [11008, 11008]
        ]},
    "Meta-Llama-3-8B": {
        "num_layers": 32, "num_weights": 7,
        "pred_sizes": [
            [4096, 4096], [4096, 4096], [4096, 4096], [4096, 4096],
            [4096, 4096], [4096, 4096], [14336, 14336]
        ]},
    "Llama-2-13b-chat-hf": {
        "num_layers": 40, "num_weights": 7,
        "pred_sizes": [
            [5120, 5120], [5120, 5120], [5120, 5120], [5120, 5120],
            [5120, 5120], [5120, 5120], [13824, 13824]
        ]},
    "Llama-2-70b-chat-hf": {
        "num_layers": 80, "num_weights": 7,
        "pred_sizes": [
            [8192, 8192], [8192, 8192], [8192, 8192], [8192, 8192],
            [8192, 8192], [8192, 8192], [28672, 28672]
        ]},
    "Meta-Llama-3-70B": {
        "num_layers": 80, "num_weights": 7,
        "pred_sizes": [
            [8192, 8192], [8192, 8192], [8192, 8192], [8192, 8192],
            [8192, 8192], [8192, 8192], [28672, 28672]
        ]}
}


def score_to_mask(score, sparsity_ratio):
    if len(score.shape) == 1:
        score = score.view(1, -1)
    indices = score.argsort(dim=-1, descending=True, stable=True)
    indices = indices[:, :int(indices.size()[-1] * (1.0 - sparsity_ratio))]
    #print(indices.shape)
    mask = torch.zeros_like(score)
    mask.scatter_(1, indices, 1)
    return mask


def calc_sparsity(inp):
    return (inp.numel() - inp.count_nonzero()) / inp.numel()


class WeightPredictor(object):
    def __init__(self, model_name, dataset_name, dtype=torch.float32, device=torch.device("cuda:0"), D=1024):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dtype = dtype
        self.device = device
        self.D = D
        self.num_layers = MODEL_CONFIGS[model_name]["num_layers"]
        self.num_weights = MODEL_CONFIGS[model_name]["num_weights"]
        self.predictors = []
        self.preds = []
        self.wmetrics = []
        self.pred_sizes = MODEL_CONFIGS[model_name]["pred_sizes"]
        self.attn_sp = 0.5
        self.mlp_sp = 0.8
        self.w_p = 0.0
        self.sparsity_accum = [0.0, 0.0]
        for ilayer in range(self.num_layers):
            self.predictors.append([])
            self.preds.append([])
            self.wmetrics.append([])
            for iweight in range(self.num_weights):
                x_size = self.pred_sizes[iweight][0]
                y_size = self.pred_sizes[iweight][1]
                query_layer = None
                self.predictors[-1].append(query_layer)
                self.preds[-1].append(torch.zeros((1, y_size), dtype=torch.int64, device=device))

                # Load w metrics
                dir_path = os.environ["PREDICTOR_DATA_DIR"]
                data_type = "mean"
                filepath = os.path.join(dir_path, f"ow{data_type}-l{ilayer}w{iweight}.npy")
                #assert os.path.exists(filepath), "Data file not exist: " + filepath
                if os.path.exists(filepath):
                    wm_arr = np.load(filepath)
                    wm_tensor = torch.from_numpy(wm_arr).to(torch.float32).to(device)
                else:
                    wm_tensor = None
                self.wmetrics[-1].append(wm_tensor)
        print(f"Init sparsity: attn {self.attn_sp}, mlp {self.mlp_sp}, w {self.w_p}")

    def load(self, weight_dir=None):
        if weight_dir is None:
            weight_dir = os.path.join("checkpoints", "weight-predictors", self.model_name)
        for ilayer in tqdm(range(1, self.num_layers), desc="Loading predictors..."):
        #for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                weight_path = os.path.join(weight_dir, f"opt-l{ilayer}-w{iweight}.pt")
                if os.path.exists(weight_path):
                    x_size = self.pred_sizes[iweight][0]
                    y_size = self.pred_sizes[iweight][1]
                    query_layer = torch.nn.Sequential(
                        torch.nn.Linear(x_size, self.D, bias=None),
                        torch.nn.Linear(self.D, y_size, bias=None),
                    ).to(self.dtype).to(self.device)
                    ckpt = torch.load(weight_path, map_location="cpu")
                    query_layer.load_state_dict(ckpt, strict=True)
                    self.predictors[ilayer][iweight] = query_layer

    def to_fp16(self):
        self.dtype = torch.float16
        for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                predictor_model = self.predictors[ilayer][iweight]
                if predictor_model is not None:
                    self.predictors[ilayer][iweight] = predictor_model.to(torch.float16)

    def to_bf16(self):
        self.dtype = torch.bfloat16
        for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                predictor_model = self.predictors[ilayer][iweight]
                if predictor_model is not None:
                    self.predictors[ilayer][iweight] = predictor_model.to(torch.bfloat16)

    def set_requires_grad(self, requires_grad):
        for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                predictor_model = self.predictors[ilayer][iweight]
                for param in predictor_model.parameters():
                    param.requires_grad = requires_grad

    def get_trainable_params(self):
        params = []
        for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                predictor_model = self.predictors[ilayer][iweight]
                for param in predictor_model.parameters():
                    if param.requires_grad:
                        params.append(param)
        return params

    def get_module_list(self):
        modules = []
        for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                predictor_model = self.predictors[ilayer][iweight]
                modules.append(predictor_model)
        return modules

    def predict(self, ilayer, iweight, x, prob_threshold=0.5):
        #print(f"Predict: ilayer {ilayer}, iweight {iweight}")
        if ilayer >= self.num_layers:
            return None
        predictor_model = self.predictors[ilayer][iweight]
        logits = predictor_model(x.to(self.dtype).to(self.device))
        probs = logits.sigmoid()
        preds = probs >= prob_threshold
        #print(f"x {x}")
        #print(f"preds {preds}")
        #preds = preds.to(torch.int64)
        self.preds[ilayer][iweight].data = preds.data
        return preds

    def predict_with_top_k(self, ilayer, iweight, x, sp):
        if ilayer >= self.num_layers:
            return None
        if sp <= 0.0 or sp > 1.0:
            return torch.ones(x.shape, dtype=x.dtype, device=x.device)
        org_shape = x.shape
        if len(org_shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
        logits = x.abs()
        thres = logits.sort(dim=-1).values[:, int(logits.shape[-1] * 1.0 * sp)].view(logits.shape[0], 1)
        preds = logits >= thres
        preds = preds.reshape(org_shape).to(torch.int64)
        #if len(org_shape) == 3:
        #    preds.data[:, 0, :] = 1
        #if ilayer >= 0:
        #    preds.data[:, :, :] = 1
        #print(f"il {ilayer}, iw {iweight}, preds_sp {calc_sparsity(preds)}")
        #self.preds[ilayer][iweight].data = preds.data

    def score_to_mask(self, x, sp):
        if len(x.shape) == 2:
            thres = x.sort(dim=-1).values[:, int(x.shape[-1] * 1.0 * sp)].view(x.shape[0], 1)
        elif len(x.shape) == 3:
            thres = x.sort(dim=-1).values[:, :, int(x.shape[-1] * 1.0 * sp)].view(x.shape[0], x.shape[1], 1)
        else:
            raise ValueError("Length of x shape must be 2 or 3")
        mask = x >= thres
        mask = mask.to(torch.int64)
        return mask

    def combine_mask(self, x_mask, w_mask):
        org_shape = x_mask.shape
        hidden_size = x_mask.shape[-1]
        x_mask = x_mask.reshape((-1, hidden_size))
        out_mask = x_mask.clone()
        nsamples = x_mask.shape[0]
        for i in range(nsamples):
            m = x_mask[i] + w_mask[0]
            m = m > 0
            m = m.to(torch.int64)
            out_mask.data[i] = m.data
        return out_mask.reshape(org_shape)

    def predict_by_x_thres(self, ilayer, iweight, x, sp, w_mask_p=0.0):
        if ilayer >= self.num_layers:
            return None
        x = x.abs()
        preds = self.score_to_mask(x, sp)
        if w_mask_p > 0.0:
            wmetrics = self.wmetrics[ilayer][iweight]
            w_mask = self.score_to_mask(wmetrics, 1 - w_mask_p)
            preds = self.combine_mask(preds, w_mask)
        self.sparsity_accum[0] += calc_sparsity(preds)
        self.sparsity_accum[1] += 1
        self.preds[ilayer][iweight].data = preds.data
        #print(f"il {ilayer}, iw {iweight}, preds_sp {calc_sparsity(preds)}")
        return preds

    def predict_heads(self, ilayer, iweight, x, head_dim, head_percent=0.5):
        #print(f"Predict: ilayer {ilayer}, iweight {iweight}")
        if ilayer >= self.num_layers:
            return None
        predictor_model = self.predictors[ilayer][iweight]
        logits = predictor_model(x.to(self.dtype).to(self.device))
        bsz, q_len, hidden_size = x.size()
        num_heads = hidden_size // head_dim
        logits = logits.reshape(bsz, q_len, num_heads, head_dim)
        logits = logits[0, -1].sum(dim=-1)
        logit_indices = logits.argsort(dim=-1)
        preds = torch.zeros((1, num_heads, head_dim), dtype=torch.int64, device=self.device)
        for i in range(int(num_heads * (1.0 - head_percent)), num_heads):
            ihead = logit_indices[i]
            preds.data[0, ihead] = 1
        preds = preds.reshape(1, num_heads * head_dim)
        #print(f"x {x}")
        #print(f"preds {preds}")
        #preds = preds.to(torch.int64)
        self.preds[ilayer][iweight].data = preds.data
        return preds

    def get_pred(self, ilayer, iweight):
        #if ilayer == 0:
        #    return None
        return self.preds[ilayer][iweight]

    def apply_pred(self, ilayer, iweight, x, pred=None):
        #if ilayer == 0:
        #    return x
        bs, q_len, hidden_size = x.size()
        #assert bs == 1
        #if q_len > 1:
        #    return x

        if pred is None:
            pred = self.get_pred(ilayer, iweight)
        #print(f"il {ilayer}, iw {iweight}, preds_sp {calc_sparsity(pred)}")
        return x * pred.to(x.dtype).to(x.device)

    def eval(self, ilayer, iweight, x, y, sparsity_ratio):
        #pred = self.get_pred(ilayer, iweight)
        pred = None
        if ilayer > 0:
            predictor_model = self.predictors[ilayer][iweight]
            logits = predictor_model(x)
            probs = logits.sigmoid()
            prob_threshold = 0.20
            pred = probs >= prob_threshold
            pred = pred.to(torch.int64)

        if pred is None:
            return {"accuracy": 0, "precision": 0, "recall": 0}
        y = score_to_mask(y, sparsity_ratio)
        dif = y.int() - pred.int()
        false_pos = dif > 0.0
        false_neg = dif < 0.0
        total = torch.ones_like(y).sum(dim=1).float()
        pos = y.sum(dim=1).float()
        neg = total - pos
        false_pos = false_pos.sum(dim=1).float()
        true_pos = pos - false_pos
        false_neg = false_neg.sum(dim=1).float()
        true_neg = neg - false_neg
        if True:
            print(f"probs: {probs}")
            print(f"dif: {dif}")
            false_pos_mask = (dif > 0.0)
            false_neg_mask = (dif < 0.0)
            false_pos_probs = probs[false_pos_mask]
            false_neg_probs = probs[false_neg_mask]
            print(f"false_pos_mask: {false_pos_mask}")
            print(f"false_neg_mask: {false_neg_mask}")
            print(f"false_pos_probs: {false_pos_probs}")
            print(f"false_neg_probs: {false_neg_probs}")
        accuracy = ((true_pos + true_neg) / total).mean().item()
        precision = ((true_pos) / pos).mean().item()
        recall = ((true_pos) / (true_pos + false_neg)).mean().item()
        true_wratio = ((pos / total).mean().item())
        pred_wratio = ((true_pos + false_neg) / total).mean().item()
        print(f"ilayer {ilayer}, iweight {iweight}, accuracy {accuracy:.4f}" + \
            f", precision {precision:.4f}, recall {recall:.4f}" + \
            f", true_wratio {true_wratio:.4f}, pred_wratio {pred_wratio:.4f}")
        return {"accuracy": accuracy, "precision": precision, "recall": recall}

    def print_weight(self):
        predictor_model = self.predictors[-1][-1]
        for param in predictor_model.parameters():
            print(param)

    def get_attn_sp(self):
        return self.attn_sp

    def get_mlp_sp(self):
        return self.mlp_sp

    def get_w_p(self):
        return self.w_p

    def set_sp_config(self, attn_sp, mlp_sp, w_p):
        self.attn_sp = attn_sp
        self.mlp_sp = mlp_sp
        self.w_p = w_p

    def reset_sparsity_accum(self):
        self.sparsity_accum = [0.0, 0.0]

    def get_avg_sparsity(self):
        return (self.sparsity_accum[0] * 1.0 / self.sparsity_accum[1]).item()


global_weight_preditor = None
global_attn_prob_threshold = 0.5
global_mlp_prob_threshold = 0.5
global_attn_sp = 0.5
global_mlp_sp = 0.8
global_w_mask_p = 0.0
global_enable_attention_predictor = True


def is_weight_predictor_enabled():
    return os.environ["ENABLE_PREDICTOR"] is not None and os.environ["ENABLE_PREDICTOR"] == "1"


def is_weight_predictor_finetune_enabled():
    return os.environ["ENABLE_PREDICTOR_FINETUNE"] is not None and os.environ["ENABLE_PREDICTOR_FINETUNE"] == "1"


def is_sparse_infer():
    return os.environ["ENABLE_SPARSE_INFER"] is not None and os.environ["ENABLE_SPARSE_INFER"] == "1"


def _init_weight_predictor():
    global global_weight_preditor
    if global_weight_preditor is not None:
        return
    model_name = os.environ["MODEL_NAME"]
    if "-ft" in model_name:
        model_name = model_name[:model_name.find("-ft")]
    dataset_name = "c4"
    dtype = torch.float32
    local_rank = os.environ["LOCAL_RANK"]
    if local_rank != "-1":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
    predictor_ckpt_path = os.environ["PREDICT_CKPT_HOME"]
    D = 1024
    checkpoint_dir = os.path.join(predictor_ckpt_path)
    print("Create and load preditor...")
    print("Local device:", device)
    print("Checkpoint dir:", checkpoint_dir)
    global_weight_preditor = WeightPredictor(
        model_name, dataset_name=dataset_name, dtype=dtype, device=device, D=D
    )
    #global_weight_preditor.load(checkpoint_dir)
    if local_rank != "-1":
        #global_weight_preditor.to_fp16()
        global_weight_preditor.to_bf16()
        global_weight_preditor.set_requires_grad(is_weight_predictor_finetune_enabled())
    else:
        global_weight_preditor.to_bf16()
    #global_weight_preditor.print_weight()


if is_weight_predictor_enabled():
    _init_weight_predictor()
