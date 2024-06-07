
import os
import torch
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
        pred_sizes = MODEL_CONFIGS[model_name]["pred_sizes"]
        for ilayer in range(self.num_layers):
            self.predictors.append([])
            self.preds.append([])
            for iweight in range(self.num_weights):
                    x_size = pred_sizes[iweight][0]
                    y_size = pred_sizes[iweight][1]
                    query_layer = torch.nn.Sequential(
                        torch.nn.Linear(x_size, self.D, bias=None),
                        torch.nn.Linear(self.D, y_size, bias=None),
                    ).to(dtype).to(device)
                    self.predictors[-1].append(query_layer)
                    self.preds[-1].append(torch.zeros((1, y_size), dtype=torch.int64, device=device))

    def load(self, weight_dir=None):
        if weight_dir is None:
            weight_dir = os.path.join("checkpoints", "weight-predictors", self.model_name)
        for ilayer in tqdm(range(1, self.num_layers), desc="Loading"):
        #for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                weight_path = os.path.join(weight_dir, f"opt-l{ilayer}-w{iweight}.pt")
                ckpt = torch.load(weight_path, map_location="cpu")
                predictor_model = self.predictors[ilayer][iweight]
                predictor_model.load_state_dict(ckpt, strict=True)

    def to_fp16(self):
        self.dtype = torch.float16
        for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                predictor_model = self.predictors[ilayer][iweight]
                self.predictors[ilayer][iweight] = predictor_model.to(torch.float16)

    def to_bf16(self):
        self.dtype = torch.bfloat16
        for ilayer in range(1, self.num_layers):
            for iweight in range(self.num_weights):
                predictor_model = self.predictors[ilayer][iweight]
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
        for i in range(int(num_heads * head_percent), num_heads):
            ihead = logit_indices[i]
            preds.data[0, ihead] = 1
        preds = preds.reshape(1, num_heads * head_dim)
        #print(f"x {x}")
        #print(f"preds {preds}")
        #preds = preds.to(torch.int64)
        self.preds[ilayer][iweight].data = preds.data
        return preds

    def get_pred(self, ilayer, iweight):
        if ilayer == 0:
            return None
        return self.preds[ilayer][iweight]

    def apply_pred(self, ilayer, iweight, x):
        if ilayer == 0:
            return x
        pred = self.get_pred(ilayer, iweight)
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


global_weight_preditor = None
global_attn_prob_threshold = 0.5
global_mlp_prob_threshold = 0.5
global_enable_attention_predictor = True


def is_weight_predictor_enabled():
    return os.environ["ENABLE_PREDICTOR"] is not None and os.environ["ENABLE_PREDICTOR"] == "1"


def is_weight_predictor_finetune_enabled():
    return os.environ["ENABLE_PREDICTOR_FINETUNE"] is not None and os.environ["ENABLE_PREDICTOR_FINETUNE"] == "1"


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
    global_weight_preditor.load(checkpoint_dir)
    if local_rank != "-1":
        #global_weight_preditor.to_fp16()
        global_weight_preditor.to_bf16()
        global_weight_preditor.set_requires_grad(is_weight_predictor_finetune_enabled())
    else:
        global_weight_preditor.to_bf16()
    #global_weight_preditor.print_weight()


if is_weight_predictor_enabled():
    _init_weight_predictor()
