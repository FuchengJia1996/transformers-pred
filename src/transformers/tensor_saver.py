
import os
import numpy as np


class TensorSaver(object):
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.cur_seq_idx = 0

    def save(self, t, il, tn):
        save_dir = os.environ["TENSOR_SAVE_DIR"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if len(t.shape) == 3:
            num_batch = t.shape[-3]
            assert num_batch == 1
        num_tokens = t.shape[-2]
        seq_idx = self.cur_seq_idx
        for i in range(num_tokens):
            filename = f"s{seq_idx}-l{il}-{tn}.npy"
            filepath = os.path.join(save_dir, filename)
            t_arr = t[0, i].cpu().detach().numpy()
            print(f"Save {filename}")
            np.save(filepath, t_arr)
            seq_idx += 1

    def add_seq(self, ns):
        self.cur_seq_idx += ns


global_tensor_saver = None


def _init_tensor_saver():
    global global_tensor_saver
    model_name = os.environ["MODEL_NAME"]
    dataset_name = "c4"
    global_tensor_saver = TensorSaver(model_name, dataset_name)


#_init_tensor_saver()
