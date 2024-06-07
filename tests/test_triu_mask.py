
import torch


def _triu_mask(mask, diagonal):
    mask = mask.clone()
    sequence_length, target_length = mask.size()
    for i in range(sequence_length):
        mask.data[i, :i+diagonal] = 0
    return mask

sequence_length = 8
target_length = 8
dtype = torch.bfloat16
device = torch.device("cuda:0")

min_dtype = torch.finfo(dtype).min

mask = torch.full(
    (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
)
print(mask)
mask = _triu_mask(mask, diagonal=1)
print(mask)
