
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import LocalMask
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import make_local_attention_mask
import numpy as np

local_mask = make_local_attention_mask((10, 10), window_size=(None,0))


print(local_mask)

kv_idx = np.arange(5,dtype=np.int32)
print(kv_idx[:, None].shape)
print(kv_idx[None, :].shape)

# def make_local_attention_mask(
#     shape: Tuple[int, int],
#     window_size: Tuple[int | None, int | None],
#     *,
#     offset: int = 0,
# ) -> np.ndarray:
#   """Makes a local attention mask."""
#   q_seq_len, kv_seq_len = shape
#   q_idx = np.arange(q_seq_len, dtype=np.int32)
#   kv_idx = np.arange(kv_seq_len, dtype=np.int32)
#   mask = np.ones((q_seq_len, kv_seq_len), dtype=np.bool_)
#   left, right = window_size
#   if left is not None:
#     # (q_seq_len, kv_seq_len) & ( (q_seq_len, 1) - left + offset )
    # if left is not None, then Mask cells where (row ix - left + offset) is <= col ix
#     mask = mask & (q_idx[:, None] - left + offset <= kv_idx[None, :])
#   if right is not None:
#     mask = mask & (q_idx[:, None] + right + offset >= kv_idx[None, :])
#   return mask.astype(np.bool_)

