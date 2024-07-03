from flax.linen.spmd import logical_to_mesh_axes 


arr_dim_names = ["embed", "heads", "kv"]
logical_rules = [
                      ['activation_batch', ['data', 'fsdp', 'fsdp_transpose',]],
                       # For pipeline parallelism the pre and post decoder layer tensors' batch dimension is sharded by stages.
                       # Microbatches are sharded by stage, so moving out of and into this sharding should be a local reshape.
                       # The "stage" needs to be listed first since the microbatch dimension is first before the reshape.
                      ['activation_embed_and_logits_batch', ['stage', 'data', 'fsdp', 'fsdp_transpose']],
                      ['activation_heads', ['tensor','sequence']],
                      ['activation_kv_heads', ['tensor','sequence']],
                      ['activation_length', 'sequence'],
                      ['activation_embed', 'tensor'],
                      ['activation_mlp', 'tensor'],
                      ['activation_kv', 'tensor'],
                      ['activation_kv_batch', ['data', 'fsdp', 'fsdp_transpose',]],
                      ['activation_kv_head_dim', 'tensor'],
                      ['activation_vocab', ['tensor', 'sequence']],
                      ['activation_vocab', 'tensor'],
                      ['activation_vocab', 'sequence'],
                      ['activation_stage','stage'],
                      ['mlp', ['fsdp_transpose', 'tensor', 'autoregressive']],
                      ['vocab', ['tensor', 'autoregressive']],
                      ['embed', ['fsdp', 'fsdp_transpose', 'sequence']],
                      ['embed', ['fsdp', 'sequence']],
                      ['norm', 'tensor'],
                      ['heads', ['tensor', 'autoregressive']],
                      ['layers', 'stage'],
                      ['kv', []],
                      ['kv_heads', ['tensor', 'autoregressive']],
                      ['kv_head_dim', []],
                      ['cache_batch', []],
                      ['cache_heads', ['autoregressive', 'tensor']],
                      ['cache_kv', []],
                      ['cache_sequence', []],
                    ]

print(logical_to_mesh_axes(arr_dim_names, rules=logical_rules))