# Nemotron 3 Nano 30B-A3B Architecture



## Structure

Layer map (52 layers total)

  ┌─────────────┬──────────────────────────────────────────────────────────────────┬───────┐
  │    Type     │                             Indices                              │ Count │
  ├─────────────┼──────────────────────────────────────────────────────────────────┼───────┤
  │ Mamba-2 SSM │ 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50  │ 23    │
  ├─────────────┼──────────────────────────────────────────────────────────────────┼───────┤
  │ MoE FFN     │ 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51 │ 23    │
  ├─────────────┼──────────────────────────────────────────────────────────────────┼───────┤
  │ Attention   │ 5, 12, 19, 26, 33, 42                                            │ 6     │
  └─────────────┴──────────────────────────────────────────────────────────────────┴───────┘

  The repeating unit is roughly Mamba → MoE → Mamba → MoE → Mamba → Attention → MoE (7-layer blocks), with attention every ~7 layers. 

  **Each MoE layer has 128 experts with up_proj/down_proj**


backbone.layers.0.mixer.in_proj\
backbone.layers.0.mixer.out_proj\
backbone.layers.1.mixer.experts.0.up_proj\
backbone.layers.1.mixer.experts.0.down_proj\
...
backbone.layers.1.mixer.experts.127.up_proj\
backbone.layers.1.mixer.experts.127.down_proj\
backbone.layers.1.mixer.shared_experts.up_proj\
backbone.layers.1.mixer.shared_experts.down_proj\
backbone.layers.2.mixer.in_proj\
backbone.layers.2.mixer.out_proj\
backbone.layers.3.mixer.experts.0.up_proj\
backbone.layers.3.mixer.experts.0.down_proj\
...
backbone.layers.3.mixer.experts.127.up_proj\
backbone.layers.3.mixer.experts.127.down_proj\
backbone.layers.3.mixer.shared_experts.up_proj\
backbone.layers.3.mixer.shared_experts.down_proj\
backbone.layers.4.mixer.in_proj\
backbone.layers.4.mixer.out_proj\
backbone.layers.5.mixer.q_proj\
backbone.layers.5.mixer.k_proj\
backbone.layers.5.mixer.v_proj\
backbone.layers.5.mixer.o_proj\
backbone.layers.6.mixer.experts.0.up_proj\
backbone.layers.6.mixer.experts.0.down_proj\