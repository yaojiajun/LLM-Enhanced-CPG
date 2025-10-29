import torch
import torch.nn as nn
import torch.nn.functional as F
# from tutel import moe as tutel_moe
from .MOELayer import MoE

# === Heuristics import & flags ===
try:
    from gpt import heuristics_v2 as heuristics
except Exception:
    from gpt import heuristics

# 与 VRPModel 保持一致的三个开关
ATTENTION_BIAS_encoder = False      # 本文件未在 Encoder 接口中使用，建议保持 False
ATTENTION_BIAS_decoder = False      # 静态启发式（图级）
ATTENTION_BIAS_decoder1 = True      # 动态启发式（基于当前节点，默认开启）

__all__ = ['MOEModel']


class MOEModel(nn.Module):
    """
        Mixture-of-Experts model for multi-task routing with heuristic-enhanced decoder.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']
        self.aux_loss = 0

        self.encoder = MTL_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)
        self.encoded_nodes = None  # shape: (B, N+1, E)
        self.device = (torch.device('cuda', torch.cuda.current_device())
                       if 'device' not in model_params.keys() else model_params['device'])

        # === 缓存：与 VRPModel 对齐，供启发式使用 ===
        self.all_nodes_xy = None           # (B, N+1, 2)
        self.all_node_demands = None       # (B, N+1)
        self.all_time_windows = None       # (B, N+1, 2)
        # 若未提供 problem_type，则回退使用 self.problem
        self.problem_type = self.model_params.get('problem_type', self.problem)
        self.attention_bias2 = None        # 静态启发式 (B, N+1, N+1)

    def pre_forward(self, reset_state):
        """
        编码 + 预缓存图信息（坐标/需求/TW），以及可选的静态启发式 bias。
        """
        depot_xy = reset_state.depot_xy              # (B, 1, 2)
        node_xy = reset_state.node_xy                # (B, N, 2)
        node_demand = reset_state.node_demand        # (B, N)
        node_tw_start = reset_state.node_tw_start    # (B, N)
        node_tw_end = reset_state.node_tw_end        # (B, N)

        # === 缓存启发式所需的图信息 ===
        all_nodes_xy = torch.cat((depot_xy, node_xy), dim=1)  # (B, N+1, 2)
        self.all_nodes_xy = all_nodes_xy

        all_node_demands = torch.cat(
            (torch.zeros_like(node_demand[:, :1]), node_demand), dim=1
        )  # (B, N+1)
        self.all_node_demands = all_node_demands

        depot_tw = torch.zeros((depot_xy.size(0), 1, 2), device=depot_xy.device)  # depot TW=0
        node_TW = torch.stack((node_tw_start, node_tw_end), dim=2)                # (B, N, 2)
        all_time_windows = torch.cat((depot_tw, node_TW), dim=1)                  # (B, N+1, 2)
        self.all_time_windows = all_time_windows

        # === 可选：静态启发式 bias（图级）===
        if ATTENTION_BIAS_decoder:
            distance_matrices = torch.cdist(all_nodes_xy, all_nodes_xy, p=2)  # (B, N+1, N+1)
            self.attention_bias2 = torch.stack([
                heuristics(distance_matrices[i], all_node_demands[i], all_time_windows[i])
                for i in range(all_nodes_xy.size(0))
            ], dim=0)  # (B, N+1, N+1)
            assert not torch.isnan(self.attention_bias2).any()
            assert not torch.isinf(self.attention_bias2).any()
        else:
            self.attention_bias2 = None

        # === 编码 ===
        node_xy_demand_tw = torch.cat(
            (node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2
        )  # (B, N, 5)

        self.encoded_nodes, moe_loss = self.encoder(depot_xy, node_xy_demand_tw)
        self.aux_loss = moe_loss
        self.decoder.set_kv(self.encoded_nodes)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None):
        """
        解码：在第 2 步之后，构造动态启发式 bias；并与静态 bias 一并注入解码器得分。
        """
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First move: depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=self.device)
            prob = torch.ones(size=(batch_size, pomo_size), device=self.device)

        elif state.selected_count == 1:  # Second move: POMO start nodes
            selected = state.START_NODE  # (B, P)
            prob = torch.ones(size=(batch_size, pomo_size), device=self.device)

        else:
            # (B,P,E)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)

            # (B,P,4): load, time, length, open
            attr = torch.cat((
                state.load[:, :, None],
                state.current_time[:, :, None],
                state.length[:, :, None],
                state.open[:, :, None]
            ), dim=2)

            # === 构造解码器启发式 bias ===
            # 静态 bias: 从 (B, N+1, N+1) 取当前节点对应行 -> (B, P, N+1)
            if ATTENTION_BIAS_decoder and (self.attention_bias2 is not None):
                attention_bias_static = self.attention_bias2[
                    torch.arange(batch_size, device=self.device).unsqueeze(1),  # (B,1)
                    state.current_node,                                         # (B,P)
                    :
                ]  # (B, P, N+1)
            else:
                attention_bias_static = None

            # 动态 bias: 依赖当前节点与所有候选的距离、时间、载荷、TW 等
            if ATTENTION_BIAS_decoder1:
                batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, pomo_size)  # (B,P)
                selected_coords = self.all_nodes_xy[batch_idx, state.current_node]   # (B,P,2)
                customer_coords = self.all_nodes_xy                                   # (B,N+1,2)
                current_distance_matrix = torch.cdist(selected_coords, customer_coords, p=2)  # (B,P,N+1)

                # 拷贝当前状态（避免原地修改）
                load = state.load.clone()
                all_time_windows = self.all_time_windows.clone()
                current_time = state.current_time.clone()
                current_length = state.length.clone()

                # VRPTW: 估计到达时间 = 当前时间 + 距离；其他问题置零
                if self.problem_type == 'VRPTW':
                    estimated_arrival = current_time.unsqueeze(-1) + current_distance_matrix  # (B,P,N+1)
                else:
                    estimated_arrival = torch.zeros_like(current_distance_matrix)

                # 需求张量（与 VRPModel 保持一致的约定）
                two_scalar = torch.tensor(2.0, device=self.device, dtype=self.all_node_demands.dtype)
                neg_two_scalar = torch.tensor(-2.0, device=self.device, dtype=self.all_node_demands.dtype)

                # pickup：仅 VRPB 使用真实负需求；否则为 -2（低优先）
                if self.problem_type == 'VRPB':
                    pickup_node_demands = torch.zeros_like(self.all_node_demands)  # (B,N+1)
                else:
                    pickup_node_demands = torch.where(
                        self.all_node_demands < 0, self.all_node_demands, neg_two_scalar
                    )

                # open-route：仅 OVRP 使用 open 版本
                if self.problem_type == 'OVRP':
                    delivery_node_demands_open = self.all_node_demands.clone()
                    load_open = load.clone()
                else:
                    delivery_node_demands_open = torch.zeros_like(self.all_node_demands)
                    load_open = torch.zeros_like(load)

                # path-length：仅 VRPL 使用当前长度；其余置零以避免影响
                if self.problem_type == 'VRPL':
                    current_length = state.length.clone()
                else:
                    current_length = torch.zeros_like(state.length)

                # delivery：正需求，否则置 2（低优先）
                delivery_node_demands = torch.where(
                    self.all_node_demands > 0, self.all_node_demands, two_scalar
                )

                # 调用启发式（按 batch 维拼接）
                attention_bias_dynamic = torch.stack([
                    heuristics(
                        current_distance_matrix[i],        # (P,N+1)
                        delivery_node_demands[i],          # (N+1)
                        load[i],                           # (P)
                        delivery_node_demands_open[i],     # (N+1)
                        load_open[i],                      # (P)
                        all_time_windows[i],               # (N+1,2)
                        estimated_arrival[i],              # (P,N+1)
                        pickup_node_demands[i],            # (N+1)
                        current_length[i]                  # (P)
                    )
                    for i in range(batch_size)
                ], dim=0)  # (B,P,N+1)

                assert not torch.isnan(attention_bias_dynamic).any()
                assert not torch.isinf(attention_bias_dynamic).any()
            else:
                attention_bias_dynamic = None

            # === 解码 ===
            probs, moe_loss = self.decoder(
                encoded_last_node, attr, ninf_mask=state.ninf_mask,
                attention_bias=attention_bias_static,
                attention_bias1=attention_bias_dynamic
            )
            self.aux_loss += moe_loss

            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    """
    encoded_nodes: (B, N+1, E)
    node_index_to_pick: (B, P)
    return: (B, P, E)
    """
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


########################################
# ENCODER
########################################

class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # [Option 1]: MoE on raw features
        if self.model_params['num_experts'] > 1 and "Raw" in self.model_params['expert_loc']:
            self.embedding_depot = MoE(
                input_size=2, output_size=embedding_dim,
                num_experts=self.model_params['num_experts'],
                k=self.model_params['topk'], T=1.0, noisy_gating=True,
                routing_level=self.model_params['routing_level'],
                routing_method=self.model_params['routing_method'],
                moe_model="Linear"
            )
            self.embedding_node = MoE(
                input_size=5, output_size=embedding_dim,
                num_experts=self.model_params['num_experts'],
                k=self.model_params['topk'], T=1.0, noisy_gating=True,
                routing_level=self.model_params['routing_level'],
                routing_method=self.model_params['routing_method'],
                moe_model="Linear"
            )
        else:
            self.embedding_depot = nn.Linear(2, embedding_dim)
            self.embedding_node = nn.Linear(5, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(i, **model_params) for i in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_tw):
        """
        depot_xy: (B,1,2)
        node_xy_demand_tw: (B,N,5)
        """
        moe_loss = 0
        if isinstance(self.embedding_depot, MoE) or isinstance(self.embedding_node, MoE):
            embedded_depot, loss_depot = self.embedding_depot(depot_xy)              # (B,1,E)
            embedded_node, loss_node = self.embedding_node(node_xy_demand_tw)       # (B,N,E)
            moe_loss = moe_loss + loss_depot + loss_node
        else:
            embedded_depot = self.embedding_depot(depot_xy)
            embedded_node = self.embedding_node(node_xy_demand_tw)

        out = torch.cat((embedded_depot, embedded_node), dim=1)  # (B,N+1,E)

        for layer in self.layers:
            out, loss = layer(out)
            moe_loss = moe_loss + loss

        return out, moe_loss


class EncoderLayer(nn.Module):
    def __init__(self, depth=0, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)

        # [Option 2]: MoE in Encoder FFN
        if self.model_params['num_experts'] > 1 and f"Enc{depth}" in self.model_params['expert_loc']:
            self.feedForward = MoE(
                input_size=embedding_dim, output_size=embedding_dim,
                num_experts=self.model_params['num_experts'],
                hidden_size=self.model_params['ff_hidden_dim'],
                k=self.model_params['topk'], T=1.0, noisy_gating=True,
                routing_level=self.model_params['routing_level'],
                routing_method=self.model_params['routing_method'],
                moe_model="MLP"
            )
        else:
            self.feedForward = FeedForward(**model_params)

        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        """
        input1: (B, N+1, E)
        两种规范可选：norm_last (AM/POMO) 或 norm_first (NLP 常用)
        """
        head_num, moe_loss = self.model_params['head_num'], 0

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)           # (B, N+1, H*D)
            multi_head_out = self.multi_head_combine(out_concat) # (B, N+1, E)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2, moe_loss = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)         # (B, N+1, E)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2, moe_loss = self.feedForward(out2)
            out3 = input2 + out2

        return out3, moe_loss


########################################
# DECODER
########################################

class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # [Option 3]: MoE in Decoder combine
        if self.model_params['num_experts'] > 1 and 'Dec' in self.model_params['expert_loc']:
            self.multi_head_combine = MoE(
                input_size=head_num * qkv_dim, output_size=embedding_dim,
                num_experts=self.model_params['num_experts'],
                k=self.model_params['topk'], T=1.0, noisy_gating=True,
                routing_level=self.model_params['routing_level'],
                routing_method=self.model_params['routing_method'],
                moe_model="Linear"
            )
        else:
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # (B,H,N+1,D)
        self.v = None  # (B,H,N+1,D)
        self.single_head_key = None  # (B,E,N+1)

    def set_kv(self, encoded_nodes):
        # encoded_nodes: (B, N+1, E)
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, attr, ninf_mask, attention_bias=None, attention_bias1=None):
        """
        encoded_last_node: (B,P,E)
        attr: (B,P,4)   -> [load, time, length, open]
        ninf_mask: (B,P,N+1)
        attention_bias:  (B,P,N+1)  # 静态启发式
        attention_bias1: (B,P,N+1)  # 动态启发式
        """
        head_num, moe_loss = self.model_params['head_num'], 0

        input_cat = torch.cat((encoded_last_node, attr), dim=2)      # (B,P,E+4)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
        if isinstance(self.multi_head_combine, MoE):
            mh_atten_out, moe_loss = self.multi_head_combine(out_concat)   # (B,P,E)
        else:
            mh_atten_out = self.multi_head_combine(out_concat)             # (B,P,E)

        score = torch.matmul(mh_atten_out, self.single_head_key)     # (B,P,N+1)

        # === 启发式 bias 叠加（先加再缩放/裁剪/掩码）===
        if attention_bias is not None:
            score = score + attention_bias
        if attention_bias1 is not None:
            score = score + attention_bias1

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)                       # (B,P,N+1)
        return probs, moe_loss


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # qkv: (B, n, head_num*D)   n ∈ {1, N+1, P}
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)  # (B,H,n,D)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """
    q: (B,H,n,D), k,v: (B,H,N+1,D)
    rank2_ninf_mask: (B, N+1)
    rank3_ninf_mask: (B, n, N+1)
    """
    batch_s, head_num, n, key_dim = q.shape
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))               # (B,H,n,N+1)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float32, device=q.device))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = F.softmax(score_scaled, dim=3)                 # (B,H,n,N+1)
    out = torch.matmul(weights, v)                            # (B,H,n,D)
    out_transposed = out.transpose(1, 2)                      # (B,n,H,D)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # (B,n,H*D)
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if model_params.get('norm_loc', 'norm_last') == "norm_last" else False
        norm_type = model_params.get("norm", "instance")

        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif norm_type == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif norm_type == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # 输入 shape: (B, N+1, E)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)                  # (B,E,N+1)
            normalized = self.norm(transposed)                  # (B,E,N+1)
            back_trans = normalized.transpose(1, 2)             # (B,N+1,E)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            B, Np1, E = added.size()
            normalized = self.norm(added.reshape(B * Np1, E))
            back_trans = normalized.reshape(B, Np1, E)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2
        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input1: (B, N+1, E)
        return self.W2(F.relu(self.W1(input1))), 0
