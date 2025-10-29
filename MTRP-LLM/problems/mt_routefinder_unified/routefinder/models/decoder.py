# -*- coding: utf-8 -*-
"""
Heuristic-Enhanced AttentionModelDecoder (AR, Pointer)
在原 Autoregressive 解码器基础上按 VRPModel 风格加入启发式偏置：
- 全局常量：ATTENTION_BIAS_encoder / ATTENTION_BIAS_decoder / ATTENTION_BIAS_decoder1
- 偏置命名：attention_bias1/2/3（此处主要用 decoder 侧的 2/3）
"""

from dataclasses import dataclass, fields
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.nn.attention import PointerAttention, PointerAttnMoE
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.ops import batchify, unbatchify

log = get_pylogger(__name__)

# ========= 启发式导入（与工程保持一致） =========
try:
    from gpt import heuristics_v2 as heuristics
except Exception:
    try:
        from gpt import heuristics
    except Exception:
        heuristics = None

try:
    from gpt_encoder import heuristics_encoder as heuristics_encoder
except Exception:
    heuristics_encoder = None

try:
    from gpt_decoder import heuristics_decoder as heuristics_decoder
except Exception:
    heuristics_decoder = None

try:
    from gpt_basic import basic_score_matrix as heuristics_basic
except Exception:
    heuristics_basic = None

try:
    from gpt_basic_tw import heuristics_tw as heuristics_tw
except Exception:
    heuristics_tw = None
# ==============================================

# ========= 全局开关（默认与您一致） =========
ATTENTION_BIAS_encoder  = False   # 编码器静态距离偏置（本文件不直接使用）
ATTENTION_BIAS_decoder  = False   # 解码器静态启发式偏置（一次性）
ATTENTION_BIAS_decoder1 = True    # 解码器动态启发式偏置（逐步）
# =========================================

# ---------- 小工具 ----------
def _td_get(td: TensorDict, *keys, default=None):
    for k in keys:
        if k in td.keys():
            return td[k]
    return default


# ---------- PrecomputedCache（原逻辑保留） ----------
@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Tensor | float
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


# ---------- 启发式上下文构建（一次性） ----------
def _build_heuristics_context(td: TensorDict, env: RL4COEnvBase) -> Optional[Dict[str, Any]]:
    """
    抽取坐标/需求/时间窗/问题类型，并可选计算静态 decoder 偏置 attention_bias2。
    返回:
      {
        "all_nodes_xy": (B,N,2), "all_node_demands": (B,N),
        "all_time_windows": (B,N,2), "problem_type": str,
        "attention_bias2": (B,N) or None
      }
    """
    try:
        # 1) 顶层
        all_nodes_xy = td.get("coords", None)
        if all_nodes_xy is None:
            all_nodes_xy = td.get("locs", None)

        all_node_demands = td.get("demands", None)
        if all_node_demands is None:
            all_node_demands = td.get("demand", None)

        all_time_windows = td.get("time_windows", None)

        # 2) td["static"]
        if (all_nodes_xy is None or all_node_demands is None or all_time_windows is None) and "static" in td.keys():
            static = td["static"]
            if all_nodes_xy is None:
                all_nodes_xy = static.get("locs", static.get("coords", None))
            if all_node_demands is None:
                all_node_demands = static.get("demand", static.get("demands", None))
            if all_time_windows is None:
                all_time_windows = static.get("time_windows", None)

        # 3) env.static 兜底
        if (all_nodes_xy is None or all_node_demands is None) and hasattr(env, "static"):
            try:
                env_static = env.static
                if all_nodes_xy is None:
                    all_nodes_xy = env_static.get("locs", env_static.get("coords", None))
                if all_node_demands is None:
                    all_node_demands = env_static.get("demand", env_static.get("demands", None))
                if all_time_windows is None:
                    all_time_windows = env_static.get("time_windows", None)
            except Exception:
                pass

        if all_nodes_xy is None or all_node_demands is None:
            log.warning("[Heuristics] Missing 'coords/locs' or 'demands/demand' in td/env; skip heuristic biases.")
            return None

        if all_time_windows is None:
            all_time_windows = torch.zeros(
                all_nodes_xy.size(0), all_nodes_xy.size(1), 2, device=all_nodes_xy.device
            )

        # 问题类型（若无，则回退为通用 VRP）
        problem_type = getattr(env, "problem_type", getattr(env, "name", "VRP"))

        # （可选）静态 decoder 偏置：一次性
        attention_bias2 = None
        if ATTENTION_BIAS_decoder:
            try:
                dist = torch.cdist(all_nodes_xy, all_nodes_xy, p=2)  # (B,N,N)
                out = []
                for i in range(all_nodes_xy.size(0)):
                    if heuristics_decoder is not None:
                        out.append(heuristics_decoder(dist[i], all_node_demands[i], all_time_windows[i]))
                    elif heuristics_basic is not None:
                        out.append(heuristics_basic(dist[i], all_node_demands[i]))
                    else:
                        out.append(torch.zeros_like(all_node_demands[i]))
                attention_bias2 = torch.stack(out, dim=0)  # (B,N)
            except Exception as e:
                log.warning(f"[Heuristics] Static decoder bias failed: {e}")
                attention_bias2 = None

        return {
            "all_nodes_xy": all_nodes_xy,
            "all_node_demands": all_node_demands,
            "all_time_windows": all_time_windows,
            "problem_type": problem_type,
            "attention_bias2": attention_bias2,
        }
    except Exception as e:
        log.warning(f"[Heuristics] build context error: {e}")
        return None


# ---------- 动态启发式偏置（逐步） ----------
def _compute_attention_bias3(td: TensorDict, ctx: Dict[str, Any]) -> Optional[Tensor]:
    if ctx is None or not ATTENTION_BIAS_decoder1 or heuristics is None:
        return None

    try:
        all_nodes_xy: Tensor = ctx["all_nodes_xy"]         # (B,N,2)
        all_node_demands: Tensor = ctx["all_node_demands"] # (B,N)
        all_time_windows: Tensor = ctx["all_time_windows"] # (B,N,2)
        problem_type: str = ctx["problem_type"]

        B, N, _ = all_nodes_xy.shape
        device = all_nodes_xy.device

        current_node: Tensor = _td_get(td, "current_node")
        if current_node is None:
            log.warning("[Heuristics] 'current_node' missing in td; skip dynamic bias.")
            return None

        load   = _td_get(td, "load",   default=torch.zeros(B, device=device))
        time_t = _td_get(td, "time",   default=torch.zeros(B, device=device))
        length = _td_get(td, "length", default=torch.zeros(B, device=device))
        # route_open = _td_get(td, "route_open", default=torch.zeros(B, device=device))  # 暂未直接使用

        cur_xy = all_nodes_xy[torch.arange(B, device=device), current_node]  # (B,2)
        current_dist = torch.cdist(cur_xy.unsqueeze(1), all_nodes_xy, p=2).squeeze(1)   # (B,N)

        delivery_node_demands = torch.where(all_node_demands > 0, all_node_demands, torch.tensor(2.0, device=device))
        pickup_node_demands   = torch.where(all_node_demands < 0, all_node_demands, torch.tensor(-2.0, device=device))

        if problem_type == "OVRP":
            delivery_node_demands_open = delivery_node_demands.clone()
            load_open = load.clone()
        else:
            delivery_node_demands_open = torch.zeros_like(delivery_node_demands)
            load_open = torch.zeros_like(load)

        if problem_type == "VRPTW":
            estimated_arrival = time_t.unsqueeze(-1) + current_dist
        else:
            estimated_arrival = torch.zeros_like(current_dist)

        if problem_type == "VRPL":
            current_length = length.clone()
        else:
            current_length = torch.zeros_like(length)

        bias = heuristics(
            current_dist,
            delivery_node_demands,
            load,
            delivery_node_demands_open,
            load_open,
            all_time_windows,
            estimated_arrival,
            pickup_node_demands,
            current_length,
        )

        if "action_mask" in td.keys():
            am = td["action_mask"].bool()
            bias = bias.masked_fill(~am, float("-inf"))

        return bias.clamp(-10.0, 10.0)
    except Exception as e:
        log.warning(f"[Heuristics] dynamic bias error: {e}")
        return None


class RouteFinderDecoder(AutoregressiveDecoder):
    """
    Auto-regressive decoder based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    在原实现基础上，加入启发式偏置（attention_bias2 静态一次性；attention_bias3 动态逐步）。
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
        # ==== 新增：启发式相关参数（与 VRPModel 风格一致） ====
        use_static_decoder_bias: bool = ATTENTION_BIAS_decoder,
        use_dynamic_heuristics: bool = ATTENTION_BIAS_decoder1,
        heuristic_scale: float = 1.0,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        if pointer is None:
            pointer_attn_class = PointerAttention if moe_kwargs is None else PointerAttnMoE
            pointer = pointer_attn_class(
                embed_dim,
                num_heads,
                mask_inner=mask_inner,
                out_bias=out_bias_pointer_attn,
                check_nan=check_nan,
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs,
            )
        self.pointer = pointer

        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=linear_bias)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

        # === 新增：启发式控制属性（缺这个会报你之前看到的错） ===
        self.use_static_decoder_bias = use_static_decoder_bias
        self.use_dynamic_heuristics = use_dynamic_heuristics
        self.heuristic_scale = heuristic_scale

        # 缓存启发式上下文
        self._heur_ctx: Optional[Dict[str, Any]] = None
        self._heuristics_banner_printed = False

    # ----- 原逻辑：Q / KVL -----
    def _compute_q(self, cached: PrecomputedCache, td: TensorDict):
        node_embeds_cache = cached.node_embeddings
        graph_context_cache = cached.graph_context

        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        glimpse_q = step_context + graph_context_cache
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            cached.glimpse_key,
            cached.glimpse_val,
            cached.logit_key,
        )
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn
        return glimpse_k, glimpse_v, logit_k

    # ----- 预解码钩子：加上启发式上下文构建 -----
    def pre_decoder_hook(self, td, env, embeddings, num_starts: int = 0):
        # ---- 新增：属性兜底，防止旧类没有这些字段 ----
        if not hasattr(self, "use_static_decoder_bias"):
            self.use_static_decoder_bias = False
        if not hasattr(self, "use_dynamic_heuristics"):
            self.use_dynamic_heuristics = True
        if not hasattr(self, "heuristic_scale"):
            self.heuristic_scale = 1.0
        if not hasattr(self, "_heur_ctx"):
            self._heur_ctx = None
        if not hasattr(self, "_heuristics_banner_printed"):
            self._heuristics_banner_printed = False
        # --------------------------------------------

        cache = self._precompute_cache(embeddings, num_starts=num_starts)

        # 这里是你构建启发式上下文的地方（如果你已经有函数就继续调用，没有就保持为 None）
        try:
            self._heur_ctx = _build_heuristics_context(td, env)  # 没有的话也可改成 self._heur_ctx = None
        except Exception as e:
            log.warning(f"[Heuristics] pre_decoder_hook context error: {e}")
            self._heur_ctx = None

        if not self._heuristics_banner_printed:
            log.info(
                f"[Heuristics] static={getattr(self, 'use_static_decoder_bias', False)}, "
                f"dynamic={getattr(self, 'use_dynamic_heuristics', True)}, "
                f"scale={getattr(self, 'heuristic_scale', 1.0)}"
            )
            self._heuristics_banner_printed = True

        return td, env, cache

    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        (glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed) = \
            self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

    # ----- 前向：在 pointer logits 上叠加启发式偏置 -----
    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute logits of next actions (with heuristic biases)."""

        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        if has_dyn_emb_multi_start:
            cached = cached.batchify(num_starts=num_starts)
        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        mask = td["action_mask"]
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)  # shape: [B, 1, N]（常见）

        # ---- 新增：属性兜底，防止旧类没有这些字段 ----
        use_static = getattr(self, "use_static_decoder_bias", False)
        use_dynamic = getattr(self, "use_dynamic_heuristics", True)
        scale = getattr(self, "heuristic_scale", 1.0)
        if not hasattr(self, "_heur_ctx"):
            self._heur_ctx = None
        # --------------------------------------------

        # ====== 在 logits 上叠加启发式偏置 ======
        try:
            bias_total = None  # (B,N)

            bias2 = self._heur_ctx.get("attention_bias2", None) if (use_static and self._heur_ctx is not None) else None
            bias3 = _compute_attention_bias3(td, self._heur_ctx) if (
                        use_dynamic and self._heur_ctx is not None) else None

            if bias2 is not None and bias3 is not None:
                bias_total = scale * (bias2 + bias3)
            elif bias2 is not None:
                bias_total = scale * bias2
            elif bias3 is not None:
                bias_total = scale * bias3

            if bias_total is not None:
                logits = logits + bias_total.unsqueeze(1)  # [B,1,N] 对齐广播
        except Exception as e:
            log.warning(f"[Heuristics] add bias failed (safe): {e}")

        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        return logits, mask
