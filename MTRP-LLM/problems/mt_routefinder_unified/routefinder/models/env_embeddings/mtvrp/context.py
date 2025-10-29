import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = step_context_dim if step_context_dim is not None else embed_dim
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class MTVRPContextEmbedding(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - open route
    - remaining distance (set to default_remain_dist if positive infinity)

    Note that the distance limit (L) and open routes (O) are only embedding during decoding
    in this version
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(MTVRPContextEmbedding, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 4
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        available_load = td["vehicle_capacity"] - used_capacity
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        context_feats = torch.cat(
            (
                available_load,
                td["current_time"],
                td["open_route"].float(),
                remaining_dist,
            ),
            -1,
        )
        return context_feats


class RouteFinderContextEmbedding(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)

    We do not need to embed the open route here since it is done encoder-side.
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(RouteFinderContextEmbedding, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 3
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        available_load = td["vehicle_capacity"] - used_capacity
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        context_feats = torch.cat(
            (
                available_load,
                td["current_time"],
                remaining_dist,
            ),
            -1,
        )
        return context_feats


# Simple wrapper for better naming. We can replace with this name after getting
# the final models
class MTVRPContextEmbeddingRouteFinder(RouteFinderContextEmbedding):
    def __init__(self, *args, **kwargs):
        super(MTVRPContextEmbeddingRouteFinder, self).__init__(*args, **kwargs)


class MTVRPContextEmbeddingM(MTVRPContextEmbeddingRouteFinder):
    """Context embedding MTVRP with mixed backhaul.
    This is for the zero-shot or few-short on backhaul_class 2 instances.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 4)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingM, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        ) * (td["backhaul_class"] == 2)

        # Note: now we need the projection to have embed_dim + 4 features!
        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
            ),
            -1,
        )


class MTVRPContextEmbeddingFull(MTVRPContextEmbeddingRouteFinder):
    """Context embedding MTVRP with full features, including the Mixed Backhaul (MB) variants
    and Multi-depot (MD) variants as well.
    In practice, we use the same features as the MTVRPContextEmbeddingRouteFinder, but we add:
    - available load for MB variants (backhaul class 2)
    - locations of the depot where we started the route, since we may need to return there
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 4 + 2)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingFull, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        ) * (td["backhaul_class"] == 2)

        start_depot_location = gather_by_index(td["locs"], td["current_depot"], dim=-2)

        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
                start_depot_location,
            ),
            -1,
        )
