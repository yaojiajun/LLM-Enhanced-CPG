import torch

from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

log = get_pylogger(__name__)


def render(
    td: TensorDict, actions=None, ax=None, scale_xy: bool = True, vehicle_capacity=None
):
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib import cm, colormaps

    num_routine = (actions == 0).sum().item() + 2
    base = colormaps["nipy_spectral"]
    color_list = base(np.linspace(0, 1, num_routine))
    cmap_name = base.name + str(num_routine)
    out = base.from_list(cmap_name, color_list, num_routine)

    if ax is None:
        _, ax = plt.subplots(dpi=100, figsize=(6, 6))

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]
    scale_demand = td["capacity_original"]
    demands_linehaul = td["demand_linehaul"] * scale_demand
    demands_backhaul = td["demand_backhaul"] * scale_demand
    num_depots = td["num_depots"]

    # scale to closest integer
    if demands_linehaul.max() <= 1:  # fallback for no scaling
        # scale min value except 0 to 1 and max value to 9
        demands_linehaul = (
            (demands_linehaul - demands_linehaul.min())
            / (demands_linehaul.max() - demands_linehaul.min())
            * 9
        )
        demands_backhaul = (
            (demands_backhaul - demands_backhaul.min())
            / (demands_backhaul.max() - demands_backhaul.min())
            * 9
        )

        demands_linehaul = demands_linehaul.round().int()
        demands_backhaul = demands_backhaul.round().int()

    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

    # Depot
    for idx in range(num_depots):
        ax.scatter(
            locs[idx, 0],
            locs[idx, 1],
            edgecolors=cm.Set2(2),
            facecolors="none",
            s=100,
            linewidths=2,
            marker="s",
            alpha=1,
        )

    for node_idx, loc in enumerate(locs):
        if node_idx < num_depots:
            continue  # skip depots
        delivery, pickup = demands_linehaul[node_idx], demands_backhaul[node_idx]
        if delivery > 0:
            ax.text(
                loc[0],
                loc[1] + 0.02,
                f"{delivery.item()}",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=10,
                color=cm.Set2(0),
            )
            # scatter delivery as downward triangle
            ax.scatter(
                loc[0],
                loc[1],
                edgecolors=cm.Set2(0),
                facecolors="none",
                s=30,
                linewidths=2,
                marker="v",
                alpha=1,
            )
        elif pickup > 0:
            ax.text(
                loc[0],
                loc[1] - 0.02,
                f"{pickup.item()}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color=cm.Set2(1),
            )
            ax.scatter(
                loc[0],
                loc[1],
                edgecolors=cm.Set2(1),
                facecolors="none",
                s=30,
                linewidths=2,
                marker="^",
                alpha=1,
            )
        else:
            print("Error: no demand")

    color_idx = 0
    next_actions = torch.roll(actions, -1, 0)
    # TODO: plot for MDVRP should consider actions going back to depot
    assert actions[0] < num_depots, "First action should be a depot"
    current_depot = actions[0]
    for ai, aj in zip(actions, next_actions):
        # if open and next is a depot, skip
        if td["open_route"].item() and aj < num_depots:
            continue
        if ai < num_depots:
            current_depot = ai
            color_idx += 1
        from_loc = locs[ai]
        # if aj is a depot, then actually we need to plot back to the current depot
        if aj < num_depots:
            to_loc = locs[current_depot]
        else:
            to_loc = locs[aj]
        # if any of from_loc or to_loc is depot, change color and linewidth
        if ai < num_depots or aj < num_depots:
            color, lw, alpha, style = "lightgrey", 1, 0.5, "--"
        else:
            color, lw, alpha, style = out(color_idx), 1, 1, ""
        ax.plot(
            [from_loc[0], to_loc[0]],
            [from_loc[1], to_loc[1]],
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=style,
        )
        ax.annotate(
            "",
            xy=(to_loc[0], to_loc[1]),
            xytext=(from_loc[0], from_loc[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha),
            size=15,
            annotation_clip=False,
        )

    if scale_xy:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
