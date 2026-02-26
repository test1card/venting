from pathlib import Path

HAS_MPL = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


def plot_basic(outdir: Path, name: str, res, node_idx: int = 0) -> None:
    if not HAS_MPL:
        return
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(res.t, (res.P[node_idx] - res.P_ext) / 1e3, lw=2, label=f"ΔP(node{node_idx}→ext) [kPa]")
    ax.set(xlabel="t, s", ylabel="ΔP, kPa", title=f"{name}: ΔP vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_dP.png", dpi=200)
    plt.close(fig)
