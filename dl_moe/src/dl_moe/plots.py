from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .analysis import load_results


def plot_specialization(results: list[dict], output: str | Path) -> None:
    grouped = {}
    for result in results:
        value = result.get("final", {}).get("causal_specialization_general")
        if value is not None:
            grouped.setdefault((result["architecture"], result["condition"]), []).append(value)
    labels, values = [], []
    for key in sorted(grouped):
        labels.append(f"{key[0]}\n{key[1]}")
        values.append(grouped[key])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel("causal_specialization_general")
    ax.set_title("DL-MoE smoke diagnostic")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    plot_specialization(load_results(args.directory), args.output)


if __name__ == "__main__":
    main()
