"""Persistent JSON/CSV/TensorBoard records and PNG evolution visualizations."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "liquid-transformer-matplotlib")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from metrics import measure_topology  # noqa: E402
from nsga2 import Individual, rank_and_crowding  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402


class EvolutionRecorder:
    def __init__(self, run_dir: Path, model_name: str) -> None:
        self.run_dir = run_dir
        self.model_name = model_name
        self.figure_dir = run_dir / "figures"
        self.topology_dir = self.figure_dir / "topology"
        self.training_dir = self.figure_dir / "training"
        self.genome_dir = run_dir / "genomes"
        for directory in (
            self.figure_dir,
            self.topology_dir,
            self.training_dir,
            self.genome_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(run_dir / "tensorboard")
        self.history: list[dict[str, float | int | str]] = []
        self.block_history: list[dict[str, object]] = []
        self.previous_best_edges: list[set[tuple[int, int]]] | None = None
        self.csv_path = run_dir / "population.csv"
        self.jsonl_path = run_dir / "evolution.jsonl"

    def record_generation(
        self, generation: int, phase: str, population: list[Individual]
    ) -> None:
        fronts = rank_and_crowding(population)
        front_ids = {item.identifier for item in fronts[0]}
        rows = []
        for individual in population:
            metrics = individual.topology_metrics
            row = {
                "generation": generation,
                "phase": phase,
                "individual": individual.identifier,
                "parents": ",".join(individual.parent_ids),
                "accuracy": float(individual.accuracy),
                "loss": float(individual.loss),
                "small_world_score": individual.small_world_score,
                "sigma": float(metrics["sigma"]),
                "clustering": float(metrics["clustering"]),
                "path_length": float(metrics["path_length"]),
                "modularity": float(metrics["modularity"]),
                "communities": float(metrics["communities"]),
                "hub_nodes": int(metrics["hub_nodes"]),
                "rank": individual.rank,
                "crowding": individual.crowding,
                "pareto": individual.identifier in front_ids,
                "runtime_seconds": individual.runtime_seconds,
                "peak_memory_gb": individual.peak_memory_gb,
            }
            rows.append(row)
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            for epoch, values in enumerate(individual.training_history):
                step = generation * 10_000 + epoch
                prefix = f"candidate/{individual.identifier}"
                for key, value in values.items():
                    self.writer.add_scalar(f"{prefix}/{key}", value, step)

        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        best_accuracy = max(population, key=lambda item: float(item.accuracy))
        current_edges = [set(topology.edges) for topology in best_accuracy.topologies]
        if self.previous_best_edges is None:
            edge_change_rates = [0.0] * len(current_edges)
        else:
            edge_change_rates = [
                len(current.symmetric_difference(previous))
                / max(1, len(current.union(previous)))
                for current, previous in zip(
                    current_edges, self.previous_best_edges, strict=True
                )
            ]
        self.previous_best_edges = current_edges
        summary = {
            "generation": generation,
            "phase": phase,
            "best_accuracy": max(float(item.accuracy) for item in population),
            "mean_accuracy": sum(float(item.accuracy) for item in population)
            / len(population),
            "best_small_world": max(item.small_world_score for item in population),
            "mean_small_world": sum(item.small_world_score for item in population)
            / len(population),
            "best_loss": min(float(item.loss) for item in population),
            "mean_loss": sum(float(item.loss) for item in population) / len(population),
            "mean_clustering": sum(
                float(item.topology_metrics["clustering"]) for item in population
            )
            / len(population),
            "mean_path_length": sum(
                float(item.topology_metrics["path_length"]) for item in population
            )
            / len(population),
            "mean_modularity": sum(
                float(item.topology_metrics["modularity"]) for item in population
            )
            / len(population),
            "pareto_size": len(fronts[0]),
            "mean_edge_change_rate": sum(edge_change_rates) / len(edge_change_rates),
        }
        self.history.append(summary)
        for key, value in summary.items():
            if key not in {"generation", "phase"}:
                self.writer.add_scalar(f"generation/{key}", value, generation)
        self.writer.flush()

        pareto_records = [item.record_dict() for item in fronts[0]]
        genome_path = self.genome_dir / f"generation-{generation:03d}-pareto.json"
        genome_path.write_text(
            json.dumps(pareto_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._plot_pareto(generation, population, front_ids)
        best_topology = max(population, key=lambda item: item.small_world_score)
        block_records = [
            {**block, "edge_change_rate": edge_change_rates[index]}
            for index, block in enumerate(best_accuracy.topology_metrics["blocks"])
        ]
        self.block_history.append(
            {
                "generation": generation,
                "blocks": block_records,
            }
        )
        for block_index, metrics in enumerate(block_records):
            for key in (
                "clustering",
                "average_path_length",
                "sigma",
                "modularity",
                "edge_change_rate",
            ):
                self.writer.add_scalar(
                    f"topology/block-{block_index:02d}/{key}",
                    metrics[key],
                    generation,
                )
        self._plot_history()
        self._plot_block_history()
        self._plot_training_history(generation, best_accuracy)
        self._plot_topology_snapshot(generation, best_accuracy, "best-accuracy")
        if best_topology.identifier != best_accuracy.identifier:
            self._plot_topology_snapshot(generation, best_topology, "best-small-world")

    def _plot_pareto(
        self, generation: int, population: list[Individual], front_ids: set[str]
    ) -> None:
        figure, axis = plt.subplots(figsize=(7.5, 5.5))
        regular = [item for item in population if item.identifier not in front_ids]
        pareto = [item for item in population if item.identifier in front_ids]
        axis.scatter(
            [item.small_world_score for item in regular],
            [item.accuracy for item in regular],
            alpha=0.65,
            color="#4472c4",
            label="population",
        )
        axis.scatter(
            [item.small_world_score for item in pareto],
            [item.accuracy for item in pareto],
            s=80,
            color="#d62728",
            edgecolor="black",
            label="Pareto front",
        )
        axis.set_xlabel("small-world score = mean(log(1 + sigma))")
        axis.set_ylabel("validation accuracy (%)")
        axis.set_title(f"{self.model_name} generation {generation}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.figure_dir / f"pareto-generation-{generation:03d}.png", dpi=180)
        plt.close(figure)

    def _plot_training_history(self, generation: int, individual: Individual) -> None:
        epochs = list(range(1, len(individual.training_history) + 1))
        if not epochs:
            return
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(
            epochs,
            [record["train_accuracy"] for record in individual.training_history],
            label="train accuracy",
        )
        axes[0].plot(
            epochs,
            [record["validation_accuracy"] for record in individual.training_history],
            label="validation accuracy",
        )
        axes[0].set_ylabel("accuracy (%)")
        axes[1].plot(
            epochs,
            [record["train_loss"] for record in individual.training_history],
            label="train loss",
        )
        axes[1].plot(
            epochs,
            [record["validation_loss"] for record in individual.training_history],
            label="validation loss",
        )
        axes[1].set_ylabel("loss")
        for axis in axes:
            axis.set_xlabel("epoch")
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle(
            f"generation {generation} best-accuracy candidate {individual.identifier}"
        )
        figure.tight_layout()
        figure.savefig(
            self.training_dir / f"generation-{generation:03d}-best-accuracy.png",
            dpi=180,
        )
        plt.close(figure)

    def _plot_block_history(self) -> None:
        figure, axes = plt.subplots(2, 3, figsize=(16, 8))
        fields = (
            ("clustering", "Clustering coefficient"),
            ("average_path_length", "Average path length"),
            ("sigma", "Small-world sigma"),
            ("modularity", "Louvain modularity"),
            ("communities", "Community count"),
            ("edge_change_rate", "Edge change rate vs previous best"),
        )
        generations = [int(record["generation"]) for record in self.block_history]
        for axis, (field, title) in zip(axes.flat, fields, strict=True):
            values = np.asarray(
                [
                    [float(block[field]) for block in record["blocks"]]
                    for record in self.block_history
                ]
            )
            image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis")
            axis.set_title(title)
            axis.set_xlabel("Transformer block")
            axis.set_ylabel("generation")
            axis.set_yticks(range(len(generations)), labels=generations)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(
            f"{self.model_name} topology evolution of generation-best accuracy candidates"
        )
        figure.tight_layout()
        figure.savefig(self.figure_dir / "block-topology-evolution.png", dpi=180)
        plt.close(figure)

    def _plot_history(self) -> None:
        generations = [int(item["generation"]) for item in self.history]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        series = (
            ("Accuracy (%)", "best_accuracy", "mean_accuracy"),
            ("Loss", "best_loss", "mean_loss"),
            ("Small-world score", "best_small_world", "mean_small_world"),
            ("Topology metrics", "mean_clustering", "mean_modularity"),
        )
        for axis, (title, best_key, mean_key) in zip(axes.flat, series, strict=True):
            axis.plot(generations, [item[best_key] for item in self.history], label=best_key)
            axis.plot(generations, [item[mean_key] for item in self.history], label=mean_key)
            axis.set_title(title)
            axis.set_xlabel("generation")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.suptitle(f"{self.model_name} NSGA-II evolution")
        figure.tight_layout()
        figure.savefig(self.figure_dir / "evolution-history.png", dpi=180)
        plt.close(figure)

    def _plot_topology_snapshot(
        self, generation: int, individual: Individual, label: str
    ) -> None:
        block_indices = sorted({0, len(individual.topologies) // 2, len(individual.topologies) - 1})
        figure, axes = plt.subplots(1, len(block_indices), figsize=(6 * len(block_indices), 5))
        axes = np.atleast_1d(axes)
        for axis, block_index in zip(axes, block_indices, strict=True):
            topology = individual.topologies[block_index]
            metrics, communities = measure_topology(topology, sample_sources=32)
            membership = {
                node: community_index
                for community_index, community in enumerate(communities)
                for node in community
            }
            quotient = nx.Graph()
            for index, community in enumerate(communities):
                quotient.add_node(index, size=len(community))
            for source, target in topology.edges:
                left, right = membership[source], membership[target]
                if left == right:
                    continue
                weight = quotient.get_edge_data(left, right, {}).get("weight", 0) + 1
                quotient.add_edge(left, right, weight=weight)
            positions = nx.spring_layout(quotient, seed=topology.seed, weight="weight")
            sizes = [80 + 12 * quotient.nodes[node]["size"] ** 0.7 for node in quotient]
            widths = [
                0.3 + 0.15 * math.log1p(data["weight"])
                for *_, data in quotient.edges(data=True)
            ]
            nx.draw_networkx_nodes(
                quotient,
                positions,
                node_size=sizes,
                node_color=list(quotient.nodes),
                cmap="viridis",
                alpha=0.9,
                ax=axis,
            )
            nx.draw_networkx_edges(
                quotient, positions, width=widths, alpha=0.45, edge_color="#555555", ax=axis
            )
            axis.set_title(
                f"block {block_index} community graph\n"
                f"C={metrics.clustering:.3f}, L={metrics.average_path_length:.2f}, "
                f"σ={metrics.sigma:.1f}\nQ={metrics.modularity:.3f}, "
                f"communities={metrics.communities}, hubs={metrics.hub_nodes}"
            )
            axis.axis("off")
        figure.suptitle(
            f"generation {generation} · {label} · {individual.identifier}\n"
            "node size = community size; edge width = inter-community connections"
        )
        figure.tight_layout()
        figure.savefig(
            self.topology_dir / f"generation-{generation:03d}-{label}.png", dpi=180
        )
        plt.close(figure)

    def close(self) -> None:
        self.writer.close()
