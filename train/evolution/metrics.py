"""Small-world, community, hub and path-length measurements."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import asdict, dataclass

import networkx as nx
from small_world import SmallWorldTopology


@dataclass
class TopologyMetrics:
    nodes: int
    edges: int
    clustering: float
    average_path_length: float
    random_clustering: float
    random_path_length: float
    sigma: float
    log_sigma: float
    modularity: float
    communities: int
    hub_nodes: int
    connected: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _adjacency(topology: SmallWorldTopology) -> list[set[int]]:
    adjacency = [set() for _ in range(topology.num_nodes)]
    for source, target in topology.edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def _average_clustering(adjacency: list[set[int]]) -> float:
    coefficients = []
    for node, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        if degree < 2:
            coefficients.append(0.0)
            continue
        neighbor_links = sum(
            1
            for neighbor in neighbors
            for other in adjacency[neighbor]
            if other in neighbors and neighbor < other
        )
        coefficients.append(2.0 * neighbor_links / (degree * (degree - 1)))
    return sum(coefficients) / max(1, len(coefficients))


def _sampled_path_length(
    adjacency: list[set[int]], sample_sources: int, seed: int
) -> tuple[float, bool]:
    num_nodes = len(adjacency)
    rng = random.Random(seed)
    sources = (
        list(range(num_nodes))
        if sample_sources >= num_nodes
        else rng.sample(range(num_nodes), sample_sources)
    )
    total_distance = 0
    pairs = 0
    connected = True
    for source in sources:
        distances = [-1] * num_nodes
        distances[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        reachable = [distance for distance in distances if distance > 0]
        connected &= len(reachable) == num_nodes - 1
        total_distance += sum(reachable)
        pairs += len(reachable)
    return total_distance / max(1, pairs), connected


def measure_topology(
    topology: SmallWorldTopology,
    sample_sources: int = 64,
    compute_communities: bool = True,
) -> tuple[TopologyMetrics, list[list[int]]]:
    adjacency = _adjacency(topology)
    degrees = [len(neighbors) for neighbors in adjacency]
    mean_degree = sum(degrees) / topology.num_nodes
    clustering = _average_clustering(adjacency)
    path_length, connected = _sampled_path_length(
        adjacency, sample_sources, topology.seed
    )
    random_clustering = mean_degree / max(1.0, topology.num_nodes - 1)
    random_path = math.log(max(2, topology.num_nodes)) / math.log(max(2.0, mean_degree))
    sigma = (clustering / max(random_clustering, 1e-12)) / (
        path_length / max(random_path, 1e-12)
    )

    graph = nx.Graph()
    graph.add_nodes_from(range(topology.num_nodes))
    graph.add_edges_from(topology.edges)
    if compute_communities:
        detected = nx.community.louvain_communities(
            graph, seed=topology.seed, resolution=1.0
        )
        communities = [sorted(community) for community in detected]
        modularity = nx.community.modularity(graph, detected)
    else:
        communities = [list(range(topology.num_nodes))]
        modularity = 0.0
    variance = sum((degree - mean_degree) ** 2 for degree in degrees) / topology.num_nodes
    hub_threshold = mean_degree + 2.0 * math.sqrt(variance)
    hub_nodes = sum(degree >= hub_threshold for degree in degrees)
    metrics = TopologyMetrics(
        nodes=topology.num_nodes,
        edges=len(topology.edges),
        clustering=clustering,
        average_path_length=path_length,
        random_clustering=random_clustering,
        random_path_length=random_path,
        sigma=sigma,
        log_sigma=math.log1p(max(0.0, sigma)),
        modularity=modularity,
        communities=len(communities),
        hub_nodes=hub_nodes,
        connected=connected,
    )
    return metrics, communities


def measure_model_topologies(
    topologies: list[SmallWorldTopology], sample_sources: int = 64
) -> dict[str, object]:
    block_metrics = [
        measure_topology(topology, sample_sources, compute_communities=True)[0]
        for topology in topologies
    ]
    count = len(block_metrics)
    return {
        "small_world_score": sum(metric.log_sigma for metric in block_metrics) / count,
        "sigma": sum(metric.sigma for metric in block_metrics) / count,
        "clustering": sum(metric.clustering for metric in block_metrics) / count,
        "path_length": sum(metric.average_path_length for metric in block_metrics) / count,
        "modularity": sum(metric.modularity for metric in block_metrics) / count,
        "communities": sum(metric.communities for metric in block_metrics) / count,
        "hub_nodes": sum(metric.hub_nodes for metric in block_metrics),
        "connected": all(metric.connected for metric in block_metrics),
        "blocks": [metric.to_dict() for metric in block_metrics],
    }
