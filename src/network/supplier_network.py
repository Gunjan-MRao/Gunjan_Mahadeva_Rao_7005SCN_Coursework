"""
Phase 4 extension: buyer-supplier network analysis.

A bipartite graph estimates supplier degree and betweenness centrality; hubs
are suppliers serving at least two buyers or in the highest configured
transaction-volume percentile. A supplier co-occurrence projection is partitioned
using greedy modularity community detection to identify recurring procurement pools.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_COOCCURRENCE_GROUP_SIZE = 40  # Exclude generic buyer-category-month pools that obscure meaningful co-occurrence.


def load_trust_records() -> pd.DataFrame:
    panel = pd.read_csv(
        config.MASTER_PANEL_PATH, low_memory=False,
        usecols=["record_type", "entity_norm", "supplier_norm", "amount", "category", "year_month"],
    )
    return panel[panel["record_type"] == "trust_spend"].dropna(subset=["entity_norm", "supplier_norm"])


def build_bipartite_graph(trust: pd.DataFrame) -> nx.Graph:
    edges = trust.groupby(["entity_norm", "supplier_norm"]).agg(
        n_transactions=("amount", "size"), total_amount=("amount", "sum"),
    ).reset_index()
    edges = edges[edges["n_transactions"] >= config.NETWORK_MIN_TRANSACTIONS]

    G = nx.Graph()
    for entity in edges["entity_norm"].unique():
        G.add_node(entity, node_type="buyer")
    for supplier in edges["supplier_norm"].unique():
        G.add_node(supplier, node_type="supplier")
    for _, row in edges.iterrows():
        G.add_edge(row["entity_norm"], row["supplier_norm"],
                    weight=int(row["n_transactions"]), total_amount=float(row["total_amount"]))

    logger.info("Bipartite buyer-supplier graph: %d nodes (%d buyers, %d suppliers), %d edges",
                G.number_of_nodes(), edges["entity_norm"].nunique(), edges["supplier_norm"].nunique(), G.number_of_edges())
    return G


def compute_bipartite_centrality(G: nx.Graph) -> pd.DataFrame:
    degree_centrality = nx.degree_centrality(G)
    # Approximate betweenness through seeded source-node sampling for computational tractability.
    k_sample = min(1000, G.number_of_nodes())
    betweenness_centrality = nx.betweenness_centrality(G, k=k_sample, normalized=True, seed=config.RANDOM_STATE)

    rows = []
    for node, data in G.nodes(data=True):
        rows.append({
            "node": node, "node_type": data["node_type"],
            "degree": G.degree(node),
            "degree_centrality": degree_centrality[node],
            "betweenness_centrality": betweenness_centrality[node],
            "weighted_degree_n_transactions": sum(d["weight"] for _, _, d in G.edges(node, data=True)),
            "total_amount": sum(d["total_amount"] for _, _, d in G.edges(node, data=True)),
        })
    node_df = pd.DataFrame(rows)

    # Sparse two-mode structure makes betweenness degenerate for degree-one suppliers.
    # Define hubs as multi-buyer suppliers (degree >=2) or high-volume suppliers in
    # the configured top transaction-count percentile, capturing reach and intensity.
    supplier_mask = node_df["node_type"] == "supplier"
    node_df["is_multi_trust_supplier"] = supplier_mask & (node_df["degree"] >= 2)

    volume_threshold = np.percentile(node_df.loc[supplier_mask, "weighted_degree_n_transactions"], config.NETWORK_HUB_PERCENTILE)
    node_df["is_high_volume_supplier"] = supplier_mask & (node_df["weighted_degree_n_transactions"] >= volume_threshold)
    node_df["is_hub_supplier"] = node_df["is_multi_trust_supplier"] | node_df["is_high_volume_supplier"]

    logger.info(
        "Hub-supplier identification: %d multi-trust suppliers (degree>=2), %d high-volume suppliers (>=p%d txn count, threshold=%.0f) -> %d hub suppliers total (union)",
        int(node_df["is_multi_trust_supplier"].sum()), int(node_df["is_high_volume_supplier"].sum()),
        config.NETWORK_HUB_PERCENTILE, volume_threshold, int(node_df["is_hub_supplier"].sum()),
    )
    return node_df


def cross_reference_hubs_with_anomalies(node_df: pd.DataFrame) -> dict:
    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["supplier", "is_anomaly"], low_memory=False)
    supplier_anomaly_rate = scores.groupby("supplier")["is_anomaly"].mean()

    suppliers = node_df[node_df["node_type"] == "supplier"].copy()
    suppliers["anomaly_rate"] = suppliers["node"].map(supplier_anomaly_rate)
    suppliers = suppliers.dropna(subset=["anomaly_rate"])

    hub_rates = suppliers.loc[suppliers["is_hub_supplier"], "anomaly_rate"]
    nonhub_rates = suppliers.loc[~suppliers["is_hub_supplier"], "anomaly_rate"]

    u_stat, p_value = stats.mannwhitneyu(hub_rates, nonhub_rates, alternative="two-sided") if len(hub_rates) > 0 else (np.nan, np.nan)

    result = {
        "n_hub_suppliers": len(hub_rates), "n_nonhub_suppliers": len(nonhub_rates),
        "mean_anomaly_rate_hub_pct": round(hub_rates.mean() * 100, 3) if len(hub_rates) else np.nan,
        "mean_anomaly_rate_nonhub_pct": round(nonhub_rates.mean() * 100, 3) if len(nonhub_rates) else np.nan,
        "mann_whitney_u": u_stat, "p_value": p_value,
    }
    logger.info("Hub vs non-hub supplier anomaly rate: hub=%.3f%% (n=%d) vs non-hub=%.3f%% (n=%d) | Mann-Whitney p=%.4g",
                result["mean_anomaly_rate_hub_pct"], result["n_hub_suppliers"],
                result["mean_anomaly_rate_nonhub_pct"], result["n_nonhub_suppliers"], p_value)
    return result


def build_cooccurrence_projection(trust: pd.DataFrame) -> nx.Graph:
    grouped = trust.dropna(subset=["category", "year_month"]).groupby(["entity_norm", "category", "year_month"])["supplier_norm"].apply(
        lambda s: sorted(set(s))
    )
    grouped = grouped[grouped.apply(len).between(2, MAX_COOCCURRENCE_GROUP_SIZE)]

    G = nx.Graph()
    for suppliers in grouped:
        for i in range(len(suppliers)):
            for j in range(i + 1, len(suppliers)):
                a, b = suppliers[i], suppliers[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    logger.info("Supplier co-occurrence graph: %d suppliers, %d edges, from %d eligible buyer-category-month pools",
                G.number_of_nodes(), G.number_of_edges(), len(grouped))
    return G


def detect_communities(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_edges() == 0:
        return pd.DataFrame(columns=["node", "community_id", "community_size"])
    communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    modularity = nx.algorithms.community.modularity(G, communities, weight="weight")
    logger.info("Detected %d supplier communities via greedy modularity (Clauset-Newman-Moore); modularity=%.4f",
                len(communities), modularity)

    rows = []
    for cid, members in enumerate(communities):
        for node in members:
            rows.append({"node": node, "community_id": cid, "community_size": len(members)})
    return pd.DataFrame(rows)


def cross_reference_communities_with_anomalies(community_df: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["supplier", "is_anomaly"], low_memory=False)
    supplier_anomaly_rate = scores.groupby("supplier")["is_anomaly"].mean()

    df = community_df.copy()
    df["anomaly_rate"] = df["node"].map(supplier_anomaly_rate)
    df = df.dropna(subset=["anomaly_rate"])

    summary = df.groupby("community_id").agg(
        community_size=("node", "size"), mean_anomaly_rate_pct=("anomaly_rate", lambda x: round(x.mean() * 100, 3)),
    ).reset_index().sort_values("mean_anomaly_rate_pct", ascending=False)
    return summary[summary["community_size"] >= 3].head(15)


def run_network_analysis():
    trust = load_trust_records()

    bipartite_G = build_bipartite_graph(trust)
    node_df = compute_bipartite_centrality(bipartite_G)
    hub_comparison = cross_reference_hubs_with_anomalies(node_df)
    node_df.to_csv(config.NETWORK_NODE_METRICS_PATH, index=False)
    logger.info("Saved bipartite node metrics -> %s", config.NETWORK_NODE_METRICS_PATH)

    cooccurrence_G = build_cooccurrence_projection(trust)
    community_df = detect_communities(cooccurrence_G)
    top_flagged_communities = cross_reference_communities_with_anomalies(community_df) if len(community_df) else pd.DataFrame()
    community_df.to_csv(config.NETWORK_COMMUNITY_PATH, index=False)
    logger.info("Saved community assignments -> %s", config.NETWORK_COMMUNITY_PATH)
    if len(top_flagged_communities):
        logger.info("Top supplier communities by mean anomaly rate (size>=3):\n%s", top_flagged_communities.to_string(index=False))

    return node_df, hub_comparison, community_df, top_flagged_communities


if __name__ == "__main__":
    run_network_analysis()
