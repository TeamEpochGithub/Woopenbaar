#!/usr/bin/env python3
import json

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib import ticker


def evaluate_multi_k(
    benchmarks: list, ks_by_type: dict, api_base_url: str = "http://localhost:5000"
) -> dict:
    metrics_by_type = {}
    for b_type, k_list in ks_by_type.items():
        metrics_by_type[b_type] = {k: {"recalls": [], "precisions": []} for k in k_list}

    for benchmark in benchmarks:
        b_type = benchmark["type"]
        question = benchmark["question"]
        relevant = benchmark["relevant_uuids"]
        num_relevant = len(relevant)
        if b_type not in ks_by_type:
            continue

        found_all = False  # Track if we've found all relevant docs
        for k in ks_by_type[b_type]:
            if found_all:  # Skip remaining k values if we found everything
                # Copy the last perfect results for larger k values
                metrics_by_type[b_type][k]["recalls"].append(1.0)
                metrics_by_type[b_type][k]["precisions"].append(
                    num_relevant / k if k > 0 else 0.0
                )
                continue

            payload = {
                "message": question,
                "filters": {},
                "documents_k": k * 30,
                "chunks_k": k,
            }

            try:
                # Use the new retrieve-context endpoint which is faster
                response = requests.post(
                    f"{api_base_url}/retrieve-context",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                if response.status_code == 404:
                    raise RuntimeError(
                        "Service not found. Please ensure the server is running."
                    )
                result = response.json()

                retrieved_ids = result.get("doc_ids", [])
                hits = sum(1 for rid in retrieved_ids if rid in relevant)
                recall = hits / num_relevant if num_relevant > 0 else 0.0
                precision = hits / k if k > 0 else 0.0

                metrics_by_type[b_type][k]["recalls"].append(recall)
                metrics_by_type[b_type][k]["precisions"].append(precision)

                print(
                    f"Question: {question[:50]}... | K={k} | Recall={recall:.2f} | Precision={precision:.2f}"
                )

                # Check if we found all relevant documents
                if hits == num_relevant:
                    found_all = True

            except Exception as e:
                print(
                    f"Error calling API for question '{question}' with k={k}: {str(e)}"
                )
                metrics_by_type[b_type][k]["recalls"].append(0.0)
                metrics_by_type[b_type][k]["precisions"].append(0.0)

    averaged = {}
    for b_type, metrics in metrics_by_type.items():
        ks = []
        recalls = []
        precisions = []
        for k in sorted(metrics.keys()):
            avg_recall = (
                np.mean(metrics[k]["recalls"]) if metrics[k]["recalls"] else 0.0
            )
            avg_precision = (
                np.mean(metrics[k]["precisions"]) if metrics[k]["precisions"] else 0.0
            )
            ks.append(k)
            recalls.append(avg_recall)
            precisions.append(avg_precision)
        averaged[b_type] = {"ks": ks, "recalls": recalls, "precisions": precisions}
    return averaged


def plot_multi_k_results(results_by_type: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    colors = {"specific": "red", "document": "blue", "multi_document": "green"}

    # Recall vs k
    ax = axes[0]
    for b_type, metrics in results_by_type.items():
        ax.plot(
            metrics["ks"],
            metrics["recalls"],
            marker="o",
            color=colors.get(b_type),
            label=b_type,
        )
    ax.set_xlabel("k (top-k)")
    ax.set_ylabel("Average Recall")
    ax.set_title("Recall@k")
    ax.legend()
    ax.grid(True)

    # Add more tick marks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))  # More major ticks on x-axis
    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(0.05)
    )  # Ticks every 0.05 on y-axis
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))  # Minor ticks every 0.01
    ax.grid(True, which="minor", linestyle=":", alpha=0.5)

    # Precision vs k
    ax = axes[1]
    for b_type, metrics in results_by_type.items():
        ax.plot(
            metrics["ks"],
            metrics["precisions"],
            marker="o",
            color=colors.get(b_type),
            label=b_type,
        )
    ax.set_xlabel("k (top-k)")
    ax.set_ylabel("Average Precision")
    ax.set_title("Precision@k")
    ax.legend()
    ax.grid(True)

    # Add more tick marks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))  # More major ticks on x-axis
    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(0.05)
    )  # Ticks every 0.05 on y-axis
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))  # Minor ticks every 0.01
    ax.grid(True, which="minor", linestyle=":", alpha=0.5)

    # Precision vs Recall
    ax = axes[2]
    for b_type, metrics in results_by_type.items():
        ax.plot(
            metrics["recalls"],
            metrics["precisions"],
            marker="o",
            color=colors.get(b_type),
            label=b_type,
        )
        for r, p, k in zip(metrics["recalls"], metrics["precisions"], metrics["ks"]):
            ax.annotate(str(k), (r, p))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs. Recall")
    ax.legend()
    ax.grid(True)

    # Add more tick marks
    ax.xaxis.set_major_locator(
        ticker.MultipleLocator(0.05)
    )  # Ticks every 0.05 on x-axis
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))  # Minor ticks every 0.01
    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(0.05)
    )  # Ticks every 0.05 on y-axis
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))  # Minor ticks every 0.01
    ax.grid(True, which="minor", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("retrieval_recall_precision_curve.png")
    plt.show()
    print("Multi-k evaluation plot saved to retrieval_recall_precision_curve.png")


def main():
    with open("../v1_benchmark.json", "r") as f:
        benchmarks = json.load(f)

    api_base_url = "http://localhost:5000"  # Default Flask server address

    ks_by_type = {
        "specific": list(range(1, 20, 1)),
        "document": list(range(1, 20, 1)),
        "multi_document": list(range(1, 20, 1)),
    }

    results = evaluate_multi_k(benchmarks, ks_by_type, api_base_url)

    with open("multi_k_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    plot_multi_k_results(results)


if __name__ == "__main__":
    main()
