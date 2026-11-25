import os
import sys
import time
import json
import argparse
import statistics
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabulate import tabulate

from backend.config import settings
from tests.test_100_complaints import SYNTHETIC_100_DATASET
from agents.triage_agent import run_triage_agent
from agents.order_agent import run_order_agent
from agents.policy_agent import run_policy_agent
from agents.resolution_agent import run_resolution_agent
from agents.quality_agent import run_quality_agent


CATEGORIES = ["refund", "replacement", "cancellation", "shipping_delay", "damaged_item", "general_inquiry"]


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Calculates accuracy, per-class & macro precision, recall, F1, and confusion matrix."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {}

    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    matrix = {actual: {pred: 0 for pred in CATEGORIES} for actual in CATEGORIES}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1

    for cat in CATEGORIES:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cat and yp == cat)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cat and yp == cat)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cat and yp != cat)
        support = sum(1 for yt in y_true if yt == cat)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cat] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support
        }
        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    macro_precision = statistics.mean(precisions) if precisions else 0.0
    macro_recall = statistics.mean(recalls) if recalls else 0.0
    macro_f1 = statistics.mean(f1s) if f1s else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix
    }


def run_qwen_100_benchmark(limit: Optional[int] = None):
    dataset = SYNTHETIC_100_DATASET[:limit] if limit else SYNTHETIC_100_DATASET
    total = len(dataset)

    print("=" * 80)
    print("RESOLVEAI: REAL LOCAL QWEN3.5:4B PERFORMANCE BENCHMARK (100 COMPLAINTS)")
    print(f"Active Provider: {settings.LLM_PROVIDER} ({settings.QWEN_MODEL})")
    print(f"Ollama Endpoint: {settings.OLLAMA_BASE_URL} (Local Inference, Zero Cloud API Cost)")
    print(f"Low Reasoning Enabled: {settings.QWEN_LOW_REASONING} (think=False)")
    print(f"Total Test Cases: {total}")
    print("=" * 80)

    # Health check verification
    health = settings.check_qwen_health()
    if health.get("status") != "healthy":
        print(f"\n❌ ERROR: Ollama health check failed: {health.get('message')}")
        print("Please ensure Ollama is running and 'qwen3.5:4b' is downloaded.")
        sys.exit(1)

    y_true = []
    y_pred = []
    triage_latencies = []
    e2e_latencies = []
    inference_errors = []
    misclassifications = []
    resolution_success_count = 0
    tool_success_count = 0
    total_tokens_consumed = 0

    benchmark_start_time = time.time()

    for idx, item in enumerate(dataset, 1):
        cid = item["id"]
        text = item["text"]
        expected_cat = item["category"]
        expected_order_id = item.get("order_id")

        t0 = time.time()

        try:
            # 1. Triage Agent (STRICT Real Ollama Qwen Inference, Fallback DISABLED)
            t_triage_start = time.time()
            triage_res = run_triage_agent(text, allow_fallback=False)
            triage_lat = (time.time() - t_triage_start) * 1000
            triage_latencies.append(triage_lat)

            pred_cat = triage_res.get("category", "general_inquiry")
            y_true.append(expected_cat)
            y_pred.append(pred_cat)

            if triage_res.get("token_usage"):
                total_tokens_consumed += triage_res["token_usage"].get("total_tokens", 0)

            # Record misclassifications
            if pred_cat != expected_cat:
                misclassifications.append({
                    "id": cid,
                    "text": text[:60] + ("..." if len(text) > 60 else ""),
                    "expected": expected_cat,
                    "predicted": pred_cat
                })

            # 2. Order Investigation Agent (Deterministic Tool)
            order_id = triage_res.get("order_id")
            order_res = run_order_agent(order_id, customer_id=2, authenticated_customer_id=2)
            if order_res:
                tool_success_count += 1

            # 3. Policy Research Agent (Local ChromaDB RAG)
            policy_res = run_policy_agent(pred_cat, text)

            # 4. Resolution Agent (Deterministic Rule Synthesis)
            res_agent = run_resolution_agent(
                complaint_text=text,
                category=pred_cat,
                order_data=order_res.get("order_data"),
                policy_citations=policy_res.get("citations", []),
                allow_fallback=True,
                use_llm=False
            )
            if res_agent.get("action"):
                resolution_success_count += 1

            # 5. Quality & Escalation Agent
            quality_res = run_quality_agent(
                complaint_text=text,
                order_data=order_res.get("order_data"),
                policy_citations=policy_res.get("citations", []),
                proposed_resolution=res_agent
            )

            total_item_lat = (time.time() - t0) * 1000
            e2e_latencies.append(total_item_lat)

            status_icon = "✓" if pred_cat == expected_cat else "✗"
            print(f"[{idx:03d}/{total:03d}] Case {cid:03d} {status_icon} Exp: {expected_cat:<15} Pred: {pred_cat:<15} (Triage: {triage_lat:.1f}ms | Total: {total_item_lat:.1f}ms)")

        except Exception as e:
            inference_errors.append({"id": cid, "error": str(e)})
            print(f"[{idx:03d}/{total:03d}] Case {cid:03d} ⚠️  INFERENCE ERROR: {e}")

    total_inference_time_sec = time.time() - benchmark_start_time

    # Compute Statistical Metrics
    metrics = compute_metrics(y_true, y_pred)
    if not metrics:
        print("\nNo successful classifications to evaluate.")
        return

    avg_triage_lat = statistics.mean(triage_latencies) if triage_latencies else 0.0
    median_triage_lat = statistics.median(triage_latencies) if triage_latencies else 0.0
    p95_triage_lat = statistics.quantiles(triage_latencies, n=20)[18] if len(triage_latencies) >= 20 else max(triage_latencies, default=0.0)

    avg_e2e_lat = statistics.mean(e2e_latencies) if e2e_latencies else 0.0
    median_e2e_lat = statistics.median(e2e_latencies) if e2e_latencies else 0.0
    p95_e2e_lat = statistics.quantiles(e2e_latencies, n=20)[18] if len(e2e_latencies) >= 20 else max(e2e_latencies, default=0.0)

    res_accuracy = (resolution_success_count / total * 100) if total > 0 else 0.0
    tool_success_rate = (tool_success_count / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 80)
    print("FINAL RESOLVEAI PERFORMANCE BENCHMARK REPORT (QWEN3.5:4B LOCAL INFERENCE)")
    print("=" * 80)

    summary_data = [
        ["Total Evaluated Cases", total],
        ["Successful Classifications", len(y_pred)],
        ["Inference API Failures / Errors", len(inference_errors)],
        ["Intent Classification Accuracy", f"{metrics['accuracy'] * 100:.2f}%"],
        ["Macro Precision", f"{metrics['macro_precision'] * 100:.2f}%"],
        ["Macro Recall", f"{metrics['macro_recall'] * 100:.2f}%"],
        ["Macro F1-Score", f"{metrics['macro_f1'] * 100:.2f}%"],
        ["Resolution Proposal Success Rate", f"{res_accuracy:.2f}%"],
        ["Tool Execution Success Rate", f"{tool_success_rate:.2f}%"],
        ["Average Triage Latency", f"{avg_triage_lat:.2f} ms"],
        ["P50 (Median) Triage Latency", f"{median_triage_lat:.2f} ms"],
        ["P95 Triage Latency", f"{p95_triage_lat:.2f} ms"],
        ["Average End-to-End Latency", f"{avg_e2e_lat:.2f} ms"],
        ["P50 (Median) End-to-End Latency", f"{median_e2e_lat:.2f} ms"],
        ["P95 End-to-End Latency", f"{p95_e2e_lat:.2f} ms"],
        ["Total Benchmark Run Time", f"{total_inference_time_sec:.2f} seconds"],
        ["Total Tokens Generated / Evaluated", total_tokens_consumed]
    ]

    print("\n1. Overall Multi-Agent Performance Summary:")
    print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # Per-Class Breakdown Table
    class_table = []
    for cat in CATEGORIES:
        c = metrics["per_class"][cat]
        class_table.append([
            cat,
            c["support"],
            f"{c['precision'] * 100:.1f}%",
            f"{c['recall'] * 100:.1f}%",
            f"{c['f1'] * 100:.1f}%"
        ])

    print("\n2. Per-Class Precision, Recall & F1 Breakdown:")
    print(tabulate(class_table, headers=["Category", "Support", "Precision", "Recall", "F1-Score"], tablefmt="fancy_grid"))

    # Confusion Matrix Table
    cm_headers = ["Actual \\ Pred"] + CATEGORIES
    cm_rows = []
    for cat in CATEGORIES:
        row = [cat] + [metrics["confusion_matrix"][cat][pred_cat] for pred_cat in CATEGORIES]
        cm_rows.append(row)

    print("\n3. Confusion Matrix:")
    print(tabulate(cm_rows, headers=cm_headers, tablefmt="grid"))

    # Misclassifications
    print(f"\n4. Misclassified Cases ({len(misclassifications)} total):")
    if misclassifications:
        misc_rows = [[m["id"], m["expected"], m["predicted"], m["text"]] for m in misclassifications[:10]]
        print(tabulate(misc_rows, headers=["Case ID", "Expected", "Predicted", "Text Snippet"], tablefmt="fancy_grid"))
    else:
        print("  None (100% classification accuracy!)")

    print("\n" + "=" * 80)

    # Save benchmark json artifact
    report_dict = {
        "model": settings.QWEN_MODEL,
        "endpoint": settings.OLLAMA_BASE_URL,
        "total_cases": total,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "resolution_success_rate": res_accuracy,
            "tool_success_rate": tool_success_rate,
            "average_triage_latency_ms": avg_triage_lat,
            "median_triage_latency_ms": median_triage_lat,
            "p95_triage_latency_ms": p95_triage_lat,
            "total_inference_time_sec": total_inference_time_sec,
            "total_tokens_consumed": total_tokens_consumed
        },
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "misclassifications": misclassifications
    }
    with open("tests/qwen_100_benchmark_results.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print("Benchmark results saved to tests/qwen_100_benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResolveAI Qwen3.5:4B Real Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test complaints")
    args = parser.parse_args()

    run_qwen_100_benchmark(limit=args.limit)

