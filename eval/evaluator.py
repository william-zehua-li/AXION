import json
import time
from collections import defaultdict


class Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate(self, dataset, output_path: str = None) -> dict:
        """
        dataset: iterable of dicts with keys:
          "input"  — the question string fed to the pipeline
          "answer" — the reference answer string used for scoring

        Returns a dict of metric name → scalar value.
        If output_path is given, also writes a JSON file with the summary
        and per-sample breakdown.
        """
        per_sample = []

        for sample in dataset:
            question  = sample["input"]
            reference = sample.get("answer", "")

            t0 = time.perf_counter()
            result = self.pipeline.run(question)
            latency = time.perf_counter() - t0

            scores = self._score_answer(result["answer"], reference)

            loop_outputs = result.get("loop_outputs", [])
            ticks = result["ticks_used"]

            # Count per-branch usage and RAG hits from loop output metadata
            mem_ticks = sum(1 for o in loop_outputs if o.get("branch") == "mem")
            rag_ticks = sum(1 for o in loop_outputs if o.get("branch") == "rag")
            rag_hits  = sum(1 for o in loop_outputs if o.get("rag_hit", False))

            ku = result.get("knowledge_updates", {})

            per_sample.append({
                "exact_match":     scores["exact_match"],
                "f1":              scores["f1"],
                "ticks":           ticks,
                "mem_ticks":       mem_ticks,
                "rag_ticks":       rag_ticks,
                "rag_hits":        rag_hits,
                "wrote_knowledge": int(bool(ku.get("written"))),
                "latency":         latency,
            })

        n = len(per_sample)
        if n == 0:
            print("Warning: dataset is empty — no metrics computed.")
            return {}

        total_ticks = sum(m["ticks"]     for m in per_sample)
        total_rag   = sum(m["rag_ticks"] for m in per_sample)

        aggregated = {
            "num_samples":           n,
            "exact_match":           sum(m["exact_match"]     for m in per_sample) / n,
            "f1":                    sum(m["f1"]              for m in per_sample) / n,
            "avg_ticks":             total_ticks / n,
            "mem_utilization_rate":  (
                sum(m["mem_ticks"] for m in per_sample) / total_ticks
                if total_ticks > 0 else 0.0
            ),
            "rag_hit_rate":          (
                sum(m["rag_hits"] for m in per_sample) / total_rag
                if total_rag > 0 else 0.0
            ),
            "knowledge_write_rate":  sum(m["wrote_knowledge"] for m in per_sample) / n,
            "avg_latency_s":         sum(m["latency"]         for m in per_sample) / n,
        }

        self._print_table(aggregated)

        if output_path:
            with open(output_path, "w") as f:
                json.dump({"summary": aggregated, "per_sample": per_sample}, f, indent=2)

        return aggregated

    def _score_answer(self, predicted: str, reference: str) -> dict:
        pred_norm = predicted.strip().lower()
        ref_norm  = reference.strip().lower()

        exact_match = 1.0 if pred_norm == ref_norm else 0.0

        pred_tokens = pred_norm.split()
        ref_tokens  = ref_norm.split()

        if not pred_tokens or not ref_tokens:
            return {"exact_match": exact_match, "f1": 0.0}

        pred_counts: dict = defaultdict(int)
        ref_counts:  dict = defaultdict(int)
        for t in pred_tokens:
            pred_counts[t] += 1
        for t in ref_tokens:
            ref_counts[t] += 1

        common = sum(
            min(pred_counts[t], ref_counts[t])
            for t in pred_counts
            if t in ref_counts
        )
        if common == 0:
            return {"exact_match": exact_match, "f1": 0.0}

        precision = common / len(pred_tokens)
        recall    = common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        return {"exact_match": exact_match, "f1": f1}

    def _print_table(self, metrics: dict) -> None:
        rows = [
            ("Samples",              str(metrics["num_samples"])),
            ("Exact Match",          f"{metrics['exact_match']:.3f}"),
            ("F1 Score",             f"{metrics['f1']:.3f}"),
            ("Avg Ticks / Query",    f"{metrics['avg_ticks']:.2f}"),
            ("Mem Utilization Rate", f"{metrics['mem_utilization_rate']:.3f}"),
            ("RAG Hit Rate",         f"{metrics['rag_hit_rate']:.3f}"),
            ("Knowledge Write Rate", f"{metrics['knowledge_write_rate']:.3f}"),
            ("Avg Latency (s)",      f"{metrics['avg_latency_s']:.3f}"),
        ]
        w = 47
        print("\n" + "=" * w)
        print(f"  {'Metric':<30} {'Value':>12}")
        print("-" * w)
        for name, val in rows:
            print(f"  {name:<30} {val:>12}")
        print("=" * w + "\n")
