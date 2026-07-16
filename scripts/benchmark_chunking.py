"""Run RAGLab's deterministic chunking benchmark."""

import argparse
from pathlib import Path

from raglab.chunking.benchmark import load_cases, run_benchmark, write_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/evaluation/chunking_benchmark_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/chunking-benchmark.json"),
    )
    arguments = parser.parse_args()
    cases = load_cases(arguments.dataset)
    results = run_benchmark(cases)
    write_results(arguments.output, results)
    print(f"Wrote {len(results)} measurements to {arguments.output}")


if __name__ == "__main__":
    main()
