"""Run isolated framework-native indexing experiments without paid services."""

import argparse
from pathlib import Path

from raglab.indexing_experiments import (
    load_plan,
    load_plan_cases,
    run_indexing_experiments,
    write_indexing_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/indexing_experiments/v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/native-indexing-v1"),
        help="Output path prefix; .json and .md are added",
    )
    arguments = parser.parse_args()

    plan, config_sha256 = load_plan(arguments.config)
    cases, dataset_sha256 = load_plan_cases(plan)
    run = run_indexing_experiments(
        plan,
        cases,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
    )
    json_path = arguments.output.with_suffix(".json")
    markdown_path = arguments.output.with_suffix(".md")
    write_indexing_report(run, json_path, markdown_path)
    print(f"run_id={run.run_id} cost=0.0")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")


if __name__ == "__main__":
    main()
