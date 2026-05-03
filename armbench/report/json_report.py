import json
import os
from datetime import datetime


def save_json_report(results, model_name, output_dir="results"):
    """Save benchmark results to a timestamped JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": results,
    }

    filename = f"{model_name}_report.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report saved: {filepath}")
    return filepath