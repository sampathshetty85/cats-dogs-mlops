import os
from datetime import datetime


def write_report(step_name: str, lines: list) -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{step_name}.txt")
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(report_path, "w") as f:
        f.write(f"# Report: {step_name}\n")
        f.write(f"# Generated: {timestamp}\n")
        f.write("#" + "-" * 60 + "\n\n")
        for line in lines:
            f.write(str(line) + "\n")
    print(f"[report_writer] Written: output/{step_name}.txt")
