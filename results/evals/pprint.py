#!/usr/bin/env python3
# Simple utility to pretty-print evals.

import json
import glob


def main():
    json_files = glob.glob("./*.json")

    for file_path in sorted(json_files):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            file_name = file_path.split("/")[-1]
            print(f"{'='*50}")
            print(f"{file_name}:")
            print("=" * 50)

            # Check train/valid/eval sections
            for section in ["train", "valid", "eval"]:
                if section in data and isinstance(data[section], dict):
                    metrics = data[section]
                    if "accuracy" in metrics and metrics["accuracy"] is not None:
                        print(f"    {section}_acc: {metrics['accuracy']:.4f}")
                    if "f1" in metrics and metrics["f1"] is not None:
                        print(f"    {section}_f1: {metrics['f1']:.4f}")

            print()

        except Exception as e:
            print(f"Error reading {file_path}: {e}")


if __name__ == "__main__":
    main()
