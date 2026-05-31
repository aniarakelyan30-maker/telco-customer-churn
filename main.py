"""
main.py
-------
One-command pipeline. Runs the full case study end to end:

    raw data -> cleaning/feature engineering -> EDA figures
             -> model training & evaluation -> business impact

Usage:  python main.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from data_preprocessing import build_processed_dataset   # noqa: E402
from eda import run_all_eda                               # noqa: E402
from train_model import train_and_evaluate               # noqa: E402
import business_impact                                    # noqa: E402


def banner(text):
    print("\n" + "#" * 64)
    print(f"#  {text}")
    print("#" * 64)


if __name__ == "__main__":
    banner("STEP 1/4  Data cleaning & feature engineering")
    build_processed_dataset()

    banner("STEP 2/4  Exploratory data analysis")
    run_all_eda()

    banner("STEP 3/4  Model training & evaluation")
    train_and_evaluate()

    banner("STEP 4/4  Business impact & retention strategy")
    business_impact.run()

    banner("PIPELINE COMPLETE  —  see reports/figures/ and models/")
