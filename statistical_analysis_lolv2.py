
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "results" / "lolv2_cross_dataset"
    / "lolv2_per_image_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "results" / "lolv2_cross_dataset"
)

METRICS = ["PSNR", "SSIM", "MAE"]

COMPARISONS = [
    ("Experiment_5", "Adaptive_Gamma_CLAHE"),
    ("Experiment_5", "Full_NightWatch"),
    ("Adaptive_Gamma_CLAHE", "Full_NightWatch")
]


def holm_correction(p_values):

    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(n)

    running_max = 0.0

    for rank, index in enumerate(order):

        value = (n - rank) * p_values[index]

        running_max = max(running_max, value)

        adjusted[index] = min(running_max, 1.0)

    return adjusted


def main():

    print("\n========================================")
    print(" LOLv2 REAL STATISTICAL VALIDATION")
    print("========================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = {"Image", "Stage", "PSNR", "SSIM", "MAE"}

    if not required.issubset(df.columns):
        raise ValueError("Required columns are missing.")

    if df.duplicated(["Image", "Stage"]).any():
        raise ValueError("Duplicate image-stage observations.")

    results = []

    for stage_a, stage_b in COMPARISONS:

        for metric in METRICS:

            pivot = df.pivot(
                index="Image",
                columns="Stage",
                values=metric
            )

            paired = pivot[[stage_a, stage_b]].dropna()

            a = paired[stage_a].to_numpy(dtype=float)
            b = paired[stage_b].to_numpy(dtype=float)

            # Positive difference means B is better.
            if metric == "MAE":
                difference = a - b
            else:
                difference = b - a

            n = len(difference)

            if n < 2:
                raise ValueError("Insufficient paired observations.")

            mean_diff = np.mean(difference)
            std_diff = np.std(difference, ddof=1)
            se = std_diff / np.sqrt(n)

            critical = stats.t.ppf(0.975, n - 1)

            ci_low = mean_diff - critical * se
            ci_high = mean_diff + critical * se

            if np.allclose(difference, 0):

                t_stat, t_p = 0.0, 1.0
                w_stat, w_p = 0.0, 1.0

            else:

                t_result = stats.ttest_rel(b, a)

                w_result = stats.wilcoxon(
                    difference,
                    alternative="two-sided",
                    method="auto"
                )

                t_stat = float(t_result.statistic)
                t_p = float(t_result.pvalue)

                w_stat = float(w_result.statistic)
                w_p = float(w_result.pvalue)

            results.append({

                "Comparison": f"{stage_a} vs {stage_b}",
                "Metric": metric,
                "N": n,

                "Stage_A_Mean": np.mean(a),
                "Stage_B_Mean": np.mean(b),

                "Mean_Improvement": mean_diff,
                "Std_Difference": std_diff,

                "CI_95_Lower": ci_low,
                "CI_95_Upper": ci_high,

                "Improved": int(np.sum(difference > 0)),
                "Unchanged": int(np.sum(difference == 0)),
                "Deteriorated": int(np.sum(difference < 0)),

                "T_Statistic": t_stat,
                "T_P_Value": t_p,

                "Wilcoxon_Statistic": w_stat,
                "Wilcoxon_P_Value": w_p
            })

    result_df = pd.DataFrame(results)

    # Holm correction across all 9 comparisons,
    # separately for each statistical test.

    result_df["T_Holm_P_Value"] = holm_correction(
        result_df["T_P_Value"].values
    )

    result_df["Wilcoxon_Holm_P_Value"] = holm_correction(
        result_df["Wilcoxon_P_Value"].values
    )

    result_df["T_Significant_Holm"] = (
        result_df["T_Holm_P_Value"] < 0.05
    )

    result_df["Wilcoxon_Significant_Holm"] = (
        result_df["Wilcoxon_Holm_P_Value"] < 0.05
    )

    output_file = (
        OUTPUT_DIR / "lolv2_statistical_results.csv"
    )

    result_df.to_csv(output_file, index=False)

    print("\nSTATISTICAL RESULTS")
    print("----------------------------------------")

    columns = [
        "Comparison",
        "Metric",
        "Mean_Improvement",
        "CI_95_Lower",
        "CI_95_Upper",
        "Improved",
        "T_P_Value",
        "T_Holm_P_Value",
        "Wilcoxon_P_Value",
        "Wilcoxon_Holm_P_Value"
    ]

    print(
        result_df[columns].round(6).to_string(index=False)
    )

    print("\nSaved:", output_file)

    print("\nHolm-corrected significance: p < 0.05")
    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()
