
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "adaptive_refinement_evaluation"
    / "adaptive_per_image_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "results" / "statistical_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_LEVEL = 0.95


# ==================================================
# LOAD AND VALIDATE DATA
# ==================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "Image", "Stage", "PSNR", "SSIM", "MAE"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df.duplicated(["Image", "Stage"]).any():
        raise ValueError(
            "Duplicate image-stage observations found."
        )

    return df


# ==================================================
# PAIRED STATISTICAL TESTS
# ==================================================

def analyze_pair(df, stage_a, stage_b):

    metrics = ["PSNR", "SSIM", "MAE"]

    results = []

    for metric in metrics:

        pivot = df.pivot(
            index="Image",
            columns="Stage",
            values=metric
        )

        paired = pivot[[stage_a, stage_b]].dropna()

        a = paired[stage_a].to_numpy(dtype=float)
        b = paired[stage_b].to_numpy(dtype=float)

        n = len(a)

        if n < 2:
            raise ValueError(
                f"Insufficient paired observations for {metric}"
            )

        # Positive difference always means stage B is better.
        if metric == "MAE":
            differences = a - b
        else:
            differences = b - a

        mean_difference = np.mean(differences)
        std_difference = np.std(
            differences, ddof=1
        )

        standard_error = (
            std_difference / np.sqrt(n)
        )

        critical_value = stats.t.ppf(
            0.975, df=n - 1
        )

        margin = critical_value * standard_error

        ci_lower = mean_difference - margin
        ci_upper = mean_difference + margin

        # Paired t-test
        if np.allclose(differences, 0):
            t_statistic = 0.0
            t_pvalue = 1.0
        else:
            t_result = stats.ttest_rel(b, a)
            t_statistic = float(t_result.statistic)
            t_pvalue = float(t_result.pvalue)

        # Wilcoxon signed-rank test
        if np.allclose(differences, 0):
            wilcoxon_statistic = 0.0
            wilcoxon_pvalue = 1.0
        else:
            w_result = stats.wilcoxon(
                differences,
                alternative="two-sided",
                method="auto"
            )

            wilcoxon_statistic = float(w_result.statistic)
            wilcoxon_pvalue = float(w_result.pvalue)

        improved = int(np.sum(differences > 0))
        unchanged = int(np.sum(differences == 0))
        deteriorated = int(np.sum(differences < 0))

        results.append({

            "Comparison": f"{stage_a} vs {stage_b}",
            "Metric": metric,
            "Images": n,

            "Stage_A_Mean": np.mean(a),
            "Stage_B_Mean": np.mean(b),

            "Mean_Improvement": mean_difference,
            "Std_Difference": std_difference,

            "CI_95_Lower": ci_lower,
            "CI_95_Upper": ci_upper,

            "Improved": improved,
            "Unchanged": unchanged,
            "Deteriorated": deteriorated,

            "Paired_t_statistic": t_statistic,
            "Paired_t_pvalue": t_pvalue,

            "Wilcoxon_statistic": wilcoxon_statistic,
            "Wilcoxon_pvalue": wilcoxon_pvalue
        })

    return results


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n========================================")
    print(" NIGHTWATCH STATISTICAL ANALYSIS")
    print("========================================")

    df = load_data()

    print("\nLoaded observations:", len(df))
    print("Unique test images:", df["Image"].nunique())

    stages = set(df["Stage"].unique())

    required_stages = {
        "Experiment_5",
        "Adaptive_Gamma_CLAHE",
        "Full_NightWatch"
    }

    if not required_stages.issubset(stages):
        raise ValueError(
            f"Missing stages: {required_stages - stages}"
        )

    comparisons = [
        (
            "Experiment_5",
            "Adaptive_Gamma_CLAHE"
        ),
        (
            "Experiment_5",
            "Full_NightWatch"
        ),
        (
            "Adaptive_Gamma_CLAHE",
            "Full_NightWatch"
        )
    ]

    all_results = []

    for stage_a, stage_b in comparisons:

        print("\n----------------------------------------")
        print(f"{stage_a} vs {stage_b}")
        print("----------------------------------------")

        pair_results = analyze_pair(
            df, stage_a, stage_b
        )

        all_results.extend(pair_results)

        for result in pair_results:

            print(
                f"\n{result['Metric']}:"
            )

            print(
                f"Mean improvement: "
                f"{result['Mean_Improvement']:.6f}"
            )

            print(
                f"95% CI: "
                f"[{result['CI_95_Lower']:.6f}, "
                f"{result['CI_95_Upper']:.6f}]"
            )

            print(
                f"Improved: {result['Improved']}/"
                f"{result['Images']}"
            )

            print(
                f"Paired t-test p-value: "
                f"{result['Paired_t_pvalue']:.6f}"
            )

            print(
                f"Wilcoxon p-value: "
                f"{result['Wilcoxon_pvalue']:.6f}"
            )

    # Save results

    results_df = pd.DataFrame(all_results)

    output_file = (
        OUTPUT_DIR / "paired_statistical_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n========================================")
    print(" STATISTICAL ANALYSIS COMPLETED")
    print("========================================")

    print("\nSaved:", output_file)

    print("\nP-value interpretation:")
    print("p < 0.05: statistically significant")
    print("p >= 0.05: not statistically significant")

    print(
        "\nNote: Interpret p-values cautiously "
        "because the test set contains only 15 images."
    )


if __name__ == "__main__":
    main()
