import csv
import os

import matplotlib.pyplot as plt


def save_comparison_csv(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "model",
        "training_method",
        "clean_acc",
        "fgsm_robust_acc",
        "pgd_robust_acc"
    ]

    with open(save_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"对抗训练对比结果已保存到: {save_path}")


def plot_comparison(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = [row["training_method"] for row in results]

    clean_acc = [row["clean_acc"] for row in results]
    fgsm_robust_acc = [row["fgsm_robust_acc"] for row in results]
    pgd_robust_acc = [row["pgd_robust_acc"] for row in results]

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width for i in x],
        clean_acc,
        width=width,
        label="Clean Acc"
    )

    plt.bar(
        x,
        fgsm_robust_acc,
        width=width,
        label="FGSM Robust Acc"
    )

    plt.bar(
        [i + width for i in x],
        pgd_robust_acc,
        width=width,
        label="PGD Robust Acc"
    )

    plt.xticks(list(x), labels)
    plt.ylabel("Accuracy")
    plt.title("Standard Training vs FGSM Adversarial Training")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i, value in enumerate(clean_acc):
        plt.text(i - width, value + 0.02, f"{value * 100:.1f}%", ha="center")

    for i, value in enumerate(fgsm_robust_acc):
        plt.text(i, value + 0.02, f"{value * 100:.1f}%", ha="center")

    for i, value in enumerate(pgd_robust_acc):
        plt.text(i + width, value + 0.02, f"{value * 100:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"对抗训练对比图已保存到: {save_path}")


def main():
    results = [
        {
            "model": "SimpleCNN",
            "training_method": "Standard",
            "clean_acc": 0.9871,
            "fgsm_robust_acc": 0.0667,
            "pgd_robust_acc": 0.0
        },
        {
            "model": "SimpleCNN",
            "training_method": "FGSM Adv Training",
            "clean_acc": 0.9232,
            "fgsm_robust_acc": 0.8671,
            "pgd_robust_acc": 0.0
        }
    ]

    save_comparison_csv(
        results,
        "results/adv_training_comparison.csv"
    )

    plot_comparison(
        results,
        "results/adv_training_comparison.png"
    )


if __name__ == "__main__":
    main()