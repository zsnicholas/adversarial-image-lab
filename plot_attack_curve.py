import argparse
import csv
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.simple_cnn import SimpleCNN


def fgsm_attack(model, images, labels, loss_fn, epsilon):
    images = images.clone().detach().requires_grad_(True)

    logits = model(images)
    loss = loss_fn(logits, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.sign()
    adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()


def pgd_attack(model, images, labels, loss_fn, epsilon, alpha, steps):
    ori_images = images.clone().detach()
    adv_images = ori_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)

        logits = model(adv_images)
        loss = loss_fn(logits, labels)

        model.zero_grad()
        loss.backward()

        grad_sign = adv_images.grad.sign()

        adv_images = adv_images + alpha * grad_sign

        perturbation = torch.clamp(
            adv_images - ori_images,
            min=-epsilon,
            max=epsilon
        )

        adv_images = torch.clamp(
            ori_images + perturbation,
            min=0,
            max=1
        ).detach()

    return adv_images


def evaluate_attack(
    model,
    data_loader,
    device,
    attack_name,
    epsilon,
    alpha,
    steps,
    max_samples=None
):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total = 0
    clean_correct = 0
    robust_correct = 0
    attack_success = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None:
            remain = max_samples - total
            if images.size(0) > remain:
                images = images[:remain]
                labels = labels[:remain]

        with torch.no_grad():
            clean_logits = model(images)
            clean_pred = torch.argmax(clean_logits, dim=1)

        clean_mask = clean_pred == labels
        clean_correct += clean_mask.sum().item()

        if attack_name == "fgsm":
            adv_images = fgsm_attack(
                model=model,
                images=images,
                labels=labels,
                loss_fn=loss_fn,
                epsilon=epsilon
            )

        elif attack_name == "pgd":
            adv_images = pgd_attack(
                model=model,
                images=images,
                labels=labels,
                loss_fn=loss_fn,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps
            )

        else:
            raise ValueError(f"不支持的攻击类型: {attack_name}")

        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_pred = torch.argmax(adv_logits, dim=1)

        robust_correct += (adv_pred == labels).sum().item()

        attack_success += ((clean_pred == labels) & (adv_pred != labels)).sum().item()

        total += labels.size(0)

    clean_acc = clean_correct / total
    robust_acc = robust_correct / total

    if clean_correct == 0:
        attack_success_rate = 0.0
    else:
        attack_success_rate = attack_success / clean_correct

    return clean_acc, robust_acc, attack_success_rate


def save_results_csv(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "attack",
        "epsilon",
        "alpha",
        "steps",
        "clean_acc",
        "robust_acc",
        "attack_success_rate"
    ]

    with open(save_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"攻击结果已保存到: {save_path}")


def plot_attack_curves(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fgsm_results = [r for r in results if r["attack"] == "fgsm"]
    pgd_results = [r for r in results if r["attack"] == "pgd"]

    fgsm_eps = [r["epsilon"] for r in fgsm_results]
    fgsm_robust = [r["robust_acc"] for r in fgsm_results]
    fgsm_asr = [r["attack_success_rate"] for r in fgsm_results]

    pgd_eps = [r["epsilon"] for r in pgd_results]
    pgd_robust = [r["robust_acc"] for r in pgd_results]
    pgd_asr = [r["attack_success_rate"] for r in pgd_results]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fgsm_eps, fgsm_robust, marker="o", label="FGSM Robust Acc")
    plt.plot(pgd_eps, pgd_robust, marker="o", label="PGD Robust Acc")
    plt.xlabel("Epsilon")
    plt.ylabel("Robust Accuracy")
    plt.title("Epsilon vs Robust Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(fgsm_eps, fgsm_asr, marker="o", label="FGSM ASR")
    plt.plot(pgd_eps, pgd_asr, marker="o", label="PGD ASR")
    plt.xlabel("Epsilon")
    plt.ylabel("Attack Success Rate")
    plt.title("Epsilon vs Attack Success Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"攻击曲线已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="simple_cnn_mnist.pth"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=40
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = SimpleCNN().to(device)

    state_dict = torch.load(
        args.model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.eval()

    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    results = []

    for attack_name in ["fgsm", "pgd"]:
        for epsilon in epsilons:
            print(f"正在评估: attack={attack_name}, epsilon={epsilon}")

            clean_acc, robust_acc, attack_success_rate = evaluate_attack(
                model=model,
                data_loader=test_loader,
                device=device,
                attack_name=attack_name,
                epsilon=epsilon,
                alpha=args.alpha,
                steps=args.steps,
                max_samples=args.max_samples
            )

            row = {
                "attack": attack_name,
                "epsilon": epsilon,
                "alpha": "-" if attack_name == "fgsm" else args.alpha,
                "steps": 1 if attack_name == "fgsm" else args.steps,
                "clean_acc": clean_acc,
                "robust_acc": robust_acc,
                "attack_success_rate": attack_success_rate
            }

            results.append(row)

            print(
                f"Clean Acc: {clean_acc:.4f} | "
                f"Robust Acc: {robust_acc:.4f} | "
                f"ASR: {attack_success_rate:.4f}"
            )

    save_results_csv(
        results,
        "results/attack_results.csv"
    )

    plot_attack_curves(
        results,
        "results/attack_curve_compare.png"
    )


if __name__ == "__main__":
    main()