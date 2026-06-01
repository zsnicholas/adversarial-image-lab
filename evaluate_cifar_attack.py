import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cifar_deep_cnn import CIFAR10DeepCNN


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def normalize_cifar10(images):
    mean = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def fgsm_attack(model, images, labels, loss_fn, epsilon):
    images = images.clone().detach().requires_grad_(True)

    logits = model(normalize_cifar10(images))
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

        logits = model(normalize_cifar10(adv_images))
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


def evaluate(model, data_loader, device, attack, epsilon, alpha, steps):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total = 0
    clean_correct = 0
    robust_correct = 0
    attack_success = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_logits = model(normalize_cifar10(images))
            clean_pred = torch.argmax(clean_logits, dim=1)

        clean_correct += (clean_pred == labels).sum().item()

        if attack == "none":
            adv_images = images

        elif attack == "fgsm":
            adv_images = fgsm_attack(
                model=model,
                images=images,
                labels=labels,
                loss_fn=loss_fn,
                epsilon=epsilon
            )

        elif attack == "pgd":
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
            raise ValueError(f"不支持的攻击类型: {attack}")

        with torch.no_grad():
            adv_logits = model(normalize_cifar10(adv_images))
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attack",
        type=str,
        default="none",
        choices=["none", "fgsm", "pgd"]
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_cifar10_deep_cnn.pth"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.005
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    transform = transforms.ToTensor()

    test_dataset = datasets.CIFAR10(
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

    model = CIFAR10DeepCNN().to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    clean_acc, robust_acc, attack_success_rate = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        attack=args.attack,
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps
    )

    print("========== CIFAR-10 Attack Evaluation ==========")
    print(f"Attack: {args.attack}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Alpha: {args.alpha}")
    print(f"Steps: {args.steps}")
    print(f"Clean Acc: {clean_acc:.4f}")
    print(f"Robust Acc: {robust_acc:.4f}")
    print(f"Attack Success Rate: {attack_success_rate:.4f}")


if __name__ == "__main__":
    main()