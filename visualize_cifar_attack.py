import argparse
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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


def fgsm_attack(model, image, label, loss_fn, epsilon):
    image = image.clone().detach().requires_grad_(True)

    logits = model(normalize_cifar10(image))
    loss = loss_fn(logits, label)

    model.zero_grad()
    loss.backward()

    grad_sign = image.grad.sign()
    adv_image = image + epsilon * grad_sign
    adv_image = torch.clamp(adv_image, 0, 1)

    return adv_image.detach()


def pgd_attack(model, image, label, loss_fn, epsilon, alpha, steps):
    ori_image = image.clone().detach()
    adv_image = ori_image.clone().detach()

    for _ in range(steps):
        adv_image.requires_grad_(True)

        logits = model(normalize_cifar10(adv_image))
        loss = loss_fn(logits, label)

        model.zero_grad()
        loss.backward()

        grad_sign = adv_image.grad.sign()
        adv_image = adv_image + alpha * grad_sign

        perturbation = torch.clamp(
            adv_image - ori_image,
            min=-epsilon,
            max=epsilon
        )

        adv_image = torch.clamp(
            ori_image + perturbation,
            min=0,
            max=1
        ).detach()

    return adv_image


def predict(model, image):
    with torch.no_grad():
        logits = model(normalize_cifar10(image))
        pred = torch.argmax(logits, dim=1)
    return pred.item()


def tensor_to_image(image_tensor):
    image = image_tensor.squeeze(0).detach().cpu()
    image = image.permute(1, 2, 0)
    return image.numpy()


def visualize(original_image, adv_image, true_label, clean_pred, adv_pred, attack, epsilon, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    perturbation = adv_image - original_image

    original_np = tensor_to_image(original_image)
    adv_np = tensor_to_image(adv_image)

    perturb_np = perturbation.squeeze(0).detach().cpu()
    perturb_np = perturb_np.permute(1, 2, 0).numpy()

    perturb_show = (perturb_np - perturb_np.min()) / (perturb_np.max() - perturb_np.min() + 1e-8)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_np)
    plt.title(
        f"Original\nTrue: {CIFAR10_CLASSES[true_label]}\nPred: {CIFAR10_CLASSES[clean_pred]}"
    )
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(perturb_show)
    plt.title(f"Perturbation\n{attack.upper()} eps={epsilon}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(adv_np)
    plt.title(
        f"Adversarial\nPred: {CIFAR10_CLASSES[adv_pred]}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"可视化图片已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attack",
        type=str,
        default="fgsm",
        choices=["fgsm", "pgd"]
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
        "--index",
        type=int,
        default=0
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
        batch_size=1,
        shuffle=False
    )

    model = CIFAR10DeepCNN().to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    selected_image = None
    selected_label = None

    for i, (image, label) in enumerate(test_loader):
        if i == args.index:
            selected_image = image.to(device)
            selected_label = label.to(device)
            break

    if selected_image is None:
        raise ValueError("index 超出测试集范围")

    clean_pred = predict(model, selected_image)

    if args.attack == "fgsm":
        adv_image = fgsm_attack(
            model=model,
            image=selected_image,
            label=selected_label,
            loss_fn=loss_fn,
            epsilon=args.epsilon
        )

        save_path = f"results/cifar_fgsm_eps_{args.epsilon}_example.png"

    else:
        adv_image = pgd_attack(
            model=model,
            image=selected_image,
            label=selected_label,
            loss_fn=loss_fn,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.steps
        )

        save_path = f"results/cifar_pgd_eps_{args.epsilon}_example.png"

    adv_pred = predict(model, adv_image)

    print("========== CIFAR-10 Visualization ==========")
    print(f"Attack: {args.attack}")
    print(f"True Label: {CIFAR10_CLASSES[selected_label.item()]}")
    print(f"Clean Pred: {CIFAR10_CLASSES[clean_pred]}")
    print(f"Adv Pred: {CIFAR10_CLASSES[adv_pred]}")

    visualize(
        original_image=selected_image,
        adv_image=adv_image,
        true_label=selected_label.item(),
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        attack=args.attack,
        epsilon=args.epsilon,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
