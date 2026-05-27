import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from eval.metrics import evaluate_attack, evaluate_clean
from models.simple_cnn import SimpleCNN
from utils.visualize import show_adversarial_example


def load_test_loader(data_dir, batch_size):
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FGSM or PGD attacks.")
    parser.add_argument("--model-path", type=str, default="simple_cnn_mnist.pth")
    parser.add_argument("--attack", choices=["none", "fgsm", "pgd"], default="fgsm")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = load_test_loader(args.data_dir, args.batch_size)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if args.attack == "none":
        clean_acc = evaluate_clean(model, test_loader, device)
        print(f"Model: SimpleCNN")
        print(f"Attack: None")
        print(f"Clean Acc: {clean_acc * 100:.2f}%")
        return

    if args.attack == "fgsm":
        attack_fn = fgsm_attack
        attack_kwargs = {"epsilon": args.epsilon}
    else:
        attack_fn = pgd_attack
        attack_kwargs = {
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.steps,
        }

    clean_acc, robust_acc, attack_success_rate = evaluate_attack(
        model,
        test_loader,
        attack_fn,
        device,
        **attack_kwargs,
    )

    print("| Model | Attack | Epsilon | Alpha | Steps | Clean Acc | Robust Acc | Attack Success Rate |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    print(
        f"| SimpleCNN | {args.attack.upper()} | {args.epsilon} | "
        f"{args.alpha if args.attack == 'pgd' else '-'} | "
        f"{args.steps if args.attack == 'pgd' else 1} | "
        f"{clean_acc * 100:.2f}% | {robust_acc * 100:.2f}% | "
        f"{attack_success_rate * 100:.2f}% |"
    )

    if args.visualize:
        images, labels = next(iter(test_loader))
        images = images.to(device)
        labels = labels.to(device)

        adv_images = attack_fn(model, images, labels, **attack_kwargs)
        show_adversarial_example(images[0], adv_images[0])


if __name__ == "__main__":
    main()
