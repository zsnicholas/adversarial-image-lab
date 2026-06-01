import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
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


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epsilon):
    model.train()

    total = 0
    correct = 0
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        adv_images = fgsm_attack(
            model=model,
            images=images,
            labels=labels,
            loss_fn=loss_fn,
            epsilon=epsilon
        )

        logits = model(adv_images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


def evaluate_clean(model, test_loader, loss_fn, device):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/adv_simple_cnn_mnist.pth"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = SimpleCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epsilon=args.epsilon
        )

        test_loss, test_acc = evaluate_clean(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Clean Test Loss: {test_loss:.4f} "
            f"Clean Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"保存最佳对抗训练模型: {args.save_path}")

    print(f"训练完成，最佳 Clean Test Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()