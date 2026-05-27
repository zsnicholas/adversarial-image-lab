import torch


def evaluate_clean(model, test_loader, device):
    """Evaluate model accuracy on clean test images."""
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    clean_acc = correct / total
    return clean_acc


def evaluate_attack(model, test_loader, attack_fn, device, **attack_kwargs):
    """Evaluate clean accuracy, robust accuracy, and attack success rate."""
    model.eval()

    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)

        adv_images = attack_fn(
            model,
            images,
            labels,
            **attack_kwargs,
        )

        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)

        clean_correct += (clean_preds == labels).sum().item()
        adv_correct += (adv_preds == labels).sum().item()
        total += labels.size(0)

    clean_acc = clean_correct / total
    robust_acc = adv_correct / total
    attack_success_rate = 1 - robust_acc

    return clean_acc, robust_acc, attack_success_rate
