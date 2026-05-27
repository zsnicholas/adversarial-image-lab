import torch
import torch.nn.functional as F


def pgd_attack(model, images, labels, epsilon, alpha, steps):
    original_images = images.clone().detach()
    adv_images = images.clone().detach()

    adv_images = adv_images + torch.empty_like(adv_images).uniform_(
        -epsilon,
        epsilon,
    )

    adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(steps):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad_sign = adv_images.grad.sign()

        adv_images = adv_images + alpha * grad_sign

        perturbation = torch.clamp(
            adv_images - original_images,
            min=-epsilon,
            max=epsilon,
        )

        adv_images = original_images + perturbation
        adv_images = torch.clamp(adv_images, 0, 1)
        adv_images = adv_images.detach()

    return adv_images
