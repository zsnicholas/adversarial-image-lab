import matplotlib.pyplot as plt
import torch


def show_adversarial_example(original_image, adv_image, save_path=None, show=True):
    original_image = original_image.detach().cpu()
    adv_image = adv_image.detach().cpu()

    if original_image.dim() == 3:
        original_image = original_image.squeeze(0)
    if adv_image.dim() == 3:
        adv_image = adv_image.squeeze(0)

    perturbation = adv_image - original_image
    perturbation_view = torch.clamp(perturbation * 10 + 0.5, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(perturbation_view, cmap="gray")
    axes[1].set_title("Perturbation")
    axes[1].axis("off")

    axes[2].imshow(adv_image, cmap="gray")
    axes[2].set_title("Adversarial Image")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
