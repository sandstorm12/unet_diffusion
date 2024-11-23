import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class SimpleNoiseScheduler(object):
    def __init__(self, num_steps=1000, alpha_start=1e-4, alpha_end=0.02, device='cpu'):
        self._num_steps = num_steps
        self._alpha_start = alpha_start
        self._alpha_end = alpha_end

        self._alpha_bars = self._compute_alpha_bar(self._alpha_start, self._alpha_end, self._num_steps, device=device)
        self._betas = self._compute_beta(self._alpha_bars)
        self._alphas = 1 - self._betas

    def _compute_alpha_bar(self, start, end, T, device):
        """Compute the cumulative alpha_bar using the cosine schedule."""
        timesteps = torch.arange(0, T + 1, dtype=torch.float32, device=device)
        alpha_bar = torch.cos((timesteps / T) * (torch.pi / 2)) ** 2
        # alpha_bar = start + (end - start) * alpha_bar  # Scale between start and end
        
        return alpha_bar

    def _compute_beta(self, alpha_bar):
        """Compute beta values (noise variance) from alpha_bar."""
        alpha_bar = alpha_bar
        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])  # Derive beta from alpha_bar
        beta = torch.clip(beta, 0, 0.999)  # Ensure beta values are valid
        
        return beta

    def get_alphas(self):
        return self._alphas

    def get_alpha_bars(self):
        return self._alpha_bars

    def get_betas(self):
        return self._betas

    def sample_noisy_image(self, images, ts):
        batch_size = images.shape[0]

        noise = torch.randn_like(images, device=images.device)

        alpha_bar_t = self._alpha_bars[ts].view(batch_size, 1, 1, 1)

        images_noisy = torch.sqrt(alpha_bar_t) * images + torch.sqrt(1 - alpha_bar_t) * noise

        return images_noisy, noise


if __name__ == "__main__":
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    from torchvision.datasets import MNIST

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PARAM_NUM_STEPS = 500

    def _load_dataset():
        return MNIST(root='/tmp', download=True, transform=ToTensor())
    
    dataset = _load_dataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    scheduler = SimpleNoiseScheduler(num_steps=PARAM_NUM_STEPS, device=device)
    alpha_bars = scheduler.get_alpha_bars()
    print(alpha_bars)

    for images, labels in dataloader:
        for i in range(PARAM_NUM_STEPS):
            images = images.to(device)
            labels = labels.to(device)

            print(torch.min(images), torch.max(images))

            ts = torch.LongTensor([i] * len(images)).to(device)
            print(images.shape, ts.shape)
            noisy_images, noises = scheduler.sample_noisy_image(images, ts)

            print(noisy_images.shape, noisy_images.dtype)
            print(noises.shape, noises.dtype)

            print("Step", i)
            cv2.imshow("Image 0", cv2.resize(noisy_images[0, 0].cpu().numpy(), (512, 512), interpolation=cv2.INTER_CUBIC))
            cv2.imshow("Image 1", cv2.resize(noisy_images[1, 0].cpu().numpy(), (512, 512), interpolation=cv2.INTER_CUBIC))
            cv2.imshow("Image 2", cv2.resize(noisy_images[2, 0].cpu().numpy(), (512, 512), interpolation=cv2.INTER_CUBIC))
            cv2.imshow("Image 3", cv2.resize(noisy_images[3, 0].cpu().numpy(), (512, 512), interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(30)
            
            # break
        # break
