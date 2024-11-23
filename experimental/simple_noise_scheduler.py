import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class SimpleNoiseScheduler(object):
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self._num_steps = num_steps
        self._beta_start = beta_start
        self._beta_end = beta_end

        self._betas = torch.linspace(self._beta_start, self._beta_end, self._num_steps, device=device)
        self._alphas = 1.0 - self._betas
        self._alpha_start = self._alphas[0]
        self._alpha_bars = torch.cumprod(self._alphas, dim=0)

        print(self._alpha_bars)

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

    PARAM_NUM_STEPS = 100

    def _load_dataset():
        return MNIST(root='/tmp', download=True, transform=ToTensor())
    
    dataset = _load_dataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    scheduler = SimpleNoiseScheduler(num_steps=PARAM_NUM_STEPS, device=device)

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
