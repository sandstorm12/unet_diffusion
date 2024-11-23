import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class SimpleNoiseScheduler(object):
    def __init__(self, alpha_start=.95, alpha_end=0.5, num_steps=20):
        self._alpha_start = alpha_start
        self._alpha_end = alpha_end
        self._num_steps = num_steps

    def sample_noisy_image(self, images, ts):
        alpha = self._alpha_start * (self._num_steps - ts) + self._alpha_end * ts
        alpha /= self._num_steps
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        noise = torch.randn_like(images, device=images.device)

        images_noisy = torch.sqrt(alpha) * images + (1 - alpha) * noise

        return images_noisy, noise


if __name__ == "__main__":
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    from torchvision.datasets import MNIST

    PARAM_NUM_STEPS = 50

    def _load_dataset():
        return MNIST(root='/tmp', download=True, transform=ToTensor())
    
    dataset = _load_dataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    scheduler = SimpleNoiseScheduler(num_steps=PARAM_NUM_STEPS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            cv2.imshow("Image 0", cv2.resize(noisy_images[0, 0].cpu().numpy(), (512, 512)))
            cv2.imshow("Image 1", cv2.resize(noisy_images[1, 0].cpu().numpy(), (512, 512)))
            cv2.imshow("Image 2", cv2.resize(noisy_images[2, 0].cpu().numpy(), (512, 512)))
            cv2.imshow("Image 3", cv2.resize(noisy_images[3, 0].cpu().numpy(), (512, 512)))
            cv2.waitKey(30)
            
            # break
        # break
