import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class SimpleNoiseScheduler(object):
    def __init__(self, num_steps=1000, device='cpu'):
        self._num_steps = num_steps

        self._alpha_bars = self._compute_alpha_bar(
            self._num_steps, device=device)
        self._betas = self._compute_beta(self._alpha_bars)
        self._alphas = 1 - self._betas

    def _compute_alpha_bar(self, T, device):
        timesteps = torch.arange(0, T + 1, dtype=torch.float32, device=device)
        alpha_bar = torch.cos((timesteps / T) * (torch.pi / 2)) ** 2
        
        return alpha_bar

    def _compute_beta(self, alpha_bar):
        alpha_bar = alpha_bar
        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        beta = torch.clip(beta, 0, 0.999)
        
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

        images_noisy = torch.sqrt(alpha_bar_t) * images \
            + torch.sqrt(1 - alpha_bar_t) * noise

        return images_noisy, noise


# Just for test
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

    progress = True
    for images, labels in dataloader:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            f"./artifacts/{labels[0]}.mp4",
            fourcc,
            60,
            (512, 512),
        )

        for i in range(PARAM_NUM_STEPS):
            images = images.to(device)
            labels = labels.to(device)

            ts = torch.LongTensor([i] * len(images)).to(device)
            noisy_images, noises = scheduler.sample_noisy_image(images, ts)

            image_np = noisy_images.detach().cpu().numpy()[0, 0]

            image_np = (image_np - image_np.min()) \
                / (image_np.max() - image_np.min()) * 255        
            
            image_np = image_np.astype(np.uint8)

            image_np = cv2.resize(image_np, (512, 512),
                                  interpolation=cv2.INTER_CUBIC)

            title = f"Image"
            cv2.imshow(title, image_np)
            writer.write(cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR))
            if cv2.waitKey(1) == ord('q'):
                progress = False

        if not progress:
            break
