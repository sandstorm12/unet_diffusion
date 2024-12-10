import torch
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm

from unet import UNet
from simple_noise_scheduler import SimpleNoiseScheduler

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


PARAM_NUM_STEPS = 500
PARAM_NUM_CLASSES = 10
PARAM_EPOCHS = 200
PARAM_LOAD_SAVE = False


def _load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return MNIST(root='/tmp', download=True, transform=transform)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = _load_dataset()
    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=True, num_workers=4)

    scheduler = SimpleNoiseScheduler(num_steps=PARAM_NUM_STEPS,
                                     device=device)

    criterion = torch.nn.MSELoss()

    model = UNet(timesteps=PARAM_NUM_STEPS,
                 classes=PARAM_NUM_CLASSES).to(device)
    
    if PARAM_LOAD_SAVE:
        model.load_state_dict(torch.load("model.pth"))
    
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    min_loss = float('inf')

    pbar = tqdm(range(PARAM_EPOCHS))
    for epoch in pbar:
        losses = []
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            ts = torch.randint(0, PARAM_NUM_STEPS, (len(images),),
                               device=device, dtype=torch.long)
            noisy_images, noises = scheduler.sample_noisy_image(images, ts)

            outputs = model(noisy_images, ts, labels)

            loss = criterion(outputs, noises)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())

        mean_loss = np.mean(losses)
        pbar.set_description(f"Loss: {np.mean(losses):.4f}")

        if mean_loss < min_loss:
            min_loss = mean_loss
            print(f"Saving model... {mean_loss:.4f}")
            torch.save(model.state_dict(), "model.pth")
