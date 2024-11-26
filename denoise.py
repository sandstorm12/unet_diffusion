import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm

from unet import UNet
from simple_noise_scheduler import SimpleNoiseScheduler

from torchvision.datasets import MNIST


PARAM_NUM_STEPS = 500
PARAM_NUM_CLASSES = 10
PARAM_NOISE_STEPS = 50
PARAM_NP_MIN = 20
PARAM_UPSCALE = True


def _load_model(device, load_save=True):
    model = UNet(timesteps=PARAM_NUM_STEPS,
                 classes=PARAM_NUM_CLASSES).to(device)

    if load_save:
        model.load_state_dict(torch.load("model.pth"))
    
    model.eval()

    return model


def _load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return MNIST(root='/tmp', download=True, transform=transform)


def _load_random_image(dataset):
    idx_rand = np.random.randint(0, len(dataset))

    image, label_gt = dataset.__getitem__(idx_rand)
    image = image.unsqueeze(0).to(device)
    label = torch.LongTensor([label_gt]).to(device)

    return image, label


def _add_noise(image, scheduler):
    step = torch.LongTensor([PARAM_NOISE_STEPS - 1]).to(device)
    image_noisy, noises = scheduler.sample_noisy_image(image, step)

    return image_noisy, noises


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(device, load_save=True)

    scheduler = SimpleNoiseScheduler(num_steps=PARAM_NUM_STEPS, device=device)
    alphas = scheduler.get_alphas()
    alpha_bars = scheduler.get_alpha_bars()
    betas = scheduler.get_betas()

    dataset = _load_dataset()

    image, label = _load_random_image(dataset)
    image_noisy, noise = _add_noise(image, scheduler)

    with torch.no_grad():
        for step in tqdm(range(PARAM_NOISE_STEPS - 1, 0, -1)):
            step = torch.LongTensor([step]).to(device)
            noises = model(image_noisy, step, label)

            if step > PARAM_NP_MIN:
                image_prv = (1 / torch.sqrt(alphas[step])) \
                    * (image_noisy - (betas[step]) \
                       / torch.sqrt(1 - alpha_bars[step]) * noises) \
                    + torch.sqrt(betas[step]) \
                        * torch.randn_like(image_noisy, device=device)
            else:
                image_prv = (1 / torch.sqrt(alphas[step])) \
                    * (image_noisy - (betas[step]) \
                       / torch.sqrt(1 - alpha_bars[step]) * noises)

            print(image_prv.shape, image_prv.min(), image_prv.max())

            image_np = image_prv.detach().cpu().numpy()[0, 0]
            noises_np = noises.detach().cpu().numpy()[0, 0]

            image_np = (image_np - image_np.min()) \
                / (image_np.max() - image_np.min())
            noises_np = (noises_np - noises_np.min()) \
                / (noises_np.max() - noises_np.min())

            if PARAM_UPSCALE:
                image_np = cv2.resize(image_np,
                                      (512, 512),
                                      interpolation=cv2.INTER_CUBIC)
                noises_np = cv2.resize(noises_np,
                                       (512, 512),
                                       interpolation=cv2.INTER_CUBIC)
            
            cv2.imshow("Image", image_np)
            cv2.imshow("Noises", noises_np)
            if cv2.waitKey(0) == ord('q'):
                break

            image_noisy = image_prv
