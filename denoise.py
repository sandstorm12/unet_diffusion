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
PARAM_NOISE_STEPS = 100
PARAM_NP_MIN = 20
PARAM_UPSCALE = True
PARAM_VISUALIZE = False
PARAM_LABEL = 9


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


def _load_random_image(dataset, device, label=None):
    while True:
        idx_rand = np.random.randint(0, len(dataset))
        image, label_gt = dataset.__getitem__(idx_rand)
        image = image.unsqueeze(0).to(device)

        if label is not None and label_gt == label:
            label = torch.LongTensor([label_gt]).to(device)
            break

    return image, label


def _add_noise(image, scheduler, device):
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

    image, label = _load_random_image(dataset, device, label=PARAM_LABEL)
    image_noisy, noise = _add_noise(image, scheduler, device)
    print(label)

    image_np = image.detach().cpu().numpy()[0, 0]
    image_noisy_np = image_noisy.detach().cpu().numpy()[0, 0]

    image_np = (image_np - image_np.min()) \
        / (image_np.max() - image_np.min()) * 255
    image_noisy_np = (image_noisy_np - image_noisy_np.min()) \
        / (image_noisy_np.max() - image_noisy_np.min()) * 255
    
    image_np = image_np.astype(np.uint8)
    image_noisy_np = image_noisy_np.astype(np.uint8)

    if PARAM_UPSCALE:
        image_np = cv2.resize(image_np,
                                (512, 512),
                                interpolation=cv2.INTER_CUBIC)
        image_noisy_np = cv2.resize(image_noisy_np,
                                (512, 512),
                                interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("./artifacts/image_{}_{}.jpg".format(
        label.item(), PARAM_NOISE_STEPS), image_np)
    cv2.imwrite("./artifacts/image_noisy_{}_{}.jpg".format(
        label.item(), PARAM_NOISE_STEPS), image_noisy_np)

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

            image_np = image_prv.detach().cpu().numpy()[0, 0]
            noise_np = noises.detach().cpu().numpy()[0, 0]

            image_np = (image_np - image_np.min()) \
                / (image_np.max() - image_np.min()) * 255
            noise_np = (noise_np - noise_np.min()) \
                / (noise_np.max() - noise_np.min()) * 255
            
            image_np = image_np.astype(np.uint8)
            noise_np = noise_np.astype(np.uint8)

            if PARAM_UPSCALE:
                image_np = cv2.resize(image_np,
                                      (512, 512),
                                      interpolation=cv2.INTER_CUBIC)
                noise_np = cv2.resize(noise_np,
                                       (512, 512),
                                       interpolation=cv2.INTER_CUBIC)
            
            if PARAM_VISUALIZE:
                cv2.imshow("Image", image_np)
                cv2.imshow("Noises", noise_np)
                if cv2.waitKey(0) == ord('q'):
                    break

            image_noisy = image_prv

    cv2.imwrite("./artifacts/image_final_{}_{}.jpg".format(
        label.item(), PARAM_NOISE_STEPS), image_np)
