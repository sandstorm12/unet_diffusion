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
PARAM_NP_MIN = 20
PARAM_UPSCALE = True
PARAM_LABEL = 9


def _load_model(device, load_save=True):
    model = UNet(timesteps=PARAM_NUM_STEPS,
                 classes=PARAM_NUM_CLASSES).to(device)

    if load_save:
        model.load_state_dict(torch.load("model.pth"))
    
    model.eval()

    return model


def _init_writer(path="./artifacts/generate.mp4"):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        path,
        fourcc,
        60,
        (512, 512),
    )

    return writer


def _generate_noise(label):
    label = torch.LongTensor([label]).to(device)
    noise = torch.randn(1, 1, 28, 28).to(device)

    return noise, label


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(device, load_save=True)

    scheduler = SimpleNoiseScheduler(num_steps=PARAM_NUM_STEPS, device=device)
    alphas = scheduler.get_alphas()
    alpha_bars = scheduler.get_alpha_bars()
    betas = scheduler.get_betas()

    image, label = _generate_noise(label=PARAM_LABEL)

    writer_image = _init_writer("./artifacts/image.mp4")
    writer_noise = _init_writer("./artifacts/noise.mp4")
    with torch.no_grad():
        for step in tqdm(range(PARAM_NUM_STEPS - 1, 0, -1)):
            step = torch.LongTensor([step]).to(device)
            noises = model(image, step, label)

            if step > PARAM_NP_MIN:
                image_prv = (1 / torch.sqrt(alphas[step])) \
                    * (image - (betas[step]) \
                       / torch.sqrt(1 - alpha_bars[step]) * noises) \
                    + torch.sqrt(betas[step]) \
                        * torch.randn_like(image, device=device)
            else:
                image_prv = (1 / torch.sqrt(alphas[step])) \
                    * (image - (betas[step]) \
                       / torch.sqrt(1 - alpha_bars[step]) * noises)

            print(image_prv.shape, image_prv.min(), image_prv.max())

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
                                      interpolation=cv2.INTER_CUBIC
                                      )
                noise_np = cv2.resize(noise_np,
                                       (512, 512),
                                       interpolation=cv2.INTER_CUBIC
                                       )
            
            cv2.imshow("Image", image_np)
            cv2.imshow("Noises", noise_np)
            if cv2.waitKey(1) == ord('q'):
                break
            
            print(image_np.shape, noise_np.shape)
            writer_image.write(cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR))
            writer_noise.write(cv2.cvtColor(noise_np, cv2.COLOR_GRAY2BGR))

            image = image_prv

    writer_image.release()
    writer_noise.release()
