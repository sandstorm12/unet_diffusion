import cv2
import torch
import torchvision.transforms as transforms

from tqdm import tqdm

from unet import UNet
from simple_noise_scheduler import SimpleNoiseScheduler

from torchvision.datasets import MNIST



PARAM_NUM_STEPS = 500
PARAM_NUM_CLASSES = 10


def _load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return MNIST(root='/tmp', download=True, transform=transform)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label = torch.LongTensor([7]).to(device)
    timestep = 500

    model = UNet(timesteps=PARAM_NUM_STEPS, classes=PARAM_NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    scheduler = SimpleNoiseScheduler(num_steps=PARAM_NUM_STEPS, device=device)

    dataset = _load_dataset()
    image, label_gt = dataset.__getitem__(4)
    image = image.unsqueeze(0).to(device)
    # label = torch.LongTensor([label_gt]).to(device)
    step = torch.LongTensor([timestep - 1]).to(device)
    print(image.shape, label.shape)

    alphas = scheduler.get_alphas()
    alpha_bars = scheduler.get_alpha_bars()
    betas = scheduler._get_betas()
    image_noisy, noises = scheduler.sample_noisy_image(image, step)

    with torch.no_grad():
        for step in tqdm(range(timestep - 1, -1, -1)):
            step = torch.LongTensor([step]).to(device)
            noises = model(image_noisy, step, label)

            image_prv = (1 / torch.sqrt(alphas[step])) \
                * (image_noisy - (betas[step]) / torch.sqrt(1 - alpha_bars[step]) * noises)
                # + torch.sqrt(betas[step]) * torch.randn_like(image_noisy, device=device)

            image_np = image_noisy.detach().cpu().numpy()[0, 0] * 255
            noises_np = noises.detach().cpu().numpy()[0, 0] * 255

            cv2.imshow("Image 0", cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_CUBIC))
            cv2.imshow("Noises 0", cv2.resize(noises_np, (512, 512), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(0) == ord('q'):
                break

            image_noisy = image_prv

    cv2.destroyAllWindows()
