import cv2
import torch

from unet import UNet
from simple_noise_scheduler import SimpleNoiseScheduler

from torchvision.datasets import MNIST
import torchvision.transforms as transforms


PARAM_NUM_STEPS = 100
PARAM_NUM_CLASSES = 10


ALPHA_START = 0.95
ALPHA_END = 0.5


def _load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return MNIST(root='/tmp', download=True, transform=transform)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label = torch.LongTensor([9]).to(device)
    timestep = 10

    model = UNet(timesteps=PARAM_NUM_STEPS, classes=PARAM_NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    scheduler = SimpleNoiseScheduler(num_steps=PARAM_NUM_STEPS)

    dataset = _load_dataset()
    image, label = dataset.__getitem__(2)
    image = image.unsqueeze(0).to(device)
    label = torch.LongTensor([label]).to(device)
    step = torch.LongTensor([timestep]).to(device)
    print(image.shape, label.shape)

    image_noisy, noises = scheduler.sample_noisy_image(image, step)
    # noises_pred = model(image, step, label)

    with torch.no_grad():
        # image = torch.randn((1, 1, 28, 28), device=device)
        for step in range(timestep - 1, -1, -1):
            print("Step", step)

            step = torch.LongTensor([step]).to(device)
            noises = model(image, torch.tensor(step), label)

            # alpha = ALPHA_START * (PARAM_NUM_STEPS - step) + ALPHA_END * step
            # alpha /= PARAM_NUM_STEPS
            # alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # print("alpha", alpha)
            # image = (image - (1 - alpha) * noises) / torch.sqrt(alpha)

            # print(torch.min(noises), torch.mean(noises), torch.max(noises))
            # print(torch.min(image), torch.mean(image), torch.max(image))

            image_np = image_noisy.detach().cpu().numpy()[0, 0] * 255
            noises_np = noises.detach().cpu().numpy()[0, 0] * 255
            # cv2.imshow("Image 0", cv2.resize(image_np, (512, 512)))
            cv2.imshow("Image 0", image_np)
            cv2.imshow("Noises 0", noises_np)
            if cv2.waitKey(0) == ord('q'):
                break

    cv2.destroyAllWindows()
