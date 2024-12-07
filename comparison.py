import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


PATH_GT = "/home/hamid/Documents/phd/Homework/A02_Proj/unet_diffusion/artifacts/image_8_100.jpg"
PATH_NOISY = "/home/hamid/Documents/phd/Homework/A02_Proj/unet_diffusion/artifacts/image_noisy_8_100.jpg"
PATH_DIFFUSION = "/home/hamid/Documents/phd/Homework/A02_Proj/unet_diffusion/artifacts/image_final_8_100.jpg"


# Function to compute RMSE
def compute_rmse(gt_image, noisy_image):
    error = gt_image.astype(np.float64) - noisy_image.astype(np.float64)
    rmse = np.sqrt(np.mean(error ** 2))
    return rmse


def _print_metrics(img_gt, img):
    # Compute RMSE
    rmse_value = compute_rmse(img_gt, img)
    print("RMSE noisy:", rmse_value)

    # Compute SSIM (for multichannel images, pass the channel axis explicitly)
    ssim_value = ssim(img_gt, img, channel_axis=-1)
    print(f"SSIM noisy: {ssim_value}")

    # Compute PSNR
    psnr_value = psnr(img_gt, img, data_range=255)
    print(f"PSNR noisy: {psnr_value}")


def _apply_arithmetic_mean(img, size=15):
    img_filtered = img.copy()

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            # Define the neighborhood
            start_x = max(x - size, 0)
            end_x = min(x + size, img.shape[1])
            start_y = max(y - size, 0)
            end_y = min(y + size, img.shape[0])

            # Calculate the mean of the neighborhood
            img_filtered[y, x] = np.mean(img[start_y:end_y, start_x:end_x])

    return img_filtered


def _apply_arithmetic_median(img, size=15):
    img_filtered = img.copy()

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            # Define the neighborhood
            start_x = max(x - size, 0)
            end_x = min(x + size, img.shape[1])
            start_y = max(y - size, 0)
            end_y = min(y + size, img.shape[0])

            # Calculate the median of the neighborhood
            img_filtered[y, x] = np.median(img[start_y:end_y, start_x:end_x])

    return img_filtered


def _apply_anisotropic_diffusion(image, iterations=1000, p1=0.35, p2=5, option=1):
    diff_image = image.copy().astype(np.float32)
    
    for _ in range(iterations):
        # Calculate the gradient in all direction
        grad_north = np.roll(diff_image, -1, axis=0) - diff_image
        grad_south = np.roll(diff_image, 1, axis=0) - diff_image
        grad_east = np.roll(diff_image, -1, axis=1) - diff_image
        grad_west = np.roll(diff_image, 1, axis=1) - diff_image
        
        if option == 1:
            # Exponential diffusion function
            c_north = np.exp(-(grad_north / p2) ** 2)
            c_south = np.exp(-(grad_south / p2) ** 2)
            c_east = np.exp(-(grad_east / p2) ** 2)
            c_west = np.exp(-(grad_west / p2) ** 2)
        elif option == 2:
            # Quadratic diffusion function
            c_north = 1.0 / (1.0 + (grad_north / p2) ** 2)
            c_south = 1.0 / (1.0 + (grad_south / p2) ** 2)
            c_east = 1.0 / (1.0 + (grad_east / p2) ** 2)
            c_west = 1.0 / (1.0 + (grad_west / p2) ** 2)

        # Update the image based on the diffusion equation
        diff_image += p1 * (
            c_north * grad_north + c_south * grad_south +
            c_east * grad_east + c_west * grad_west
        )

    # Clip the image to the valid range
    diff_image = np.clip(diff_image, 0, 255).astype(np.uint8)
    
    return diff_image


if __name__ == "__main__":
    img_ns = cv2.cvtColor(cv2.imread(PATH_NOISY), cv2.COLOR_BGR2GRAY)
    img_gt = cv2.cvtColor(cv2.imread(PATH_GT), cv2.COLOR_BGR2GRAY)
    img_df = cv2.cvtColor(cv2.imread(PATH_DIFFUSION), cv2.COLOR_BGR2GRAY)

    _print_metrics(img_gt, img_ns)
    _print_metrics(img_gt, img_df)

    img_ftr = _apply_arithmetic_mean(img_ns)
    # img_ftr = _apply_arithmetic_median(img_ftr)
    # img_ftr = _apply_anisotropic_diffusion(img_ftr)

    _print_metrics(img_gt, img_ftr)

    cv2.imshow("Image noisy", img_ns)
    cv2.imshow("Image filterd", img_ftr)
    cv2.waitKey(0)
