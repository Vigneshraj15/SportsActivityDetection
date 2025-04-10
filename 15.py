import cv2
import numpy as np
import matplotlib.pyplot as plt

def bilinear_interpolation(img, new_width, new_height):
    height, width, channels = img.shape
    resized_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    x_ratio = float(width - 1) / (new_width - 1) if new_width > 1 else 0
    y_ratio = float(height - 1) / (new_height - 1) if new_height > 1 else 0

    for i in range(new_height):
        for j in range(new_width):
            x_l = int(x_ratio * j)
            y_l = int(y_ratio * i)
            x_h = min(x_l + 1, width - 1)
            y_h = min(y_l + 1, height - 1)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = img[y_l, x_l]
            b = img[y_l, x_h]
            c = img[y_h, x_l]
            d = img[y_h, x_h]

            pixel = (
                a * (1 - x_weight) * (1 - y_weight) +
                b * x_weight * (1 - y_weight) +
                c * y_weight * (1 - x_weight) +
                d * x_weight * y_weight
            )

            resized_img[i, j] = pixel

    return resized_img

# Load the original image
image_path = r'E:\WINTER SEMESTER 24-45\DIGITAL IMAGE PROCESSING\archive\test\a4.jpg'  # Replace with your image path
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize image to 200x200
resized_img = bilinear_interpolation(img_rgb, 200, 200)

# Display original and resized images
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Resized Image (Bilinear)")
plt.imshow(resized_img)
plt.axis('off')

plt.show()
