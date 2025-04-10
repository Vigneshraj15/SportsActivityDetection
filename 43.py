import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

# 1. Set path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Load the image
image_path = r"E:\WINTER SEMESTER 24-45\DIGITAL IMAGE PROCESSING\archive\test\1.png"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# 3. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. Threshold using Otsu's method
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. Invert if necessary (Tesseract prefers black text on white)
white_pixels = np.sum(thresh == 255)
black_pixels = np.sum(thresh == 0)
if black_pixels > white_pixels:
    print("Inverting image as it has white text on black background")
    thresh = cv2.bitwise_not(thresh)

# 6. Morphological operations to improve OCR
kernel = np.ones((2, 2), np.uint8)
processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 7. OCR Extraction
custom_config = r'--oem 3 --psm 6'  # Assume a single uniform block of text
extracted_text = pytesseract.image_to_string(processed, config=custom_config).strip()

# 8. Debug output
if extracted_text:
    print("✅ Extracted Text:\n", extracted_text)
else:
    print("❌ No text was extracted. Try adjusting the image or PSM mode.")

# 9. Show images using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed, cmap='gray')
plt.title('Processed Image for OCR')
plt.axis('off')
plt.show()

# 10. Show text in a popup OpenCV window
if extracted_text:
    display_image = np.ones((300, 800, 3), dtype=np.uint8) * 255
    y_offset = 30
    for i, line in enumerate(extracted_text.split('\n')):
        cv2.putText(display_image, line, (10, y_offset + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Extracted Text", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
