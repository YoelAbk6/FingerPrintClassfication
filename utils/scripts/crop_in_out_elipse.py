import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ("NIST302a-M",
#  "/home/uoriko/FingerPrintClassfication/data/train/NIST302_TrainData.list",
#  "/home/uoriko/FingerPrintClassfication/data/test/NIST302_TestData.list"),
# ("NIST4",
#  "/home/uoriko/FingerPrintClassfication/data/train/NIST4_TrainData.list",
#  "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list"),
# ("SOCOfing",
#  "/home/uoriko/FingerPrintClassfication/data/train/SOCOfing_TrainData.list",
#  "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list"),

DS = [
    (
        "/home/uoriko/FingerPrintClassfication/data/train/NIST4_TrainData.list",
        "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/*/train/",
    ),
    (
        "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list",
        "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/*/test/",
    ),
    (
        "/home/uoriko/FingerPrintClassfication/data/train/SOCOfing_TrainData.list",
        "/home/uoriko/FingerPrintClassfication/data/SOCOFing/*/train/",
    ),
    (
        "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list",
        "/home/uoriko/FingerPrintClassfication/data/SOCOFing/*/test/",
    ),
]

data_dir = "C:/Users/nsahalu/OneDrive - Intel Corporation/Desktop/Studies/FinalProject/FingerPrintClassfication/data/"

dirs = [
    # f"{data_dir}NIST302/auxiliary/flat/M/500/plain/png/regular",
    f"{data_dir}SOCOFing/Real",
    f"{data_dir}sd04/png_txt/figs"
]

figs = [
    ("inner", 255, 0),
    ("outer", 0, 255),
]
crop_zise = [
    ("50", 0.25),
    ("60", 0.3),
]

for ds_list, out_dir in DS:
    for crop_str, crop_pram in crop_zise:
        for fig, mask, fill_color in figs:

            output_dir = out_dir.replace("*", f'{fig}{crop_str}')
            os.makedirs(output_dir, exist_ok=True)
            print(f"Starting {output_dir} {fig} {crop_str}")

            # Iterate over all fingerprint images in the directory
            with open(ds_list, 'r') as f:
                for line in f:
                    # Load the image
                    img = cv2.imread(line.strip())
                    # Convert the image to grayscale and apply thresholding
                    inverted_img = cv2.bitwise_not(img)
                    gray = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)

                    x, y = 0, 0
                    h, w = gray.shape[:2]

                    # plt.imshow(gray, cmap='gray')
                    # plt.show()
                    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                    # plt.imshow(thresh, cmap='gray')
                    # plt.show()

                    erode_kernel = np.ones((3, 3), np.uint8)
                    dilate_kernel = np.ones((9, 9), np.uint8)

                    eroded = cv2.erode(thresh, erode_kernel, iterations=1)

                    # plt.imshow(eroded, cmap='gray')
                    # plt.show()

                    dilated = cv2.dilate(eroded, dilate_kernel, iterations=3)

                    # plt.imshow(dilated, cmap='gray')
                    # plt.show()

                    # Find the contours of the fingerprint
                    contours, _ = cv2.findContours(
                        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) is not 0:
                        # Find the contour with the largest area
                        largest_contour = max(contours, key=cv2.contourArea)

                        cv2.drawContours(
                            dilated, [largest_contour], -1, (0, 255, 0), 2)

                        # Get the bounding rectangle of the largest contour (assumed to be the fingerprint)
                        # idx = 0 if len(contours) == 1 else 1
                        x, y, w, h = cv2.boundingRect(largest_contour)

                    # Calculate the center point and major/minor axes of the ellipse
                    center_x = x + int(w / 2)
                    center_y = y + int(h / 2)
                    major_axis = int(w * crop_pram)
                    minor_axis = int(h * crop_pram)

                    # Create a white image the same size as the fingerprint
                    ellipse_mask = np.full(img.shape, mask, dtype=np.uint8)

                    # Draw an ellipse on the white image, we can change the angle of what's left
                    cv2.ellipse(ellipse_mask, (center_x, center_y),
                                (major_axis, minor_axis), 0, 0, 360, (fill_color, fill_color, fill_color), thickness=-1)

                    # plt.imshow(ellipse_mask)
                    # plt.show()

                    # ----For outer----#
                    # inverted_img = cv2.bitwise_not(img)

                    # ----Toggle between or/and + img/inverted for inner/outer----#
                    # Use the ellipse image as a mask to erase the center of the fingerprint image
                    img = cv2.bitwise_or(img, ellipse_mask)

                    # ----For outer----#
                    # img = cv2.bitwise_not(img)

                    # plt.imshow(img)
                    # plt.show()

                    # Save the modified image to a new file
                    cv2.imwrite(os.path.join(
                        output_dir, f'{fig}{crop_str}_{os.path.basename(line.strip())}'), img)
