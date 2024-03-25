import cv2
import numpy as np
import os
import random

def estimate_homography(matches, kp1, kp2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M, mask.ravel().tolist()

main_image_path = 'tp3/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn.jpg'
fragments_dir = 'tp3/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn/frag_eroded'

main_image = cv2.imread(main_image_path, cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
kp1, des1 = orb.detectAndCompute(main_image, None)

height, width = main_image.shape
reconstructed_image = np.zeros((height, width), dtype=np.uint8)

for fragment_file in sorted(os.listdir(fragments_dir)):
    fragment_path = os.path.join(fragments_dir, fragment_file)
    fragment = cv2.imread(fragment_path, cv2.IMREAD_GRAYSCALE)
    kp2, des2 = orb.detectAndCompute(fragment, None)

    if not kp2 or len(kp2) < 4:
        continue

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    if len(matches) >= 4:
        M, mask = estimate_homography(matches, kp1, kp2)
        if M is not None and np.all(np.abs(M) < 1000):
            transformed_fragment = cv2.warpPerspective(fragment, M, (width, height))
            reconstructed_image = cv2.bitwise_or(reconstructed_image, transformed_fragment)
        else:
            print(f"Unreasonable homography for {fragment_file}")

cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
