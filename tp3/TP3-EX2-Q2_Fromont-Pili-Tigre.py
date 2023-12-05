import cv2
import numpy as np
import random

def estimate_transformation(subset):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in subset]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in subset]).reshape(-1, 1, 2)

    transformation, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return transformation

import cv2
import numpy as np
import os

main_image_path = 'tp3/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn.jpg'
main_image = cv2.imread(main_image_path, cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_HARRIS_SCORE, edgeThreshold=1)

fragments_dir = 'tp3/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn/frag_eroded'

fragment_files = sorted(os.listdir(fragments_dir))

kp1, des1 = orb.detectAndCompute(main_image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for fragment_file in fragment_files:
    if fragment_file.endswith('.png'):
        fragment_path = os.path.join(fragments_dir, fragment_file)
        fragment = cv2.imread(fragment_path, cv2.IMREAD_GRAYSCALE)

        kp2 = orb.detect(fragment)

        if len(kp2) < 10:
            print(f"Not enough keypoints found for {fragment_file}")
            continue

        kp2, des2 = orb.compute(fragment, kp2)

        if des2 is None:
            print(f"Descriptors not found for {fragment_file}")
            continue

        if des1 is None or des1.dtype != des2.dtype:
            print(f"Descriptors not compatible for {fragment_file}")
            continue

        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

num_iterations = 1000
inlier_threshold = 5.0
best_consistent_subset = []

for i in range(num_iterations):
    
    subset = random.sample(matches, 3)

    transformation = estimate_transformation(subset)

    consistent_subset = []
    for match in matches:
        src_pt = np.float32([kp1[match.queryIdx].pt]).reshape(-1, 1, 2)
        dst_pt = np.float32([kp2[match.trainIdx].pt]).reshape(-1, 1, 2)

        transformed_pt = cv2.transform(src_pt, transformation)

        if np.linalg.norm(transformed_pt - dst_pt) < inlier_threshold:
            consistent_subset.append(match)

    if len(consistent_subset) > len(best_consistent_subset):
        best_consistent_subset = consistent_subset

final_transformation = estimate_transformation(best_consistent_subset)

print("Nombre de correspondances dans le sous-ensemble cohérent :", len(best_consistent_subset))
print("Index des points correspondants dans le sous-ensemble cohérent :", [match.queryIdx for match in best_consistent_subset])
print("Paramètres de la transformation estimée (x, y, θ) :", final_transformation)

