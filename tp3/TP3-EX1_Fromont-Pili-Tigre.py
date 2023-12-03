import cv2
import numpy as np
import os

main_image_path = 'tp3/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn.jpg'
main_image = cv2.imread(main_image_path, cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_HARRIS_SCORE, edgeThreshold=15)

fragments_dir = 'tp3/Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn/frag_eroded'

fragment_files = sorted(os.listdir(fragments_dir))

kp1, des1 = orb.detectAndCompute(main_image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

associations_file = open('tp3//associations.txt', 'w')

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

        associations_file.write(f"Matches for {fragment_file}:\n")
        for match in matches[:10]:
            associations_file.write(f"Distance: {match.distance}, TrainIdx: {match.trainIdx}, QueryIdx: {match.queryIdx}\n")
        associations_file.write("\n")

associations_file.close()
