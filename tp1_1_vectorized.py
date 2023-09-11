import cv2 as cv
import numpy as np
import time as time

time_start = time.time()

target_resolution = (1707, 775)
background_color = (128, 128, 128)
fragments_file_name_template = 'frag_eroded/frag_eroded_{}.png'
fragments_text_file = "fragments.txt"

def rotate_image(image, theta):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, theta, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

background = np.zeros((*target_resolution[::-1], 4), dtype=np.uint8)
background[:, :, :3] += np.array([*background_color], dtype=np.uint8)

fragment_counter = 0

with open(fragments_text_file) as file:
    for line in file:
        # Get fragment info
        index, x_center_pos, y_center_pos, angle = map(float, line.split())
        x_center_pos, y_center_pos = int(x_center_pos), int(y_center_pos)

        # Read and rotate
        fragment = cv.imread(fragments_file_name_template.format(int(index)), flags=cv.IMREAD_UNCHANGED)
        fragment = rotate_image(fragment, angle)

        # Compute area to override
        fragment_size_y, fragment_size_x = fragment.shape[0], fragment.shape[1]
        x_top_left_corner_pos = x_center_pos - fragment_size_x // 2
        x_bot_right_corner_pos = x_center_pos + fragment_size_x // 2
        y_top_left_corner_pos = y_center_pos - fragment_size_y // 2
        y_bot_right_corner_pos = y_center_pos + fragment_size_y // 2

        # Ensure coordinates are within bounds
        if (0 <= x_top_left_corner_pos < target_resolution[0] and
            0 <= x_bot_right_corner_pos < target_resolution[0] and
            0 <= y_top_left_corner_pos < target_resolution[1] and
            0 <= y_bot_right_corner_pos < target_resolution[1]):

            # Paste fragment with transparency
            x1, x2 = max(x_top_left_corner_pos, 0), min(x_bot_right_corner_pos, target_resolution[0])
            y1, y2 = max(y_top_left_corner_pos, 0), min(y_bot_right_corner_pos, target_resolution[1])

            fragment_x1 = x1 - x_top_left_corner_pos
            fragment_x2 = fragment_x1 + (x2 - x1)
            fragment_y1 = y1 - y_top_left_corner_pos
            fragment_y2 = fragment_y1 + (y2 - y1)

            # Blend pixels manually with transparency
            fragment_alpha = fragment[fragment_y1:fragment_y2, fragment_x1:fragment_x2, 3]
            background_alpha = background[y1:y2, x1:x2, 3]

            for c in range(3):  # RGB channels
                background[y1:y2, x1:x2, c] = (
                    (1 - fragment_alpha / 255.0) * background[y1:y2, x1:x2, c] +
                    (fragment_alpha / 255.0) * fragment[fragment_y1:fragment_y2, fragment_x1:fragment_x2, c]
                )
            background[y1:y2, x1:x2, 3] = np.maximum(fragment_alpha, background_alpha)

        print(fragment_counter)
        fragment_counter += 1

time_end = time.time()
print("Time elapsed: ", time_end - time_start)

cv.imshow("Result", background[:, :, :3])
cv.waitKey(0)
cv.destroyAllWindows()
