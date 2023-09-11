import cv2 as cv
import numpy as np
import time as time

time_start = time.time()

target_resolution: tuple[int] = 1707, 775
background_color: tuple[int] = 128, 128, 128
fragments_file_name_template: str = f'frag_eroded/frag_eroded_{{}}.png'
fragments_text_file: str = "fragments.txt"


def rotate_image(image, theta):
    # https://stackoverflow.com/a/9042907
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, theta, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


background = cv.imread("Michelangelo_ThecreationofAdam_1707x775.jpg", flags=cv.IMREAD_UNCHANGED)
#Add an alha canal, then make the background semi-transparent
background = cv.cvtColor(background, cv.COLOR_RGB2RGBA)
background[:, :, 0:3] //= 2

fragment_counter = 0

with open(fragments_text_file) as file:
    line: str
    for line in file:
        # Get fragment info
        index, x_center_pos, y_center_pos, angle = line.split(' ')
        x_center_pos = int(x_center_pos)
        y_center_pos = int(y_center_pos)
        angle = float(angle)

        # Read and rotate
        fragment = cv.imread(fragments_file_name_template.format(index),
                             flags=cv.IMREAD_UNCHANGED)
        fragment = rotate_image(fragment, angle)

        # Compute area to override
        fragment_size_y, fragment_size_x = fragment.shape[0], fragment.shape[1]
        x_top_left_corner_pos = x_center_pos - fragment_size_x // 2
        x_bot_right_corner_pos = x_center_pos + fragment_size_x // 2
        y_top_left_corner_pos = y_center_pos - fragment_size_y // 2
        y_bot_right_corner_pos = y_center_pos + fragment_size_y // 2

        # Paste fragment according to the mask in the alpha channel
        for i in range (x_top_left_corner_pos, x_bot_right_corner_pos):
            for j in range (y_top_left_corner_pos, y_bot_right_corner_pos):
                if fragment[j-y_top_left_corner_pos, i-x_top_left_corner_pos, 3] > 0:
                    if(i > 0 and i < target_resolution[0] and j > 0 and j < target_resolution[1]):
                        background[j, i, :] = fragment[j-y_top_left_corner_pos, i-x_top_left_corner_pos, :]

        print(fragment_counter)
        fragment_counter += 1

time_end = time.time()
print("Time elapsed: ", time_end - time_start)

cv.imshow(" ", background)
cv.waitKey(0)
cv.destroyAllWindows()
