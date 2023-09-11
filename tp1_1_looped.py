import cv2 as cv
import numpy as np

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


background = np.zeros((*target_resolution[::-1], 4), dtype=np.uint8)
background += np.array([*background_color, 0], dtype=np.uint8)

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
        x_bot_right_corner_pos = x_center_pos + fragment_size_x // 2 + fragment_size_x % 2
        y_top_left_corner_pos = y_center_pos - fragment_size_y // 2
        y_bot_right_corner_pos = y_center_pos + fragment_size_y // 2 + fragment_size_y % 2

        # Paste fragment according to the mask in the alpha channel
        try:
            for i in range (x_top_left_corner_pos, x_bot_right_corner_pos):
                for j in range (y_top_left_corner_pos, y_bot_right_corner_pos):
                    if fragment[j-y_top_left_corner_pos, i-x_top_left_corner_pos, 3] > 0:
                        background[j, i, :] = fragment[j-y_top_left_corner_pos, i-x_top_left_corner_pos, :]
        except Exception as e:
            print(index + " " + str(e))

cv.imshow(" ", background)
cv.waitKey(0)
cv.destroyAllWindows()
