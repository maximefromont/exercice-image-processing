import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
background[:, :, :3] = background_color

def process_fragment(fragment_info):
    index, x_center_pos, y_center_pos, angle = fragment_info
    x_center_pos = int(x_center_pos)
    y_center_pos = int(y_center_pos)
    angle = float(angle)

    fragment = cv.imread(fragments_file_name_template.format(index), flags=cv.IMREAD_UNCHANGED)
    fragment = rotate_image(fragment, angle)

    fragment_size_y, fragment_size_x = fragment.shape[0], fragment.shape[1]
    x_top_left_corner_pos = x_center_pos - fragment_size_x // 2
    x_bot_right_corner_pos = x_center_pos + fragment_size_x // 2 + fragment_size_x % 2
    y_top_left_corner_pos = y_center_pos - fragment_size_y // 2
    y_bot_right_corner_pos = y_center_pos + fragment_size_y // 2 + fragment_size_y % 2

    # Clip coordinates to stay within background bounds
    x_top_left_corner_pos = max(0, x_top_left_corner_pos)
    x_bot_right_corner_pos = min(target_resolution[0], x_bot_right_corner_pos)
    y_top_left_corner_pos = max(0, y_top_left_corner_pos)
    y_bot_right_corner_pos = min(target_resolution[1], y_bot_right_corner_pos)

    fragment_x_start = max(0, -x_top_left_corner_pos)
    fragment_x_end = min(fragment_size_x, x_bot_right_corner_pos - x_top_left_corner_pos)
    fragment_y_start = max(0, -y_top_left_corner_pos)
    fragment_y_end = min(fragment_size_y, y_bot_right_corner_pos - y_top_left_corner_pos)

    background_x_start = x_top_left_corner_pos
    background_x_end = x_top_left_corner_pos + fragment_x_end - fragment_x_start
    background_y_start = y_top_left_corner_pos
    background_y_end = y_top_left_corner_pos + fragment_y_end - fragment_y_start

    mask = fragment[fragment_y_start:fragment_y_end, fragment_x_start:fragment_x_end, 3] > 0
    background_mask = background[background_y_start:background_y_end, background_x_start:background_x_end, 3] > 0

    background[background_y_start:background_y_end, background_x_start:background_x_end, :3][mask] = fragment[
        fragment_y_start:fragment_y_end, fragment_x_start:fragment_x_end, :3][mask]
    background[background_y_start:background_y_end, background_x_start:background_x_end, 3][mask] = 255

with open(fragments_text_file) as file:
    fragment_infos = [line.split() for line in file]

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    executor.map(process_fragment, fragment_infos)

cv.imshow("Result", background)
cv.waitKey(0)
cv.destroyAllWindows()
