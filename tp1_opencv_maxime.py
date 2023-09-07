import cv2 as cv
import numpy as np

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

target_resolution = 1707, 775
background_color = 128, 128, 128
fragment_file_name_template = f'frag_eroded/frag_eroded_{{}}.png'

background_image = np.zeros((*target_resolution, 4), dtype=np.uint8)

with open('fragments.txt') as file:
    line: str
    for line in file:

        index, x, y, angle = line.split(' ')
        x = int(x)
        y = int(y)
        angle = float(angle)

        fragment_image_name = fragment_file_name_template.format(index)
        fragment_image = cv.imread(fragment_image_name, cv.IMREAD_UNCHANGED)
        fragment_image = rotate_image(fragment_image, angle)

        size_x = fragment_image.shape[1]
        size_y = fragment_image.shape[0]
        x_offset = x - size_x//2
        y_offset = y - size_y//2

        print(fragment_image.shape)

        background_image[x_offset: x_offset + size_x, y_offset : y_offset + size_y,:] = fragment_image
        break #TO remove

        

cv.imshow('image', background_image)
cv.waitkey(0)