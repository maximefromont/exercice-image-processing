import cv2 as cv
import numpy as np

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

target_resolution = 776, 1708
background_color = 128, 128, 128
fragment_file_name_template = f'frag_eroded/frag_eroded_{{}}.png'

background_image = np.zeros((*target_resolution, 4), dtype=np.uint8)

with open('fragments.txt') as file:
    
    line: str

    counter = 0

    for line in file:

        index, x, y, angle = line.split(' ')
        x = int(x)
        y = int(y)
        angle = float(angle)

        fragment_image_name = fragment_file_name_template.format(index)
        fragment_image = cv.imread(fragment_image_name, cv.IMREAD_UNCHANGED)
        fragment_image = rotate_image(fragment_image, angle)

        size_x_fragment = fragment_image.shape[1]
        size_y_fragment = fragment_image.shape[0]

        size_x_backgroundimage = background_image.shape[1]
        size_y_backgroundimage = background_image.shape[0]

        x_offset = x - size_x_fragment//2
        y_offset = y - size_y_fragment//2

        target_x_offset = x_offset + size_x_fragment
        target_y_offset = y_offset + size_y_fragment

        for i in range (x_offset, target_x_offset):
            for j in range (y_offset, target_y_offset):
                if fragment_image[j-y_offset, i-x_offset, 3] > 0:
                    background_image[j, i, :] = fragment_image[j-y_offset, i-x_offset, :]

        print(counter)
        counter += 1

        #background_image[x_offset : target_x_offset, y_offset : target_y_offset, :] = fragment_image

        

cv.imshow(" ", background_image)
cv.waitKey(0)
cv.destroyAllWindows()