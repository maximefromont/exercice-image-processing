import cv2 as cv
import numpy as np

D_X, D_Y, D_A = 1, 1, 1

target_resolution: tuple[int] = 1707, 775
fragments_file_name_template: str = 'frag_eroded/frag_eroded_{}.png'
fragments_text_file: str = "fragments.txt"
solution_text_file: str = "solution.txt"


def compute_surface(image):
    """
    Compute the surface of a given image as the  number of non-transparent pixels.
    """
    return np.sum(image[:, :, 3] > 0)


correct_fragment_surface: int = 0  # Surface of well-placed fragments in the proposed solution
incorrect_fragment_surface: int = 0  # Surface of wrongly placed fragments int the proposed solution
total_surface: int = 0

incorrect_fragment_indexes: list[int] = open('fragments_s.txt').read().strip().split('\n')

with open(solution_text_file) as solution_file:
    line: str
    for line in solution_file:
        if line.strip() == "":  # Skip empty lines
            continue

        # Get fragment info
        index, x, y, angle = line.split(' ')
        x = int(x)
        y = int(y)
        angle = float(angle)
        fragment = cv.imread(fragments_file_name_template.format(index),
                             flags=cv.IMREAD_UNCHANGED)
        surface = compute_surface(fragment)

        # Check if an incorrect fragment is used
        if index in incorrect_fragment_indexes:
            incorrect_fragment_surface += surface
            continue

        with open(fragments_text_file) as fragment_file:
            f_line: str
            for f_line in fragment_file:
                f_index, f_x, f_y, f_angle = f_line.split()
                if f_index != index:
                    continue
                f_x = int(f_x)
                f_y = int(f_y)
                f_angle = float(f_angle)

                # Check if a correct fragment is well-placed
                if abs(x - f_x) <= D_X and abs(y - f_y) <= D_Y and abs(angle - f_angle) <= D_A:
                    correct_fragment_surface += surface
                else:
                    incorrect_fragment_surface += surface

# Compute the total correct area
with open(fragments_text_file) as file:
    line: str
    for line in file:
        index, *_ = line.split()
        fragment = cv.imread(fragments_file_name_template.format(index),
                             flags=cv.IMREAD_UNCHANGED)
        total_surface += compute_surface(fragment)

precision = (correct_fragment_surface - incorrect_fragment_surface) / total_surface
print(f"Precision: {precision}")
