from PIL import Image as im

target_resolution = 1707, 775
background_color = 128, 128, 128
fragment_file_name_template = f'frag_eroded/frag_eroded_{{}}.png'

background: im.Image = im.new('RGBA', target_resolution, color=background_color)

with open('fragments.txt') as file:
    line: str
    for line in file:
        index, x, y, angle = line.split(' ')
        x = int(x)
        y = int(y)
        angle = float(angle)

        fragment: im.Image = im.open(fragment_file_name_template.format(index)).convert('RGBA').rotate(angle)
        background.paste(fragment,
                         (x-fragment.size[0]//2, y-fragment.size[1]//2),
                         )

background.show()
background.save('tp1_1_result.png')
