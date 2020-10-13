import cv2
import os
import glob

dataset_path = 'data/datasets/original/'
out_path = 'data/datasets/'

slices_x = 7
slices_y = 5
green = (0,255,0)
red = (0,255,255)
width = 224
height = 224

step_x = width // slices_x
step_y = height // slices_y

class vel:

    def __init__(self, ident):
        self.ident = ident
        self.count = 0

    def __str__(self):
        return f'{self.ident}: {self.count}'

    

dict_v = {4: vel('slow'),
          3: vel('moderate'),
          2: vel('fast'),
          1: vel('very fast'),
          0: vel('backwards')}

dict_w = {0: vel('radically left'),
          1: vel('moderate left'),
          2: vel('slightly left'),
          3: vel('straight'),
          4: vel('slightly right'),
          5: vel('moderate right'),
          6: vel('radically right')}

def calculate_class_w(img, x0):
    class_w = -1

    for n in range(slices_x):
        if x0 > n*step_x:
            class_w = n
        else:
            break
        
    return class_w

def calculate_class_v(img, x1):
    class_v = -1

    for n in range(slices_y):
        if x1 > n*step_y:
            class_v = n
        else:
            break
    
    return class_v

def calculate_classes(img, x_img, y_img):
    class_w = calculate_class_w(img, x_img)
    class_v = calculate_class_v(img, y_img)

    return class_w, class_v

def draw_vertical_lines(img, width, height):
    for n in range(slices_x):
        orig = (n*step_x, 0)
        end = (n*step_x, height)
        cv2.line(img, orig, end, green, 1)


def draw_horizontal_lines(img, width, height):
    for n in range(slices_y):
        orig = (0, n*step_y)
        end = (width, n*step_y)
        cv2.line(img, orig, end, green, 1)


def draw_grid(img, slices_x: int, slices_y: int):
    draw_vertical_lines(img, width, height)
    draw_horizontal_lines(img, width, height)


def draw_orig_dest(img, dest_x, dest_y):
    width, height, _ = img.shape

    # draw robot position (bottom-center)
    cv2.circle(img, (width//2, height), 2, green, 2)
    cv2.circle(img, (width//2, height), 7, green, 1)

    # draw destination point
    cv2.circle(image, (dest_x, dest_y), 2, red, 2)
    cv2.circle(image, (dest_x, dest_y), 7, red, 1)

    cv2.line(image, (width//2, height), (dest_x, dest_y), (255,0,0), 2)


def get_x(path):
    return int(path.split('_')[1])

def get_y(path):
    return int(path.split('_')[2])


def get_id(path):
    """Gets the uuid from the image filename"""
    return path.split('_')[-1].split('.')[0]


def get_image_info(image_name):
    x_img = get_x(image_name)
    y_img = get_y(image_name)
    id_img = get_id(image_name)
    
    return x_img, y_img, id_img

def save_image(out_path, class_w, class_v, id_img):
    out_name = out_path + str(f'vw_{class_v}_{class_w}_{id_img}.jpg')

    cv2.imwrite(out_name, original)


if __name__ == '__main__':

    for filename in glob.glob(dataset_path + 'dataset_izq_segmento/*.jpg'):
        image_name = os.path.basename(filename)
        image = cv2.imread(filename)
        original = image.copy()

        #draw_grid(image, slices_x, slices_y)
        x_img, y_img, id_img = get_image_info(image_name)
        draw_orig_dest(image, x_img, y_img)

        class_w, class_v = calculate_classes(image, x_img, y_img)

        save_image(out_path, class_w, class_v, id_img)

        #print(f'V class: {class_v}\t\t W class: {class_w}')
        cv2.imshow('result', image)
        cv2.waitKey(0)

        dict_w[class_w].count += 1
        dict_v[class_v].count += 1

    for item in dict_v:
        print(dict_v[item])

    print('-----------------')
    
    for item in dict_w:
        print(dict_w[item])

