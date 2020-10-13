import cv2
import os
import glob

mouseX, mouseY = 0, 0

def paint_circle(image, x, y):

    destination = (x, y)
    bottom = (int(image.shape[0] / 2), image.shape[1])

    cv2.circle(image, destination, 20, (0,255,0), 2)
    cv2.circle(image, destination, 4, (0,255,0), -1)
    cv2.circle(image, bottom, 5, (255,255,255), -1)
    cv2.line(image, bottom, destination, (0,255,255), 1)
    return image

def get_x(path):
    """Gets the x value from the image filename"""
    return int(path[3:6])

def get_y(path):
    """Gets the y value from the image filename"""
    return int(path[7:10])

def get_id(path):
    """Gets the uuid from the image filename"""
    return path.split('_')[-1].split('.')[0]

def get_clone():
    global image
    del image
    return orig.copy()

def on_click(event, x, y, p1, p2):
    global mouseX, mouseY, image
    if event == cv2.EVENT_LBUTTONDOWN:
        image = get_clone()
        paint_circle(image, x, y)
        draw_grid(image, slices_y, slices_y)
        mouseX, mouseY = x, y

def save_new_image(x, y, id_img):
    name = str(f'xy_{x}_{y}_{id_img}.jpg')
    print('result:', name)
    cv2.imwrite(out_path + name, orig)

def get_image_info(image_name):
    x_img = get_x(image_name)
    y_img = get_y(image_name)
    id_img = get_id(image_name)

    return x_img, y_img, id_img

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


slices_x = 7
slices_y = 5
green = (0, 255, 0)
red = (0, 255, 255)
gray = (51, 51, 51)
width = 224
height = 224

step_x = width // slices_x
step_y = height // slices_y

def draw_vertical_lines(img, width, height):
    for n in range(slices_x):
        orig = (n * step_x, 0)
        end = (n * step_x, height)
        cv2.line(img, orig, end, gray, 1)


def draw_horizontal_lines(img, width, height):
    for n in range(slices_y):
        orig = (0, n * step_y)
        end = (width, n * step_y)
        cv2.line(img, orig, end, gray, 1)


def draw_grid(img, slices_x: int, slices_y: int):
    draw_vertical_lines(img, width, height)
    draw_horizontal_lines(img, width, height)

if __name__ == '__main__':

    image_path = "data/datasets/original/"

    for idx, file_ in enumerate(glob.glob(image_path + '**/*.jpg')):
        out_path = 'data/datasets/modified/' + file_.split('\\')[1] + '/'
        print('--------------------------')
        print('count:', idx)

        image_name = os.path.basename(file_)
        print('original:', image_name)
        create_folder(out_path)
        image = cv2.imread(file_)
        orig = image.copy()

        x_img, y_img, id_img = get_image_info(image_name)
        mouseX = x_img
        mouseY = y_img

        image = paint_circle(image, x_img, y_img)
        draw_grid(image, slices_x, slices_y)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_click)

        while(1):
            cv2.imshow('image', image)
            k = cv2.waitKey(20) & 0xFF
            if k == 13:     # enter
                save_new_image(mouseX, mouseY, id_img)
                break
            elif k == 27:     # esc
                break
            elif k == ord('a'):
                print(mouseX, mouseY)

        cv2.destroyAllWindows()
    

