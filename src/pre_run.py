import cv2

# Functions used for preparing images to be used as an input to CNN.
# Image of an arbitrary size -> downsampling -> histogram equalization -> CNN


def resize_smaller_side(image_data, dest_size, interpol=cv2.INTER_NEAREST):
    im_y, im_x = image_data.shape[:2]
    smaller_side = (im_y, dest_size) if im_x < im_y else (
        dest_size, im_x)
    return cv2.resize(
        image_data, smaller_side, interpolation=interpol)


def im_upsample_to(image_data, dest_size):
    im_y, im_x = image_data.shape[:2]
    smaller_edge = min(im_x, im_y)
    resize_factor = dest_size/smaller_edge
    return cv2.resize(image_data, None, fx=resize_factor, fy=resize_factor,
                      interpolation=cv2.INTER_LINEAR)


def im_fit_to_shape(image_data, dest_size, interpol=cv2.INTER_NEAREST):
    im_y, im_x = image_data.shape[:2]
    if im_x == im_y:
        return cv2.resize(
            image_data, (dest_size, dest_size), interpolation=interpol)
    else:
        resized = []
        if im_x < 31 or im_y < 31:
            resized = im_upsample_to(image_data, 31)
        else:
            resized = resize_smaller_side(image_data, dest_size)
        resized_y, resized_x = resized.shape[:2]
        left = int(resized_x/2 - dest_size/2)
        bottom = int(resized_y/2 - dest_size/2)
        return resized[bottom: bottom + dest_size, left: left + dest_size]


def im_prepare(image_data):
    return im_fit_to_shape(image_data, 31)


# image = cv2.imread(
#     '/home/desktop/Desktop/git/traffic_signs/images/00000/00000_00056.jpg', cv2.IMREAD_COLOR)

# cv2.imwrite('./output.png', im_fit_to_shape(image, 31))
