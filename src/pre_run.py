import cv2

# Functions used for preparing images to be used as an input to CNN.
# Image of an arbitrary size -> downsampling -> histogram equalization -> CNN


def resize_smaller_side(image_data, dest_size, interpol=cv2.INTER_NEAREST):
    im_y, im_x = image_data.shape[:2]
    smaller_side = (im_y, dest_size) if im_x < im_y else (
        dest_size, im_x)
    return cv2.resize(
        image_data, smaller_side, interpolation=interpol)


def im_downsample(image_data, dest_size, interpol=cv2.INTER_NEAREST):
    im_x, im_y = image_data.shape[:2]
    if im_x == im_y:
        return cv2.resize(
            image_data, (dest_size, dest_size), interpolation=interpol)
    else:
        shrinked = resize_smaller_side(image_data, dest_size)
        shrinked_y, shrinked_x = shrinked.shape[:2]
        left = int(shrinked_x/2 - dest_size/2)
        bottom = int(shrinked_y/2 - dest_size/2)
        return shrinked[bottom: bottom + dest_size, left: left + dest_size]


def im_prepare(image_data):
    downsampled = im_downsample(image_data, 31)
    return cv2.equalizeHist(downsampled)


# image = cv2.imread('../images/00000/00000_00013.ppm', cv2.IMREAD_COLOR)
# cv2.imwrite('./output.png', im_downsample(image, 31))
