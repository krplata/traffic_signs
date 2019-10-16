import cv2


def resize_smaller_side(image_data, dest_size, interpol=cv2.INTER_NEAREST):
    '''
    Returns an image resized along the smaller dimension.

    Parameters:
        image_data (cv2_image): Image to be resized.
        dest_size (int): Destination size for the resized dimension.
        interpol (cv2.interpolation_methods): Interpolation method used during resizing. 
    '''
    im_y, im_x = image_data.shape[:2]
    smaller_side = (im_y, dest_size) if im_x < im_y else (
        dest_size, im_x)
    return cv2.resize(
        image_data, smaller_side, interpolation=interpol)


def im_upsample_to(image_data, dest_size):
    '''
    Returns an image upsampled by dest_size/smaller_edge.
    This way, the aspect ratio of an image is preserved.

    Parameters:
        image_data (cv2_image): Image to be resized.
        dest_size (int): Upsampling target size.
    '''
    im_y, im_x = image_data.shape[:2]
    smaller_edge = min(im_x, im_y)
    resize_factor = dest_size/smaller_edge
    return cv2.resize(image_data, None, fx=resize_factor, fy=resize_factor,
                      interpolation=cv2.INTER_LINEAR)


def im_fit_to_shape(image_data, dest_size, interpol=cv2.INTER_NEAREST):
    '''
    Fits an image into target size.
    Needs a little love, this thing is nasty.
    '''
    im_y, im_x = image_data.shape[:2]
    if im_x == im_y:
        return cv2.resize(
            image_data, (dest_size, dest_size), interpolation=interpol)
    else:
        resized = []
        if im_x < dest_size or im_y < dest_size:
            resized = im_upsample_to(image_data, dest_size)
        else:
            resized = resize_smaller_side(image_data, dest_size)
        resized_y, resized_x = resized.shape[:2]
        left = int(resized_x/2 - dest_size/2)
        bottom = int(resized_y/2 - dest_size/2)
        return resized[bottom: bottom + dest_size, left: left + dest_size]


def im_prepare(image_data):
    return im_fit_to_shape(image_data, 31)
