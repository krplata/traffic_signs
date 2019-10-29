import cv2


class Image_Pyramid:
    '''
    Creates a tuple filled with images scaled down by a factor.
    Used in a sliding window method for detecting shapes (in this case traffic signs).

    Parameters:
        image (cv2_image): Image used for scaling down and creating the pyramid.
        scale_factor (float): Factor by which the dimensions will be resized at each iteration. (Range: (0, 1))
        min_dim (list(int, int)): Cutoff point for generating images. If the dimensions
        don't align with the scale factor, an image smaller than min_dim won't be created.
    '''

    def __init__(self, image, scale_factor, min_dim):
        self.__images__ = (image, )
        if scale_factor <= 0 or scale_factor >= 1:
            return
        while True:
            resized_width = int(self.__images__[-1].shape[1] * scale_factor)
            resized_height = int(self.__images__[-1].shape[0] * scale_factor)
            if resized_height > min_dim[1] and resized_width > min_dim[0]:
                resized = cv2.resize(
                    self.__images__[-1], dsize=(resized_width, resized_height))
                self.__images__ += (resized, )
            else:
                break

    def sliding_window(self, size, step):
        '''
        Runs a 'size' sized window with a 'step' over the image.
        On each iteration, the function yields a window available for further classification.

        Parameters:
            - size (x:int, y:int): Defines the dimensions of the sliding window.
            - step (x:int, y:int): Defines the step sizes along the horizontal and vertical axis.
        '''
        for image in self.__images__:
            for y in range(0, image.shape[0] - size[1], step[1]):
                for x in range(0, image.shape[1] - size[0], step[0]):
                    yield image[y:y+size[1], x:x+size[0]]
