from PIL import Image, ImageOps
from PIL.Image import Image as Img

def fitByVisualCenter(image, size, method=Image.Resampling.BICUBIC,visual_center=(0.5, 0.5)):
    """
    Returns a resized and cropped version of the image, cropped to the
    requested aspect ratio and size.

    This function was contributed by Kevin Cazabon.

    :param image: The image to resize and crop.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.
                   See :ref:`concept-filters`.
    :param visual_center: current center (wc,hc) of image
    :return: An image.
    """

    # ensure centering is mutable
    visual_center = list(visual_center)

    live_size = (
        image.size[0],
        image.size[1],
    )

    # calculate the aspect ratio of the live_size
    live_size_ratio = live_size[0] / live_size[1]

    # calculate the aspect ratio of the output image
    output_ratio = size[0] / size[1]

    # figure out if the sides or top/bottom will be cropped off
    if live_size_ratio == output_ratio:
        # live_size is already the needed ratio
        crop_width = live_size[0]
        crop_height = live_size[1]
    elif live_size_ratio >= output_ratio:
        # live_size is wider than what's needed, crop the sides
        crop_width = output_ratio * live_size[1]
        crop_height = live_size[1]
    else:
        # live_size is taller than what's needed, crop the top and bottom
        crop_width = live_size[0]
        crop_height = live_size[0] / output_ratio

    # make the crop
    wc,hc = visual_center[0]*live_size[0],visual_center[1]*live_size[1]

    crop_left = wc - crop_width/2
    crop_right = wc + crop_width/2
    crop_top = hc - crop_height/2
    crop_bottom = hc + crop_height/2

    if crop_left<0:
        crop_left = 0
    if crop_right>live_size[0]:
        crop_left = live_size[0]-crop_width
    if crop_top<0:
        crop_top = 0
    if crop_bottom>live_size[1]:
        crop_top = live_size[1]-crop_height   


    crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)

    # resize the image and return it
    return image.resize(size, method, box=crop)
rawImage = Image.open('xxxx.jpg')
image = fitByVisualCenter(
    rawImage,
    (200, 250),
    visual_center=(0.5, 0.5),
    method=Image.Resampling.BICUBIC
).convert(mode='RGB')
image.save('a.jpg')
