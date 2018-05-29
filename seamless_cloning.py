#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import cv2

def seamless_cloning( obj,im, mask, mode = 'mixed_clone'):
    """Seamless cloning to make obj looked real in im backgound
    
    Arguments:
        obj {image} -- object - prossed image from model
        im {image} -- backgound image - noise image        
        mask {image} -- mask of the obj
    
    Keyword Arguments:
        mode {str} -- mode of seamless cloning (default: {'mixed_clone'})
        - mixed_clone
        - normal_clone
        - monochrome_transfer
    Returns:
        darray -- processed image after cloning
    """
    # pre-processing the mask
    # convert to grayscale image
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # thresh = 127
    # im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]
    # https://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white
    # convert to only 0 and 255
    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    # The location of the center of the src in the dst
    width, height, channels = im.shape
    # caculate the center point of the obj image
    minx = 1e5
    maxx = 1
    miny = 1e5
    maxy = 1

    for y in range(1, height):
        for x in range(1, width):
            # check wh
            # if ((im_bw[x][y] != 0) and (im_bw[x][y] != 255)):
            #     print(x, y , " : ",im_bw[x][y])
            if im_bw[x][y] > 0:
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)

    center = (int(miny+(maxy-miny)/2), int(minx+(maxx-minx)/2))

    # center = (int(height / 2), int(width / 2))
    print("center", center)
    
    dest_cloned = obj

    # Seamlessly clone src into dst and put the results in output
    if mode == 'normal_clone':
        dest_cloned = cv2.seamlessClone(obj, im, im_bw, center, cv2.NORMAL_CLONE)
    elif mode == 'mixed_clone':
        dest_cloned = cv2.seamlessClone(obj, im, im_bw, center, cv2.MIXED_CLONE)
    else: # monochrome_transfer mode
        dest_cloned= cv2.seamlessClone(obj, im, im_bw, center, cv2.MONOCHROME_TRANSFER)

    # cv2.imshow("obj", obj)
    # cv2.imshow("im", im)    
    # cv2.imshow("mask", mask)
    # cv2.imshow("dest_cloned", dest_cloned)

    # cv2.imshow("dest_cloned", dest_cloned)
    # cv2.waitKey(0)
    return dest_cloned


if __name__ == '__main__':
    print("seamless_cloning")
