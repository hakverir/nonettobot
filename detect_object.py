#!/usr/bin/python
from scipy.ndimage import measurements, morphology
from numpy import array, ones
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def is_grey(pixel):
    # print("is_grey 0-1", abs(pixel[0]-pixel[1]))
    # print("is_grey 1-2", abs(pixel[1]-pixel[2]))
    return abs(pixel[0]-pixel[1]) <= 10 and abs(pixel[1]-pixel[2]) <= 10

def is_equal(a, b):
    # print("is_equal 0", (a[0]-b[0])*(a[0]-b[0]))
    # print("is_equal 1", (a[1]-b[1])*(a[1]-b[1]))
    # print("is_equal 2", (a[2]-b[2])*(a[2]-b[2]))
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) < 900

def ball_control():
    # http://en.wikipedia.org/wiki/Mathematical_morphology
    # load image and threshold to make sure it is binary
    img = Image.open("photo.jpg")
    width, height = img.size
    print("w ", width, " h ", height)

    left = width/4
    top = height/4
    right = 3 * width/4
    bottom = 3 * height/4
    # print(left,top,right,bottom)

    img = img.crop((left, top, right, bottom))
    img.show()

    width, height = img.size
    print("w ", width, " h ", height)

    # imgplot = plt.imshow(img)
    # plt.show()

    pixels = list(img.getdata())
    # print("pixels before", pixels)
    # a = input()
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    # print("pixels after", pixels)
    # a = input()
    colours = {(10, 10, 10): 0}
    # print(colours)
    # a = input()

    for i in xrange(height):
        for j in xrange(width):
            if is_grey(pixels[i][j]):
                # print("grey ", i, " ", j, " ", pixels[i][j])
                continue
            if pixels[i][j] not in colours:
                # print("not in colours ", i, " ", j, " ", pixels[i][j])
                colours[pixels[i][j]] = 1
            else:
                # print("else ", i, " ", j, " ", pixels[i][j])
                colours[pixels[i][j]] += 1

    colour = max(colours, key=colours.get)
    # print(colour)

    im = array(Image.open("photo.jpg").convert('L'))
    im = 1*(im < 128)

    labels, nbr_objects = measurements.label(im)

    # morphology - opening to separate objects better
    im_open = morphology.binary_opening(im, ones((9, 5)), iterations=2)

    labels_open, nbr_objects_open = measurements.label(im_open)

    found = False

    # if nbr_objects > 0:
    #     if colour[1] + colour[2] < colour[0] and listener.color == 'red':
    #         found = True
    #     if colour[0] + colour[2] < colour[1] and listener.color == 'green':
    #         found = True
    #     if colour[0] + colour[1] < colour[2] and listener.color == 'blue':
    #         found = True

    # if found:
    x = 0
    cnt = 0
    grey = 0
    for i in xrange(height):
        for j in xrange(width):
            if is_grey(pixels[i][j]):
                grey += 1
                continue
            if is_equal(pixels[i][j], colour):
                x += j
                cnt += 1
    # print((float(x)/cnt)/width)
    # print(cnt)
    # print(float(grey)/(height*width))
    print(cnt)
    return cnt
    # return True, (float(x)/cnt)/width, float(cnt)/(height*width)

    # return False, None, None

def main():
    return ball_control()

if __name__ == '__main__':
    main()
