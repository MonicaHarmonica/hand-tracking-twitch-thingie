import numpy as np

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle


def get_distance(landmark_list):

    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

def get_midpoint(landmark_list):

    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    xres = int((x1 + x2) / 2)
    yres = int((y1 + y2) / 2)

    return (xres, yres)

def point_in_rectangle(point, orig, size):
    (x, y) = point
    (x0, y0) = orig
    (w, h) = size
    return (x > x0 and x < x0 + w) and (y > y0 and y < y0 + h)