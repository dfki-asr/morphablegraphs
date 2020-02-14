#!/usr/bin/env python
#
# Copyright 2019 DFKI GmbH.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

class HeightMapInterface(object):
    def __init__(self, image, width, depth, scale, height_scale, pixel_is_tuple=False):
        self.height_map_image = image
        self.scale = scale
        self.height_scale = height_scale
        self.width = width
        self.depth = depth
        self.x_offset = 0
        self.z_offset = 0
        self.is_tuple = pixel_is_tuple

    def to_relative_coordinates(self, center_x, center_z, x, z):
        """ get position relative to upper left
        """
        relative_x = x - center_x
        relative_z = z - center_z
        relative_x /= self.scale[0]
        relative_z /= self.scale[1]
        relative_x += self.width / 2
        relative_z += self.depth / 2

        # scale by width and depth to range of 1
        relative_x /= self.width
        relative_z /= self.depth
        return relative_x, relative_z

    def get_height_from_relative_coordinates(self, relative_x, relative_z):
        if relative_x < 0 or relative_x > 1.0 or relative_z < 0 or relative_z > 1.0:
            print("Coordinates outside of the range")
            return 0
        # scale by image width and height to image range
        ix = relative_x * self.height_map_image.size[0]
        iy = relative_z * self.height_map_image.size[1]
        p = self.height_map_image.getpixel((ix, iy))
        if self.is_tuple:
            p = p[0]
        return (p / 255) * self.height_scale

    def get_height(self, x, z):
        rel_x, rel_z = self.to_relative_coordinates(self.x_offset, self.z_offset, x, z)
        y = self.get_height_from_relative_coordinates(rel_x, rel_z)
        #print("get height", x, z,":", y)
        return y
