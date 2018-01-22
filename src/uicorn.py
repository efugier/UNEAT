# Simple game which is to be solved by the NEAT algorithm

import tkinter as tk
from random import random, randint

# PARAMETERS

canh, canw = 350, 1000         # canvas dimensions in px

max_nb_jumps = 3               # max number of consecutive jumps
dt = 50                        # time step in ms


# CLASSES

class Box:
    """A simple box object with can collide with others"""

    def __init__(self, x, y, w, h):
        """(x,y) is the top-left corner of the box"""
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = x + w
        self.y2 = y + h

    def collide(self, box):
        """pos relative = self.pos - pos"""
        if (box.x + box.w <= self.x  # the other box is left, over, right, under
                or box.y + box.h <= self.y
                or self.x + self.w <= box.x
                or self.y + self.h <= box.y):
            return False
        return True

    def reposition(self, newx, newy):
        self.x, self.y = newx, newy
        self.x2, self.y2 = newx + self.w, newy + self.h


class DrawableBox(Box):
    """A bow that can be drawn on a canvas"""

    def __init__(self, x, y, w, h, can, obj=None, color='violet'):
        Box.__init__(self, x, y, w, h,)
        self.can = can
        if obj:
            self.obj = obj
        else:
            self.obj = can.create_rectangle(x, y, self.x2, self.y2, fill=color)

    def reposition(self, newx, newy):
        Box.reposition(self, newx, newy)
        self.can.coords(self.obj, newx, newy, newx + self.w, newy + self.h)

    def erase(self):
        self.can.delete(self)


class Obsatcle(DrawableBox):
    """An obstacle which the unicorn lust avoid"""

    def __init__(self, x, y, w, h, can, obj=None, color='violet'):
        DrawableBox.__init__(self, x, y, w, h, can, None, color='violet')

    def update(self, v):
        """updates the box going left at a speed v"""
        self.reposition(self.x - v * dt, self.y)
        if self.x < -self.w:  # removes the box if it is out of the screen
            self.erase()
