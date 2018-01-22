# Simple game which is to be solved by the NEAT algorithm

from tkinter import *
from random import random, randint


class Box:
    """A simple box object
       Box can collide with each others"""

    def __init__(self, can, x, y, w, h, color='violet'):
        self.can = can
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = x + w
        self.y2 = y + h
        self.obj = can.create_rectangle(x, y, self.x2, self.y2, fill=color)

    def redraw(self, newx, newy):
        self.can.coords(self.obj, newx, newy, newx + self.w, newy + self.h)
        if newx < -self.w - 10:
            self.w = randint(10, 50)
            self.h = randint(10, 40)
            self.x += self.can.winfo_width() * (1 + random())
            self.y = self.can.winfo_height() - self.h

    def collide(self):
        """pos relative = self.pos - pos"""
        global x, y, w, h  # uni est gauche, dessus, droite, dessous
        if 100 + w <= self.x - x or y + h <= self.y or self.x + self.w - x <= 100 or self.y + self.h <= y:
            return False
        return True


class DrawableBox(Box):
    """A bow that can be drawn on a canva"""
