# Simple game which is to be solved by the NEAT algorithm

import tkinter as tk
from random import random, randint

# PARAMETERS

CANH, CANW = 350, 1000         # canvas dimensions in px

MAX_NB_JUMPS = 3               # max number of consecutive jumps

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

        # Physics
        self.dx = 0
        self.dy = 0

    def set_dx(self, dx):
        self.dx = dx

    def set_dy(self, dy):
        self.dy = dy

    def update(self):
        """Updates the position"""
        self.x += self.dx * dt
        self.y += self.dy * dt

    def collide(self, box):
        if (box.x + box.w <= self.x  # the other box is left, over, right, under
                or box.y <= self.y + self.h
                or self.x + self.w <= box.x
                or self.y <= box.y + box.h):
            return False
        return True


class DrawableBox(Box):
    """A bow that can be drawn on a canvas"""

    def __init__(self, x, y, w, h, can, obj=None, color='violet'):
        Box.__init__(self, x, y, w, h,)
        self.can = can
        self.alive = True
        if obj:
            self.obj = obj
            self.is_image = True
        else:
            self.obj = can.create_rectangle(x, y, self.x2, self.y2, fill=color)
            self.is_image = False

    def update(self):
        """Updates the position and the drawing's position"""
        Box.update(self)
        if self.is_image:
            self.can.coords(self.obj, self.x, self.y)
        else:
            self.can.coords(self.obj, self.x, self.y,
                            self.x + self.w, self.y + self.h)

    def erase(self):
        self.can.delete(self)


class Unicorn(DrawableBox):
    def __init__(self, x, y, w, h, can, obj=None, color='violet'):
        DrawableBox.__init__(self, x, y, w, h, can, obj, color='violet')
        self.nb_jumps = 0
        self.update()

    def jump(self):
        if self.nb_jumps < MAX_NB_JUMPS and self.dy >= 0:
            self.nb_jumps += 1
            self.dy -= 0.5

    def update(self):
        DrawableBox.update(self)
        if self.y + self.h > CANH:  # Prevents the unicorn from going underground
            self.dy = 0
            self.y = 0


# FUNCTIONS

def randomObstacle():
    """Generates a random obstacle"""
    pass


root = tk.Tk()
root.title("Jumping Unicorn")
can = tk.Canvas(root, bg='purple', height=CANH, width=CANW)
can.pack()

photo = tk.PhotoImage(file="../img/unicorn.png")
uni_drawing = can.create_image(100, CANH - 50, anchor=tk.NW, image=photo)
uni = Unicorn(100, CANH - 50, 50, 50, can, uni_drawing)

root.mainloop()
