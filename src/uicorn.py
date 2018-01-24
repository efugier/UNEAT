# Simple game which is to be solved by the NEAT algorithm

import tkinter as tk
from random import random

# PARAMETERS

CANH, CANW = 350, 1000         # canvas dimensions in px

MAX_NB_JUMPS = 3               # max number of consecutive jumps

dt = 50                        # time step in ms


# CLASSES

class Box:
    """A simple box object which can collide with others"""

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
                or box.y + box.h <= self.y
                or self.x + self.w <= box.x
                or self.y + self.h <= box.y):
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


class World:
    """The world in which the unicron runs"""

    def __init__(self, tk_root, tk_can, tk_unicorn):
        self.root = tk_root
        self.can = tk_can

        self.unicorn = tk_unicorn
        self.obstacle_list = []
        self.speed = 0.2
        self.score = 0

    def createObstacle(self):
        """Adds a new obstable arbitrarly generated"""
        if self.obstacle_list:
            x_max = max(CANW, max([obs.x for obs in self.obstacle_list]))
        else:
            x_max = CANW
        x = x_max + 100 + int(random() * CANW)
        h = max(10, int(random() * 40))
        y = CANH - h
        w = max(10, random() * 30)
        return DrawableBox(x, y, w, h, self.can)

    def reset(self, event=None):
        self.score = 0

        # Reseting the unicorn
        self.unicorn.x = 100
        self.unicorn.y = CANH - self.unicorn.h
        self.unicorn.dx = 0
        self.unicorn.dy = 0
        self.unicorn.alive = True

        # Reseting the obstacles
        self.obstacle_list = []
        for _ in range(3):
            self.obstacle_list.append(self.createObstacle())

        self.gameEngine()

    def update(self):
        self.unicorn.update()

        for i, obs in enumerate(self.obstacle_list):
            obs.dx = -self.speed
            obs.update()
            if self.unicorn.collide(obs):
                self.unicorn.alive = False
            if obs.x + obs.w < 0:  # If the obstacle goes out of the screen
                self.obstacle_list[i] = self.createObstacle()

    def gameEngine(self):
        self.speed *= 1.0001
        self.update()
        if self.unicorn.alive:
            self.root.after(dt, self.gameEngine)


# FUNCTIONS


root = tk.Tk()
root.title("Jumping Unicorn")
can = tk.Canvas(root, bg='purple', height=CANH, width=CANW)
can.pack()

photo = tk.PhotoImage(file="../img/unicorn.png")
uni_drawing = can.create_image(100, CANH - 50, anchor=tk.NW, image=photo)
unicorn = Unicorn(100, CANH - 50, 50, 50, can, uni_drawing)

world = World(root, can, unicorn)
world.reset()

root.mainloop()
