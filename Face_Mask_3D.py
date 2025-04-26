import pygame as pg
from OpenGL.GL import *


class APP:
    def __init__(self):
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        