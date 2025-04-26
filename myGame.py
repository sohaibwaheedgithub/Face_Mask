import cv2
import utils
import ctypes
import numpy as np
import pygame as pg
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

class APP:
    def __init__(self):
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        glClearColor(0.1, 0.2, 0.2, 1.0)
        self.shader = self.createShader('shaders\myVertex.txt', 'shaders\myFragment.txt')
        self.shader2 = self.createShader('shaders\myVertex2.txt', 'shaders\myFragment2.txt')
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        self.square = Square()
        init_array = cv2.imread('textures\cat.jpg')
        self.wood_texture = Material(init_array)
        self.clock = pg.time.Clock()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mainLoop()

    def createShader(self, vertexFilePath, fragmentFilePath):
        with open(vertexFilePath, 'r') as f:
            vertex_src = f.readlines()
        with open(fragmentFilePath, 'r') as f:
            fragment_src = f.readlines()
        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        return shader

    def mainLoop(self):
        cap = cv2.VideoCapture(0)
        running = True
        while running:
            success, frame = cap.read()
            if not success:
                continue
            
            # Processing for mediapipe face mesh
            image = frame.copy()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                ls_single_face=results.multi_face_landmarks[0]
                
                offset_landmarks_list = []
                x_min = ((ls_single_face.landmark[234].x * image.shape[1]) - 320) / 320
                y_min = ((ls_single_face.landmark[10].y * image.shape[0]) - 240) / 240
                x_max = ((ls_single_face.landmark[454].x * image.shape[1]) - 320) / 320
                y_max = ((ls_single_face.landmark[152].y * image.shape[0]) - 240) / 240

                x_range = x_max - x_min
                y_range = y_max - y_min

                for idx in ls_single_face.landmark:
                    x = ((idx.x * image.shape[1]) - 320) / 320
                    y = -1 * (((idx.y * image.shape[0]) - 240) / 240)
                    offset_landmarks_list.append((x, y))
                    
                gl_triangles_vertices = utils.gl_triangulation(offset_landmarks_list, x_min, y_min, x_range, y_range)
            

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                    

            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(self.shader)
            self.wood_texture.update_texture(img_array=frame)
            self.wood_texture.use()
            glBindVertexArray(self.square.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.square.vertex_count)
            
            self.square.update_vertices(gl_triangles_vertices)
            
            glUseProgram(self.shader2)
            glBindVertexArray(self.square.vao2)
            glDrawArrays(GL_TRIANGLES, 0, self.square.vertex_count_2)

            pg.display.flip()
            self.clock.tick(60)

        self.quit()
        cap.release()
        cv2.destroyAllWindows()

    def quit(self):
        self.square.destroy()
        self.wood_texture.destroy()
        glDeleteProgram(self.shader)
        glDeleteProgram(self.shader2)
        pg.quit()

class Square:
    def __init__(self):
        vertices = (
            -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

        self.vertices = np.array(vertices, np.float32)
        self.vertex_count = self.vertices.shape[0] // 8
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))

       
        vertices2 = np.zeros((15354,), dtype=np.float32)
        self.vertices2 = np.array(vertices2, dtype=np.float32)
        self.vertex_count_2 = self.vertices2.shape[0] // 6
        self.vao2 = glGenVertexArrays(1)
        glBindVertexArray(self.vao2)
        self.vbo2 = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo2)
        glBufferData(GL_ARRAY_BUFFER, self.vertices2.nbytes, self.vertices2, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def update_vertices(self, vertices: np.ndarray):
        glBindVertexArray(self.vao2)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo2)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))
        glDeleteVertexArrays(1, (self.vao2,))
        glDeleteBuffers(1, (self.vbo2,))

class Material:
    def __init__(self, img_array: np.ndarray):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        array = img_array.transpose([1, 0, 2])
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        image = pg.surfarray.make_surface(array)
        img_width, img_height = image.get_rect().size
        img_data = pg.image.tostring(image, 'RGBA')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def update_texture(self, img_array: np.ndarray):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        array = img_array.transpose([1, 0, 2])
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        image = pg.surfarray.make_surface(array)
        img_width, img_height = image.get_rect().size
        img_data = pg.image.tostring(image, 'RGBA')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))


class Material1:
    def __init__(self, img_array):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        array = array.transpose([1, 0, 2])
        image = pg.surfarray.make_surface(array)
        img_width, img_height = image.get_rect().size
        img_data = pg.image.tostring(image, 'RGBA')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        pass
    def destroy(self):
        glDeleteTextures(1, (self.texture,))

        

        



if __name__ == '__main__':
    app = APP()