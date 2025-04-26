import cv2
import numpy as np
from face_triangle_vertices import trianlge_vertices


def custom_face_mask(image, triangle_coordinates, landmarks_list):
    for tri_cords in triangle_coordinates:
        cv2.line(image, tri_cords[0], tri_cords[1], color=(0, 255, 0), thickness=1)
        cv2.line(image, tri_cords[1], tri_cords[2], color=(0, 255, 0), thickness=1)
        cv2.line(image, tri_cords[2], tri_cords[0], color=(0, 255, 0), thickness=1)
    for cord in landmarks_list:
        cv2.putText(image, '.', cord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    return image



def gl_triangulation(landmarks_list, x_min, y_min, x_range, y_range):
    landmarks_array = np.array(landmarks_list)
    trianlge_indices = trianlge_vertices
    # Get the coordinates of the triangles
    gl_triangle_vertices = []
    for triangle_index in trianlge_indices:
        triangle = landmarks_array[triangle_index]
        gl_triangle = np.empty((3, 6))
        for idx, vertex in enumerate(triangle):
            #s = (vertex[0] - x_min) / x_range
            #t = (vertex[1] - y_min) / y_range
            gl_vertex = np.concatenate((vertex, [0.0, 0.0, 1.0, 0.0]), axis=0)
            gl_triangle[idx] = gl_vertex
            
        gl_triangle_vertices.append(gl_triangle)
    # Convert the list of coordinates to a NumPy array
    gl_triangle_vertices = np.reshape(np.array(gl_triangle_vertices), (-1,)).astype('float32')
    return gl_triangle_vertices



def triangulation(landmarks_list):
    landmarks_array = np.array(landmarks_list)
    trianlge_indices = trianlge_vertices
    # Get the coordinates of the triangles
    triangle_coordinates = []
    for triangle_index in trianlge_indices:
        triangle = landmarks_array[triangle_index]
        triangle_coordinates.append(triangle)
    # Convert the list of coordinates to a NumPy array
    triangle_coordinates = np.array(triangle_coordinates)
    return triangle_coordinates
