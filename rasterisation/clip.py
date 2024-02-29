from copy import copy
from typing import List

import numpy as np

from Mesh import Mesh
from MeshVertex import MeshVertex
from ClippingPlane import ClippingPlane


def clip(mesh : Mesh, planes : List[ClippingPlane]) -> Mesh:
    """ clip the mesh with the given planes."""
    clipped_mesh = copy(mesh)
    clipped_mesh.clear()

    for f in range(mesh.faces.shape[0]):
        vertices = mesh.get_face(f).get_vertex(np.arange(mesh.faces[f]))

        positions = vertices.get_position()
        colors = vertices.get_color()
        vertex_count = 3

        for plane in planes:
            vertex_count, positions, colors = clip_plane(vertex_count, positions, colors, plane)

        if vertex_count != 0:
            clipped_mesh.add_face(vertex_count, positions, colors)

    return clipped_mesh

def clip_plane(vertex_count : int, positions : np.ndarray, colors : np.ndarray, plane : ClippingPlane) -> List[np.ndarray]:
    """ clips all vertices defined in positions against the clipping
             plane clipping_plane. Clipping is done by using the Sutherland
             Hodgman algorithm.

        Input Parameter
            vertex_count          ... number of vertices of the face that is clipped
            positions             ... n x 4 matrix with positions of n vertices
                                    one row corresponds to one vertex position
            colors                ... n x 3 matrix with colors of n vertices
                                    one row corresponds to one vertex color
            plane                 ... plane to clip against

        Returns:
            vertex_count_clipped  ... number of resulting vertices after clipping;
                                    this number depends on how the plane intersects
                                    with the face and therefore is not constant
            pos_clipped           ... n x 4 matrix with positions of n clipped vertices
                                    one row corresponds to one vertex position
            col_clipped           ... n x 3 matrix with colors of n clipped vertices
                                    one row corresponds to one vertex color"""
 
    # clear output
    pos_clipped = np.zeros((vertex_count + 1,  4))
    col_clipped = np.zeros((vertex_count + 1,  3))
    vertex_count_clipped = 0

    for n in range(vertex_count):

        current_pos = positions[n]

        next_pos = positions[(n-1)%vertex_count]

        col_current = colors[n]
        col_next = colors[(n-1)%vertex_count]

        t = plane.intersect(current_pos, next_pos)
        pos_intersect = MeshVertex.mix(current_pos, next_pos, t)
        col_intersect = MeshVertex.mix(col_current, col_next, t)

        if(plane.inside(current_pos)):

            if(not plane.inside(next_pos)):
                pos_clipped[vertex_count_clipped] = pos_intersect
                col_clipped[vertex_count_clipped] = col_intersect
                vertex_count_clipped += 1

            pos_clipped[vertex_count_clipped] = current_pos
            col_clipped[vertex_count_clipped] = colors[n]
            vertex_count_clipped += 1
        else:
            if(plane.inside(next_pos)):
                pos_clipped[vertex_count_clipped] = pos_intersect
                col_clipped[vertex_count_clipped] = col_intersect
                vertex_count_clipped += 1

    return vertex_count_clipped, pos_clipped, col_clipped
