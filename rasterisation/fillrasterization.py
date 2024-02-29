import numpy as np
from numpy.matlib import repmat

from MeshVertex import MeshVertex
from Framebuffer import Framebuffer
from MeshVertex import MeshVertex

def fill_rasterization(mesh : MeshVertex, framebuffer : Framebuffer):
    """ applies the fill rasterization algorithm. Draws a mesh to the Framebuffer."""

    for i in range(mesh.faces.shape[0]):
        v1 = mesh.get_face(i).get_vertex(0)
        for j in range(mesh.faces[i][0]-1):
            i, j = np.array(i).reshape(np.asarray(i).size), np.array(j).reshape(np.asarray(j).size)

            v2 = mesh.get_face(i).get_vertex(j)
            v3 = mesh.get_face(i).get_vertex(j+1)
            draw_triangle(framebuffer, v1, v2, v3)


def line_eq(A : float, B : float, C : float, x : float, y : float) -> float:
    """defines the line equation described by the provided parameters and
        returns the distance of a point (x, y) to this line.
        A    ... line equation parameter 1
        B    ... line equation parameter 2
        C    ... line equation parameter 3
        x    ... x coordinate of point to test against the line
        y    ... y coordinate of point to test against the line
        res  ... distance of the point (x, y) to the line (A, B, C)."""

    res = A * x + B * y + C
    return res

def draw_triangle(framebuffer : Framebuffer, v1 : MeshVertex, v2 : MeshVertex, v3 : MeshVertex):
    """ draws a triangle defined by v1,v2,v3 to the given framebuffer"""
    
    x1, y1, depth1 = v1.get_screen_coordinates()
    x2, y2, depth2 = v2.get_screen_coordinates()
    x3, y3, depth3 = v3.get_screen_coordinates()

    col1 = v1.get_color()
    col2 = v2.get_color()
    col3 = v3.get_color()

    # calc triangle area * 2
    a = ((x3-x1)*(y2-y1) - (x2-x1)*(y3-y1))

    if not np.isclose(a, 0):
        # Swap order of clockwise triangle to make them counter-clockwise
        if a < 0:
            t = x2
            x2 = x3 
            x3 = t

            t = y2
            y2 = y3
            y3 = t

            t = depth2
            depth2 = depth3
            depth3 = t

            t = col2
            col2 = col3
            col3 = t

        e1 = np.array([x3 - x2, y3 - y2])
        e2 = np.array([x1 - x3, y1 - y3])
        e3 = np.array([x2 - x1, y2 - y1])

        c1 = -((-e1[1])*x2 + e1[0]*y2)
        c2 = -((-e2[1])*x3 + e2[0]*y3)
        c3 = -((-e3[1])*x1 + e3[0]*y1)

        o1v1 = line_eq(-e1[1],e1[0],c1,x1,y1)
        o2v2 = line_eq(-e2[1],e2[0],c2,x2,y2)
        o3v3 = line_eq(-e3[1],e3[0],c3,x3,y3)

        for x in range(int(min(x1,x2,x3)),int(max(x1,x2,x3))):
            for y in range(int(min(y1,y2,y3)),int(max(y1,y2,y3))):
                o1 = line_eq(-e1[1],e1[0],c1,x,y)
                o2 = line_eq(-e2[1],e2[0],c2,x,y)
                o3 = line_eq(-e3[1],e3[0],c3,x,y)
                if(o1 <= 0 and o2 <= 0 and o3 <= 0):
                    bary_a = o1/o1v1
                    bary_b = o2/o2v2
                    bary_c = o3/o3v3  
                    col_interp = MeshVertex.barycentric_mix(col1,col2,col3,bary_a,bary_b,bary_c)
                    depth_interp = MeshVertex.barycentric_mix(depth1,depth2,depth3,bary_a,bary_b,bary_c)
                    framebuffer.set_pixel(np.array([x]),np.array([y]),depth_interp, col_interp)
