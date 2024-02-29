from typing import List
import numpy as np

class ClippingPlane:

    def __init__(self, plane : np.ndarray):
        """ plane     ... plane stored in Hessian normal form as a 1x4 vector"""
        self.plane = plane
    
    def inside(self, pos : np.ndarray) -> bool:
        """Checks if a given point lies behind the plane (opposite direction
        of normal vector). Points lying on the plane are considered to be
        inside.
        position  ... homogeneous position with 4 components
        return res... logical value which indicates if the point is
                      inside or not """

        res = np.dot(pos, self.plane) < 0
        return res

    def intersect(self, pos1 : np.ndarray, pos2 : np.ndarray) -> float:
        """ Intersects the plane with a line between pos1 and pos2.
        pos1      ... homogeneous position with 4 components
        pos2      ... homogeneous position with 4 components
        return t  ... normalized intersection value t in [0, 1]"""

        a = np.dot(pos1, self.plane)
        b = np.dot(pos2, self.plane)

        # a-b != 0 -> a != b
        t = (a)/(a-b) if(a != b) else 0

        return t
    
    @staticmethod
    def get_clipping_planes() -> List:
        """creates and returns a list of the six Clipping planes defined in the task description."""

        res = [
            ClippingPlane(np.array([1, 0, 0, -1])), 
            ClippingPlane(np.array([-1, 0, 0, -1])), 
            ClippingPlane(np.array([0, 1, 0, -1])), 
            ClippingPlane(np.array([0, -1, 0, -1])),
            ClippingPlane(np.array([0, 0, 1, -1])),
            ClippingPlane(np.array([0, 0, -1, -1]))
        ]

        return res
