from typing import List, Tuple

import numpy as np
import math

def define_triangle() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P1 = np.array([(1 + 2), -(1 + 1), -(1 + 9)])
    P2 = np.array([-(0 + 2), -(2 + 1), (2 + 9)])
    P3 = np.array([-(1 + 1), (1 + 4), -(1 + 2)])

    return P1, P2, P3

def define_triangle_vertices(P1:np.ndarray, P2:np.ndarray, P3:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P1P2 = P2 - P1
    P2P3 = P3 - P2
    P3P1 = P1 - P3

    return P1P2, P2P3, P3P1

def compute_lengths(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> List[float]:
    distance_P1P2 = math.sqrt(sum(pow(P1P2, 2)))
    distance_P2P3 = math.sqrt(sum(pow(P2P3, 2)))
    distance_P3P1 = math.sqrt(sum(pow(P3P1, 2)))
    norms = np.array([distance_P1P2, distance_P2P3, distance_P3P1])

    return norms

def compute_normal_vector(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.cross(P1P2, P3P1)
    lenght_n = math.sqrt(sum(pow(n, 2)))
    n_normalized = n / lenght_n

    return n, n_normalized

def compute_triangle_area(n:np.ndarray) -> float:
    area = 0.5 * math.sqrt(sum(pow(n ,2)))

    return area
def compute_angles(P1P2: np.ndarray, P2P3: np.ndarray, P3P1: np.ndarray) -> Tuple[float, float, float]:
    # Calculate the squares of the lengths of the sides
    a_squared = np.sum(np.square(P2P3))
    b_squared = np.sum(np.square(P3P1))
    c_squared = np.sum(np.square(P1P2))

    # Calculate the angles using the law of cosines
    alpha = math.degrees(np.arccos((b_squared + c_squared - a_squared) / (2 * math.sqrt(b_squared * c_squared))))
    beta = math.degrees(np.arccos((c_squared + a_squared - b_squared) / (2 * math.sqrt(c_squared * a_squared))))
    gamma = math.degrees(np.arccos((a_squared + b_squared - c_squared) / (2 * math.sqrt(a_squared * b_squared))))

    return alpha, beta, gamma
