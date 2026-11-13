"""
    This module contains util methods for the visualization of the track in the MuJoCo sim
"""

import jax.numpy as jp
import xml.etree.ElementTree as ET

def distance(p1, p2):
    return jp.linalg.norm(p1 - p2)

def midpoint(p1, p2):
    return (p1 + p2) / 2

def rotation_to_align(p1, p2):
    """
    determines the roation of the track lines between to track points p1, p2 for mjc
    """
    x_axis = jp.array([1.0, 0.0])
    direction = p2 - p1
    
    # Normalize
    dir_norm = direction / jp.linalg.norm(direction)
    
    # Angle (radians)
    dot = jp.dot(x_axis, dir_norm)
    angle_rad = jp.arccos(dot)
    # Determine direction (sign) using cross product
    cross = x_axis[0] * dir_norm[1] - x_axis[1] * dir_norm[0]
    signed_angle_rad = angle_rad * jp.sign(cross)

    # Convert to degrees and make it positive clockwise
    angle_deg = jp.degrees(signed_angle_rad)
    return angle_deg


def create_line_element(pos, angle: float, length: float, rgba: tuple[float, float, float, float]):
    line_element = ET.Element("geom")
    line_element.set("type", "box")
    line_element.set("size", f"{length / 2} 0.005  0.0001")
    line_element.set("pos", f"{pos[0]} {pos[1]} 0")
    line_element.set("euler", f"0 0 {angle}")
    line_element.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
    return line_element

def compute_line_element(p1, p2, rgba: tuple[float, float, float, float] = (0.9, 0, 0, 1)):
    """
    Computes the line element between two points p1 and p2.
    """
    pos = midpoint(p1, p2)
    angle = rotation_to_align(p1, p2)
    length = distance(p1, p2)
    return create_line_element(pos, angle, length, rgba)

def compute_circle_element(xy: tuple[float, float], radius: float, rgba: tuple[float, float, float, float], z: float = 0):
    """
    Adds a circular ground geom (like the 'track' cylinder) at the given xy coordinates with the given color.
    
    Args:
        xy: tuple (x, y) coordinates where the circle should be placed.
        rgba: tuple (r, g, b, a) color values.
        z: height of the circle above the ground.
    """

    # create the new circle geom
    geom = ET.Element("geom")
    geom.set("type", "cylinder")
    geom.set("size", f"{radius} 0.0001") 
    geom.set("pos", f"{xy[0]} {xy[1]} {z}")
    geom.set("rgba", " ".join(str(c) for c in rgba))
    geom.set("contype", "0")
    geom.set("conaffinity", "0")

    return geom
