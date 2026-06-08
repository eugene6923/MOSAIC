"""
Configuration handling utilities.
This module handles loading and processing of configuration files.
"""

from data_generation.constants import *

def get_color_from_name(color_name):
    """Convert color name string to color tuple"""
    available_colors = [GREEN, RED, BLUE, YELLOW, PURPLE, GRAY, WHITE, ORANGE, CYAN, BLACK]

    color_map = {
        "RED": RED,
        "GREEN": GREEN,
        "BLUE": BLUE,
        "YELLOW": YELLOW,
        "PURPLE": PURPLE,
        "ORANGE": ORANGE,
        "CYAN": CYAN,
        "GRAY": GRAY,
        "WHITE": WHITE,
        "BLACK": BLACK,
        "PINK": PINK,
        "BROWN": BROWN,
        "MAROON": MAROON,
        "OLIVE": OLIVE,
        "LIME": LIME,
        "NAVY": NAVY,
        "TEAL": TEAL,
        "SILVER": SILVER,
        "GOLD": GOLD,
        "AVAILABLE_COLORS": available_colors,
        "available_colors": available_colors,  # Handle lowercase version

    }
    return color_map.get(color_name.upper(), BLUE)


def get_shape_from_name(shape_name):
    """Convert shape name string to shape constant"""
    # Available comfort car objects (excluding SOPHIA which is addressee)
    available_objects = [
        BICYCLE_MOUNTAIN, CAR_SEDAN, COUCH, BASKETBALL, 
        DOG, BED, DUCK, LAPTOP, BENCH, CHAIR
    ]
    
    shape_map = {
        "SPHERE": SPHERE,
        "Sphere": SPHERE,
        "CUBE": CUBE,
        "Cube": CUBE,
        "CYLINDER": CYLINDER,
        "AVAILABLE_SHAPES": [SPHERE, CUBE, CYLINDER],
        "AVAILABLE_OBJECTS": available_objects,
        "available_objects": available_objects
    }
    return shape_map.get(shape_name.upper(), SPHERE)




def color_to_name(color_tuple):
    """Convert color tuple back to name for saving."""
    color_map = {
        RED: "RED",
        GREEN: "GREEN",
        BLUE: "BLUE",
        YELLOW: "YELLOW",
        PURPLE: "PURPLE",
        ORANGE: "ORANGE",
        CYAN: "CYAN",
        GRAY: "GRAY",
        WHITE: "WHITE",
        BLACK: "BLACK",
        PINK: "PINK",
        BROWN: "BROWN",
        MAROON: "MAROON",
        OLIVE: "OLIVE",
        LIME: "LIME",
        NAVY: "NAVY",
        TEAL: "TEAL",
        SILVER: "SILVER",
        GOLD: "GOLD",
    }
    
    # Handle tuple comparison
    for color, name in color_map.items():
        if color == color_tuple:
            return name
    
    return str(color_tuple)  # Fallback to string representation