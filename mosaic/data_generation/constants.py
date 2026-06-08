# COMFORT BALL ASSETS
SPHERE = "Sphere"
CUBE = "ShapeCube"
CYLINDER = "ShapeCylinder"

# COMFORT CAR ASSETS
BICYCLE_MOUNTAIN = "bicycle_mountain"
CAR_SEDAN = "car_sedan"
COUCH = "Sofa"
BASKETBALL = "Basketball"
CHAIR = "Chair"
DOG = "Dog"
BED = "Bed"
DUCK = "Duck"
LAPTOP = "Laptop"
HORSE_R = "HorseR"
BENCH = "Bench"
SOPHIA = "Sophia" # addressee
HORSE_L = "HorseL"

SPECIAL = [SOPHIA, COUCH, BASKETBALL, CHAIR, DOG, BED, DUCK, LAPTOP, HORSE_L, HORSE_R, BENCH]

# define color codes
RED = (1, 0, 0, 1)
GREEN = (0, 1, 0, 1)
BLUE = (0, 0, 1, 1)
YELLOW = (1, 1, 0, 1)
PURPLE = (1, 0, 1, 1)
ORANGE = (1, 0.5, 0, 1)
CYAN = (0, 1, 1, 1)
GRAY = (0.5, 0.5, 0.5, 1)
DARK_GRAY = (0.1, 0.1, 0.1, 1)
WHITE = (1, 1, 1, 1)
AIRPLANE_WHITE = (0.937, 0.937, 0.957, 1)
CHARCOAL_GRAY = (0.3, 0.3, 0.3, 1)
CAR_RED = (0.95, 0.1, 0.1, 1)
CAR_BLUE = (0.1, 0.3, 0.9, 1)
BLACK = (0, 0, 0, 1)

# new colors by me
PINK = (1, 0.75, 0.8, 1)
BROWN = (0.6, 0.4, 0.2, 1)
MAROON = (0.5, 0, 0, 1)
OLIVE = (0.5, 0.5, 0, 1)
LIME = (0.75, 1, 0, 1)
NAVY = (0, 0, 0.5, 1)
TEAL = (0, 0.5, 0.5, 1)
SILVER = (0.75, 0.75, 0.75, 1)
GOLD = (1, 0.84, 0, 1)
CHARCOAL_GRAY = (0.3, 0.3, 0.3, 1)

COLORS = {
    "RED": [RED, CAR_RED],
    "GREEN": [GREEN],
    "BLUE": [BLUE, CAR_BLUE],
    "YELLOW": [YELLOW],
    "PURPLE": [PURPLE],
    "GRAY": [GRAY, CHARCOAL_GRAY],
    "WHITE": [WHITE],
    "ORANGE": [ORANGE],
    "CYAN": [CYAN],
    "BLACK": [BLACK],

}


def color_to_name(color):
    """Convert a color value to its name."""
    for name, value in COLORS.items():
        if color in value:
            return name.lower()
    return "unknown"


BEHIND = "behind"
FRONT = "infrontof"
LEFT = "totheleft"
RIGHT = "totheright"

ROTATION_LIST = [BEHIND, FRONT, LEFT, RIGHT]
ALL_RELATIONS = ROTATION_LIST

BASE_SCENE = "data_generation/background/base_scene_centered.blend"
# BASE_SCENE = "data_generation/background/base_scene_with_ground_material4.blend"
# BASE_SCENE = "data_generation/background/base_scene_with_ground_material.blend"
# BASE_SCENE = "data_generation/assets/clevrtexv2_materials/coast_sand_rocks_02_4k_packed.blend"
# BASE_SCENE = "data_generation/background/base_scene_centered.blend"  # Original without ground material
MATERIAL_DIR = "data_generation/materials/"
SHAPE_DIR = "data_generation/assets/"

IM_SIZE = 512
