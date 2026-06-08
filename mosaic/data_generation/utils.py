
import os
import sys
import bpy
import random
import math
import random
from data_generation.constants import *

def _apply_object_material(obj, color):
    if obj is None or obj.type != 'MESH':
        return

    if obj.data and obj.data.users > 1:
        obj.data = obj.data.copy()

    material = bpy.data.materials.new(name="DistractorMaterial")
    material.use_nodes = True
    bsdf_node = material.node_tree.nodes.get("Principled BSDF")
    bsdf_node.inputs["Base Color"].default_value = color

    if obj.material_slots:
        for slot in obj.material_slots:
            slot.material = material
    else:
        obj.data.materials.append(material)

def calculate_position_with_angle_range(ref_position, obj_size, angle_range, existing_obj, scene_bound=4.0):
    """
    Calculate object position within angle range from reference object.
    Ensures position stays within scene bounds.
    
    Args:
        ref_position: (x, y, z) tuple of reference object
        obj_size: size of object to place
        angle_range: (start_angle, end_angle) tuple in degrees
        existing_obj: list of existing objects for distance calculation
        scene_bound: scene boundary (default 4.0 for comfort_ball)
    
    Returns:
        (x, y, z) tuple of new position, clamped to scene bounds
    """
    start_angle, end_angle = angle_range
    current_angle = random.uniform(start_angle, end_angle)
    angle_rad = math.radians(current_angle)
    
    # Calculate minimum distance from reference object
    min_distance = (obj_size + existing_obj[0]['size']) * 1.5
    
    # Calculate maximum distance based on scene bounds
    max_distance_x_pos = scene_bound - ref_position[0]
    max_distance_x_neg = scene_bound + ref_position[0]
    max_distance_y_pos = scene_bound - ref_position[1]
    max_distance_y_neg = scene_bound + ref_position[1]
    
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    max_dist_x = max_distance_x_pos if cos_angle > 0 else max_distance_x_neg
    max_dist_y = max_distance_y_pos if sin_angle > 0 else max_distance_y_neg
    max_distance = min(max_dist_x, max_dist_y, 6.0)
    max_distance = max(max_distance, min_distance + 0.5)
    
    distance = random.uniform(min_distance, max_distance)
    
    new_x = ref_position[0] + distance * cos_angle
    new_y = ref_position[1] + distance * sin_angle
    
    # Clamp to ensure visibility
    new_x = max(-scene_bound, min(scene_bound, new_x))
    new_y = max(-scene_bound, min(scene_bound, new_y))
    
    return (new_x, new_y, obj_size)

def add_objects(obj_shape, obj_color, obj_size, num_objects, existing_obj=None, comfort_ball=True, angle_range=None, ref_position=None):
    
    # Initialize distractors list if None
    if existing_obj is None:
        existing_obj = []

    existing_positions = []

    iter = 0  # Initialize iter with default value

    if existing_obj:
        for distractor in existing_obj:
            existing_positions.append((distractor['location'], distractor['dimensions']))

        iter = num_objects - len(existing_obj)
    else:
        iter = num_objects if isinstance(num_objects, int) else 0
    
    for idx, _ in enumerate(range(iter)): # here
        placed = False
        while not placed:
            if comfort_ball:
                # If angle_range and ref_position are provided, place within angle range
                if angle_range is not None and ref_position is not None and len(existing_obj) > 0:
                    new_position = calculate_position_with_angle_range(
                        ref_position, obj_size, angle_range, existing_obj
                    )
                elif angle_range is not None and ref_position is None and len(existing_obj) == 0:
                    # First object - central box placement for position dataset
                    box_size = 3.0
                    new_position = (
                        random.uniform(-box_size/2, box_size/2),
                        random.uniform(-box_size/2, box_size/2),
                        obj_size,
                    )
                else:
                    # Original random placement - guaranteed to be visible
                    new_position = (
                        random.uniform(-4, 4),
                        random.uniform(-4, 4),
                        obj_size,
                    )

            collision = False
            for position, dimensions in existing_positions:
                
                offset = dimensions[0] / 1.5 if comfort_ball else dimensions[0] / obj_size / 2
                
                new_offset = obj_size / 1.25 if not comfort_ball else obj_size
                if (
                    abs(new_position[0] - position[0]) < (new_offset + offset) and
                    abs(new_position[1] - position[1]) < (new_offset + offset) and
                    abs(new_position[2] - position[2]) < (new_offset + offset)
                ):
                    collision = True
                    break
            
            if not collision:
                placed = True
                new_obj = add_object(
                    SHAPE_DIR,
                    obj_shape,
                    obj_size,
                    new_position,
                    comfort_ball=comfort_ball
                )
                _apply_object_material(new_obj, obj_color)
                
                existing_positions.append(
                    (new_obj.location, (obj_size, obj_size, obj_size))
                )
                
        existing_obj.append({'location': new_obj.location.copy(), 'dimensions': new_obj.dimensions.copy(), 'shape': obj_shape, 'color': obj_color, 'size': obj_size, 'position': existing_positions})

    return existing_obj if existing_obj else []

def add_object(object_dir, name, scale, loc, theta=0, relation=None, comfort_ball=True):
    """
    Load an object from a file. We assume that in the directory object_dir, there
    is a file named "$name.blend" which contains a single object named "$name"
    that has unit size and is centered at the origin.

    - scale: scalar giving the size that the object should be in the scene
    - loc: tuple (x, y) giving the coordinates on the ground plane where the
      object should be placed.
    """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    print(bpy.data.objects.keys(), file=sys.stderr)
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    if comfort_ball:
        filename = os.path.join(object_dir, "%s.blend" % name, "Object", name)
        bpy.ops.wm.append(filename=filename)
        

        # Give it a new name to avoid conflicts
        new_name = "%s_%d" % (name, count)
        bpy.data.objects[name].name = new_name

    else:
        if "_" in name:
            # For objects with underscore (e.g., bicycle_mountain, car_sedan)
            filename = os.path.join(object_dir, f"{name}.blend", "Object", name.split("_")[1])
            bpy.ops.wm.append(filename=filename)
            new_name = "%s_%d" % (name, count)
            bpy.data.objects[name.split("_")[1]].name = new_name

        else:
            # For objects without underscore, use known names
            new_name_list = {
                "Dog": "German Shepherd Dog",
                "Laptop": "Mcbook Laptop",
                "Basketball": "Basketball",
                "Bed": "Bed Luan3dr",
                "Bench": "Park Bench", # working
                "Chair": "Wooden Chair", # working
                "Duck": "rubber_duck_toy",
                "HorseL": "Horse (Beige) PL",
                "HorseR": "Horse (Brown) PL",
                "Sofa": "Taipei Sofa",
                "Sophia": "Sophia",
            }

            filename = os.path.join(object_dir, f"{name}.blend", "Object", new_name_list[name])
            bpy.ops.wm.append(filename=filename)
            new_name = "%s_%d" % (name, count)
            bpy.data.objects[new_name_list[name]].name = new_name

    x, y, z = loc

    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, z))

    return bpy.context.object


def choose_random_option(value):
    """Return a random entry when given a list/tuple, otherwise return the value as-is."""
    if isinstance(value, (list, tuple)) and value:
        return random.choice(list(value))
    return value

def render_scene_config(
        num_objects: int,
        obj_shape: str,
        obj_color: tuple,
        obj_size: float,
        obj2_shape: str = None,
        obj2_color: tuple = None,
        obj2_size: float = None,
        distractor_shape: str = None,
        distractor_color: tuple = None,
        distractor_size: float = None,
        angle_range: tuple = None,
        cam_position: tuple = None,
        distractors: list = None,
        dataset_name: str = None,
        cuda: bool = True,
        step = None,
        background = BASE_SCENE,
        save_path = None,
        **kwargs
) -> dict:
    # Initialize return variables to ensure they're always defined
    if distractors is None:
        distractors = []
    added_distractors = []
    mapping = {}

    
    bpy.context.scene.render.engine = "CYCLES"
    if True:
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.get_devices()

        if cuda:
            for device in preferences.devices:
                device.use = True
            preferences.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
            
    if "background" in dataset_name:
        background = choose_random_option(background)
        if 'sphere_material' in kwargs:
            kwargs['sphere_material'] = choose_random_option(kwargs.get('sphere_material'))
        if 'ground_material' in kwargs:
            kwargs['ground_material'] = choose_random_option(kwargs.get('ground_material'))
    
    bpy.ops.wm.open_mainfile(filepath=BASE_SCENE)
    bpy.context.scene.render.resolution_x = IM_SIZE
    bpy.context.scene.render.resolution_y = IM_SIZE
    bpy.context.scene.render.resolution_percentage = 100

    if "comfort_ball" in dataset_name:
        comfort_ball = True
    else:
        comfort_ball = False
    
    camera = bpy.data.objects['Camera']
    if comfort_ball:
        camera.location = (7.8342, 0, 3.6126)
    else:
        camera.location = (14.0, 0, 7.0)

    if cam_position is not None:
        camera.location = cam_position
    
    if "count" in dataset_name:
    
        added_distractors = add_objects(
                obj_shape, 
                obj_color, 
                obj_size,
                num_objects, 
                existing_obj= None if num_objects == 1 else added_distractors, 
                comfort_ball=comfort_ball,
            )
        
        bpy.context.scene.render.filepath = save_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered reference image path: {save_path}", file=sys.stderr)
        mapping[f'{save_path}'] = {
            'obj_color': obj_color,
            'obj_shape': obj_shape,
            'obj_size': obj_size,
            'num_objects': num_objects,
        }
        
    elif "attribution" in dataset_name:
        if num_objects <= 2:
            obj_num = 1
        else:
            obj_num = random.randint(1, num_objects - 1)

        added_distractors = add_objects(
                obj_shape, 
                obj_color, 
                obj_size,
                obj_num, 
                existing_obj= None, 
                comfort_ball=comfort_ball,
            )
        
        added_distractors = add_objects(
                obj2_shape, 
                obj2_color, 
                obj2_size,
                num_objects,
                existing_obj=added_distractors,
                comfort_ball=comfort_ball,
                angle_range=angle_range,
                ref_position=added_distractors[0]['location'] if added_distractors else None,
            )
        
        bpy.context.scene.render.filepath = save_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered reference image path: {save_path}", file=sys.stderr)
        mapping[f'{save_path}'] = {
            'obj_color': obj_color,
            'obj_shape': obj_shape,
            'obj_size': obj_size,
            'obj2_color': obj2_color,
            'obj2_shape': obj2_shape,
            'obj2_size': obj2_size,
            'num_obj': obj_num,
            'num_obj2': num_objects - obj_num,
        }

    elif "position" in dataset_name:
        assert angle_range is not None, "Angle range must be provided for position dataset."
        
        # Add first object
        added_distractors = add_objects(
                obj_shape, 
                obj_color, 
                obj_size,
                1, 
                existing_obj=None, 
                comfort_ball=comfort_ball,
                angle_range = angle_range,
            )
        
        # Get first object's position for angle-based placement
        ref_position = added_distractors[0]['location'] if added_distractors else None
        
        # Add second object with angle constraint
        # Pass num_objects=2 so iter = 2 - 1 = 1 (add 1 new object)
        added_distractors = add_objects(
                obj2_shape, 
                obj2_color, 
                obj2_size,
                2,  # Total target is 2 objects
                existing_obj=added_distractors,  # Pass first object so iter = 2 - 1 = 1
                comfort_ball=comfort_ball,
                angle_range=angle_range,
                ref_position=ref_position,
            )
    
        # distractors
        if num_objects > 2:
            assert distractor_shape is not None and distractor_color is not None and distractor_size is not None, "Distractor shape, color, and size must be provided when num_objects > 2 for position dataset."
            distractor_num = random.randint(0, num_objects-2)
            if distractor_num > 0:
                added_distractors = add_objects(
                    distractor_shape, 
                    distractor_color, 
                    distractor_size,
                    distractor_num + 2,
                    existing_obj=added_distractors,
                    comfort_ball=comfort_ball,
                )
        
        bpy.context.scene.render.filepath = save_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered reference image path: {save_path}", file=sys.stderr)
        mapping[f'{save_path}'] = {
            'obj_color': obj_color,
            'obj_shape': obj_shape,
            'obj_size': obj_size,
            'obj2_color': obj2_color,
            'obj2_shape': obj2_shape,
            'obj2_size': obj2_size,
            'num_obj': 1,
            'num_obj2': 1,
            'distractor_shape': distractor_shape,
            'distractor_color': distractor_color,
            'num_distractors': distractor_num if num_objects > 2 else 0,
        }

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return mapping, added_distractors
