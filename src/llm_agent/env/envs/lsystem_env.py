import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
import re
import open3d as o3d

# Define the L-System rules
l_system_def = '''
class LSystem3D:
    def __init__(self, axiom, rules):
        self.axiom = axiom
        self.rules = rules
        self.iterations = 5
        self.angle = np.radians(30)
        self.length = 0.7
        self.result = self.generate()

    def generate(self):
        current_string = self.axiom
        for _ in range(self.iterations):
            next_string = "".join(self.rules.get(c, c) for c in current_string)
            current_string = next_string
        return current_string

    def get_segments_with_width(self):
        pos = np.array([0, 0, 0])
        direction = np.array([0, 0, 1])  # Initial growth along z-axis
        stack = []
        segments = []
        current_color = 'brown'
        current_width = 1.0  # Default width
        colors = {'C0': 'brown', 'C1': 'green', 'C2': 'darkgreen', 'C3': 'yellowgreen'}
        
        # Extract commands including width control (Wn)
        commands = re.findall(r'C[0-3]|W\d*\.?\d+|[F\[\]\+\-&^<>]', self.result)

        for command in commands:
            if command.startswith("W"):  # Set width
                current_width = float(command[1:])
            elif command == "F":  # Move forward and draw
                new_pos = pos + self.length * direction
                segments.append((pos.copy(), new_pos.copy(), current_color, current_width))
                pos = new_pos
            elif command == "[":  # Push state
                stack.append((pos.copy(), direction.copy(), current_color, current_width))
            elif command == "]":  # Pop state
                pos, direction, current_color, current_width = stack.pop()
            elif command in colors:  # Change color
                current_color = colors[command]
            else:  # Handle rotations
                rotation_matrices = {
                    "+": np.array([[1, 0, 0], [0, np.cos(self.angle), -np.sin(self.angle)], [0, np.sin(self.angle), np.cos(self.angle)]]),
                    "-": np.array([[1, 0, 0], [0, np.cos(-self.angle), -np.sin(-self.angle)], [0, np.sin(-self.angle), np.cos(-self.angle)]]),
                    "&": np.array([[np.cos(self.angle), 0, np.sin(self.angle)], [0, 1, 0], [-np.sin(self.angle), 0, np.cos(self.angle)]]),
                    "^": np.array([[np.cos(-self.angle), 0, np.sin(-self.angle)], [0, 1, 0], [-np.sin(-self.angle), 0, np.cos(-self.angle)]]),
                    "<": np.array([[np.cos(self.angle), -np.sin(self.angle), 0], [np.sin(self.angle), np.cos(self.angle), 0], [0, 0, 1]]),
                    ">": np.array([[np.cos(-self.angle), -np.sin(-self.angle), 0], [np.sin(-self.angle), np.cos(-self.angle), 0], [0, 0, 1]])
                }
                if command in rotation_matrices:
                    direction = np.dot(rotation_matrices[command], direction)
        
        return segments
'''

# Define visualization function
def plot_3d_tree(segments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for start, end, color in segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def plot_3d_tree_open3d(segments):
    lines = []
    colors = []
    color_map = {
        'brown': [0.5, 0.3, 0.1], 
        'green': [0.0, 0.5, 0.0], 
        'darkgreen': [0.0, 0.3, 0.0], 
        'yellowgreen': [0.6, 0.8, 0.2]
    }
    
    points = []
    for start, end, color in segments:
        idx1, idx2 = len(points), len(points) + 1
        points.extend([start, end])
        lines.append([idx1, idx2])
        colors.append(color_map.get(color, [1, 1, 1]))  # Default white
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(points)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    )
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

    o3d.visualization.draw_geometries([line_set])

def create_cylinder(start, end, radius, color):
    """Creates a cylinder mesh from start to end with given radius and color"""
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(end - start))
    cylinder.paint_uniform_color(color)
    
    # Compute direction vector and align cylinder
    direction = (end - start) / np.linalg.norm(end - start)
    default_direction = np.array([0, 0, 1])  # Open3D cylinder points along Z by default

    # Compute rotation matrix to align default Z with our direction
    v = np.cross(default_direction, direction)
    s = np.linalg.norm(v)
    c = np.dot(default_direction, direction)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2 if s != 0 else 1))

    # Apply rotation and translation
    cylinder.rotate(R, center=(0, 0, 0))
    cylinder.translate(start)

    return cylinder

def plot_3d_tree_open3d_and_save_to_file(segments, filename="tree.png", save_type="image"):
    """
    Visualizes and saves the 3D tree using Open3D.
    
    Parameters:
        segments (list): List of tree branch segments [(start, end, color, width)].
        filename (str): Output file name (supports .png, .ply, .obj).
        save_type (str): "image" (PNG), "pointcloud" (PLY), or "mesh" (OBJ).
    """
    points, lines, colors = [], [], []
    color_map = {'brown': [0.5, 0.3, 0.1], 'green': [0.0, 0.5, 0.0], 'darkgreen': [0.0, 0.3, 0.0], 'yellowgreen': [0.6, 0.8, 0.2]}

    for start, end, color, width in segments:
        idx1, idx2 = len(points), len(points) + 1
        points.extend([start, end])
        lines.append([idx1, idx2])
        colors.append(color_map.get(color, [1, 1, 1]))  # Default white

    print("Creating Open3D LineSet")
    # Create Open3D LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(points)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    )
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

    print("Saving tree visualization")
    if save_type == "image":
        # Save as image
        vis = o3d.visualization.Visualizer()
        print("Creating window")
        vis.create_window(visible=False)  # Headless mode
        print("Adding geometry")
        vis.add_geometry(line_set)
        print("Polling events")
        vis.poll_events()
        print("Updating renderer")
        vis.update_renderer()
        print("Capturing screen image")
        vis.capture_screen_image(filename)
        print("Destroying window")
        vis.destroy_window()
        print(f"Saved tree visualization as {filename}")
    
    elif save_type == "pointcloud":
        # Save as point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved tree point cloud as {filename}")

    elif save_type == "mesh":
        # Save as a mesh-like representation
        o3d.io.write_line_set(filename, line_set)
        print(f"Saved tree structure as {filename}")

    else:
        print("Invalid save_type. Use 'image', 'pointcloud', or 'mesh'.")

from typing import Dict, List, Optional, Tuple
import numpy as np
from llm_agent.env.base_env import BaseEnv, Observation, Action

class LSystemEnv(BaseEnv):
    """Environment for generating L-system rules based on user instructions"""
    
    def __init__(self, config: Dict):
        """Initialize L-system environment
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_steps = config.get('max_steps', 10)
        self._observation = None
        self.steps = 0
        self.id = config.get('problem_id', 0)
        self.category = "lsystem"
        self.axiom = config.get('axiom', "F")
        self.iterations = config.get('iterations', 5)
        self.angle = config.get('angle', 30)
        self.length = config.get('length', 0.7)
        
    def reset(self) -> Observation:
        """Reset environment to initial state
        
        Returns:
            Initial observation containing user instructions for the desired tree
        """
        self.steps = 0
        
        # Get initial user instructions for the desired tree
        goal = "Create a short, fat tree."
        self.goal = goal
        
        self._observation = f"Goal: {goal}\nCreate L-system rules to generate the described tree. Use the format:\nrules = {{\"F\": \"replacement string\"}}"
        return self._observation, {}
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Take action in environment by executing generated L-system rules
        
        Args:
            action: Python code containing L-system rules
            
        Returns:
            Tuple containing:
            - Next observation (visualization result and feedback)
            - Reward (based on user input)
            - Done flag 
            - Info dict
        """
        self.steps += 1
        
        # For now, assume reward = "todo", done = True, info = {}
        return "todo", 0, True, {}
        
    def get_action_space(self) -> List[str]:
        """Return the action space for the environment
        
        Returns:
            List of valid action types (only Python code for L-system rules)
        """
        return {
            "type": "string",
            "description": f"""
                You will write the L-system rules to generate the tree. The L-system rules will be written in a Python dictionary format. The dictionary will have a key 'F' and a value which is the replacement string for the 'F' symbol. The following code will be used to generate the tree: {l_system_def}. Your response must be a single line. Anything after a newline character will be ignored.
            """
        }