import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate
from .utils import *


class _Canvas(app.Canvas):

    vertex_shader = """
    attribute vec3 position;
    attribute vec3 color_in;

    uniform mat4 model;
    uniform mat4 view;
    uniform int point_size;
    uniform mat4 projection;
    varying vec3 color;

    void main() {
        gl_Position = projection * view * model * vec4(position, 1.0);
        gl_PointSize = point_size;
        color = color_in;
    }
    """

    fragment_shader = """
    varying vec3 color;

    void main() {
        gl_FragColor = vec4(color, 1.0);
    }
    """
    
    def __init__(self, 
                 point_clouds,  # F, N, 3 
                 color=None,  # N, 3 or F, N, 3
                 fps=24,
                 point_size=5,
                 ):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600),
                            title='Interactive Point Clouds')
        self.program = gloo.Program(self.vertex_shader, self.fragment_shader)
        self.point_clouds = point_clouds  - np.mean(point_clouds.reshape(-1, 3), axis=0)

        max_value = np.max(np.abs(self.point_clouds))
        self.point_clouds *= 2. / max_value

        self.current_frame = 0
        sequence_speed = 1. / fps
        self.timer = app.Timer(interval=sequence_speed, connect=self.on_timer, start=True)

        # Camera parameters
        self.view = translate((0, 0, -5))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = perspective(45.0, self.size[0] / float(self.size[1]), 1.0, 100.0)

        self.program['model'] = self.model
        self.program['view'] = self.view
        self.program['projection'] = self.projection
        self.program['point_size'] = point_size

        self.theta, self.phi = 0, 0
        self.mouse_pos = 0, 0
        self.wheel_pos = 0
        
        self.color = color
        if not self.color is None:
            self.color_seq = len(self.color.shape) == 3

        self.init = True

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        current_point_cloud = self.point_clouds[self.current_frame % len(self.point_clouds)]
        self.program['position'] = current_point_cloud.astype(np.float32)
        if self.color is not None:
            if not self.color_seq:
                self.program['color_in'] = self.color.astype(np.float32)
            else:
                self.program['color_in'] = self.color[self.current_frame % len(self.point_clouds)].astype(np.float32)
        else:
            self.program['color_in'] = np.ones_like(current_point_cloud).astype(np.float32)
        
        self.program.draw('points')

    def on_resize(self, event):
        if not hasattr(self, 'init'): return
        self.projection = perspective(45.0, event.size[0] / float(event.size[1]), 1.0, 100.0)
        self.program['projection'] = self.projection

    def on_mouse_move(self, event):
        x, y = event.pos
        dx, dy = x - self.mouse_pos[0], y - self.mouse_pos[1]
        self.mouse_pos = (x, y)

        if event.is_dragging:
            self.theta += dx
            self.phi += dy

            self.model = np.dot(rotate(self.theta, (0, 1, 0)), rotate(self.phi, (1, 0, 0)))
            self.program['model'] = self.model
            self.update()

    def on_mouse_wheel(self, event):
        self.wheel_pos += event.delta[1]
        self.view = translate((0, 0, -5 - 0.1 * self.wheel_pos))
        self.program['view'] = self.view
        self.update()

    def on_timer(self, event):
        self.current_frame += 1
        self.update()
        
        
def play(point_clouds,  # F, N, 3
         fps=24,
         color=None,
         point_size=5):
    player = _Canvas(point_clouds=point_clouds,
                     fps=fps,
                     color=color,
                     point_size=point_size)
    player.show(run=True)
        
        
def play_static(point_clouds,  # N, 3 
                  color=None,  # N, 3
                  point_size=5):
    player = _Canvas(point_clouds=point_clouds[None],
                     fps=1.,
                     color=color[None] if color is not None else None,
                     point_size=point_size)
    player.show(run=True)


def demo(num_points=1000,
         frames=100):
    pcd_init = generate_synthetic_point_cloud(num_points)[None]  # 1, N, 3
    rand_diff = (np.random.rand(frames, num_points, 3) - 0.5) * 2. / frames
    traj = np.cumsum(rand_diff, axis=0)  # F, N, 3
    point_clouds_sequence = pcd_init + traj

    play(point_clouds_sequence, 
         fps=24, 
         color=generate_gradient_color(pcd_init[0, :, 2]))
    

def demo_static(num_points=1000):
    pcd_init = generate_synthetic_point_cloud(num_points)  # N, 3
    play_static(pcd_init, color=generate_gradient_color(pcd_init[..., 2]))
