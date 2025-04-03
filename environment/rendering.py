import pygame
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
from OpenGL.GL import *
from pygame.locals import *
from gymnasium import error

class OpenGLRenderer:
    def __init__(self, width=1000, height=700):
        self.width = width
        self.height = height
        self.screen = None
        self.clock = None
        self.isopen = False
        self.angle = 0  # For student movement
        self.last_reward = None
        self.reward_timer = 0
        self.animation_timer = 0
        self.font = None
        self.termination_state = None

        # Colors (adjusted per your request)
        self.white = (1.0, 1.0, 1.0)  # Content screen
        self.black = (0.0, 0.0, 0.0)  # Text/lines
        self.red = (1.0, 0.0, 0.0)    # Action indicator
        self.green = (0.0, 1.0, 0.0)  # Understanding bar, active devices
        self.blue = (0.0, 0.0, 1.0)   # Student, energy bar
        self.yellow = (1.0, 1.0, 0.0) # Time bar
        self.ground_color = (0.8, 0.8, 0.8)  # Soft gray for ground
        self.device_panel_color = (0.4, 0.4, 0.6)  # Soft darker blue-gray for devices

    def init_display(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("AccessLearn Navigator - AI Assistant for Visually Impaired Learning")
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        self.clock = pygame.time.Clock()
        self.isopen = True
        self.font = pygame.font.Font(None, 36)

        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glu.gluPerspective(45, (self.width/self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -12)  # Pull back camera for wider view
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))

    def render_static_scene(self, env_state=None, action=None, reward=None, steps=0, max_steps=20, terminated=False):
        if not self.isopen:
            self.init_display()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.9, 0.95, 1.0, 1.0)  # Light blue background for contrast
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -12)
        glRotatef(20, 1, 0, 0)

        if env_state is not None:
            self._draw_student(env_state)  # Student moves in circle
            self._draw_content(env_state, action)
            self._draw_devices(env_state)
            self._draw_status_bars(env_state)
            self._draw_dashboard(env_state, steps, max_steps)
            if action is not None:
                self._draw_action(action)
            if reward is not None:
                self.last_reward = reward
                self.reward_timer = 60
                self.animation_timer = 30

        if self.reward_timer > 0:
            self._draw_reward()
            self.reward_timer -= 1
        if self.animation_timer > 0:
            self.animation_timer -= 1

        if terminated:
            self._draw_termination(steps, env_state)

        self._draw_ground()
        pygame.display.flip()
        self.clock.tick(60)
        self.angle += 1  # Slower movement
        if self.angle >= 360:
            self.angle = 0

    def _draw_student(self, env_state):
        # Student moves in a circle around content (radius 3)
        x = 3 * np.cos(np.radians(self.angle))
        z = 3 * np.sin(np.radians(self.angle))
        glPushMatrix()
        glTranslatef(x, 0, z)
        
        # Humanoid shape
        glColor3f(*self.blue)
        quad = glu.gluNewQuadric()
        if self.animation_timer > 0:  # Head nod
            glRotatef(np.sin(self.animation_timer * 0.5) * 10, 1, 0, 0)
        glu.gluSphere(quad, 0.2, 20, 20)  # Head
        glTranslatef(0, -0.4, 0)
        glColor3f(0.1, 0.1, 0.7)
        self._draw_cuboid(0.3, 0.5, 0.2)  # Torso
        glTranslatef(-0.2, 0.2, 0)
        self._draw_cuboid(0.1, 0.3, 0.1)  # Left arm
        glTranslatef(0.4, 0, 0)
        self._draw_cuboid(0.1, 0.3, 0.1)  # Right arm
        glTranslatef(-0.2, -0.5, 0)
        self._draw_cuboid(0.1, 0.3, 0.1)  # Left leg
        glTranslatef(0, 0, 0.2)
        self._draw_cuboid(0.1, 0.3, 0.1)  # Right leg

        # Impairment Indicator (Eye)
        impairment = env_state[1]
        glTranslatef(0, 0.9, -0.1)
        glColor4f(0, 0, 0, 1.0 - impairment/3)
        self._draw_circle(0.1)
        glPopMatrix()

    def _draw_content(self, env_state, action):
        # Expanded content types
        content_types = ["Text", "Diagram", "Complex Diagram", "Chart", "Video", "Interactive", 
                        "Math Equation", "Map", "3D Model"]
        content_type = int(env_state[0]) % len(content_types)  # Wrap around for more types
        glPushMatrix()
        glTranslatef(0, 0, 0)  # Center of scene
        scale = 1.0 + 0.1 * np.sin(self.animation_timer * 0.5) if self.animation_timer > 0 else 1.0
        glScalef(scale, scale, scale)
        glColor3f(*self.white)
        self._draw_cuboid(1.5, 1.0, 0.1)  # Larger screen

        glTranslatef(0, 0, 0.06)
        if content_type == 0:  # Text
            glColor3f(0, 0, 0)
            for i in range(4):
                glTranslatef(0, -0.2, 0)
                self._draw_cuboid(1.2, 0.05, 0.01)
        elif content_type in [1, 2]:  # Diagrams
            glColor3f(0, 0, 0)
            self._draw_circle(0.5)
            if content_type == 2:
                glTranslatef(0.3, 0.3, 0)
                self._draw_circle(0.2)
        elif content_type == 3:  # Chart
            glColor3f(0, 0.5, 0.8)
            for i in range(4):
                glTranslatef(0.3, 0, 0)
                self._draw_cuboid(0.2, np.random.uniform(0.3, 0.7), 0.01)
        elif content_type == 4:  # Video
            glColor3f(0, 0, 0)
            self._draw_circle(0.4)  # Play button
        elif content_type == 5:  # Interactive
            glColor3f(0.2, 0.6, 0.2)
            self._draw_cuboid(0.4, 0.4, 0.03)
        elif content_type == 6:  # Math Equation
            glColor3f(0, 0, 0)
            self._draw_cuboid(0.8, 0.05, 0.01)  # x²
            glTranslatef(0, 0.2, 0)
            self._draw_cuboid(0.6, 0.05, 0.01)  # +
        elif content_type == 7:  # Map
            glColor3f(0.1, 0.5, 0.1)
            self._draw_cuboid(1.2, 0.8, 0.01)  # Land
            glTranslatef(0, 0, 0.01)
            glColor3f(0, 0, 1)
            self._draw_circle(0.2)  # City
        elif content_type == 8:  # 3D Model
            glColor3f(0.5, 0.5, 0.5)
            glu.gluSphere(glu.gluNewQuadric(), 0.4, 20, 20)
        glPopMatrix()
        self._render_text(content_types[content_type], (self.width*0.45, self.height*0.65), (0, 0, 0))

    def _draw_devices(self, env_state):
        devices = int(env_state[2])
        device_names = ["Screen Reader", "Braille", "Audio", "Tactile"]
        glPushMatrix()
        glTranslatef(-4, 0, 0)  # Moved left for space
        glColor3f(*self.device_panel_color)  # Soft darker blue-gray
        self._draw_cuboid(1, 0.8, 0.1)

        for i, (x, y) in enumerate([(-0.4, 0.3), (0.2, 0.3), (-0.4, 0), (0.2, 0)]):
            glPushMatrix()
            glTranslatef(x, y, 0.06)
            glColor3f(*self.green if (devices & (1 << i)) else (0.5, 0.5, 0.7))  # Lighter blue-gray for inactive
            self._draw_cuboid(0.3, 0.2, 0.03)
            glPopMatrix()
            self._render_text(device_names[i], (self.width*0.1 + x*100, self.height*0.5 - y*100), 
                            (0, 150, 0) if (devices & (1 << i)) else (200, 200, 200))
        glPopMatrix()

    def _draw_status_bars(self, env_state):
        understanding, time_remaining, energy = env_state[3], env_state[4], env_state[6]
        glPushMatrix()
        glTranslatef(4, 1, 0)  # Moved right and up for space

        glColor3f(*self.green)
        self._draw_cuboid(understanding / 5 * 2, 0.1, 0.05)
        self._render_text(f"Understanding: {understanding:.1f}/5", (self.width*0.75, self.height*0.85), (0, 200, 0))
        glTranslatef(0, 0.3, 0)

        glColor3f(*self.yellow)
        self._draw_cuboid(time_remaining / 10 * 2, 0.1, 0.05)
        self._render_text(f"Time: {time_remaining:.1f}/10", (self.width*0.75, self.height*0.9), (255, 255, 0))
        glTranslatef(0, 0.3, 0)

        glColor3f(*self.blue)
        self._draw_cuboid(energy / 10 * 2, 0.1, 0.05)
        self._render_text(f"Energy: {energy:.1f}/10", (self.width*0.75, self.height*0.95), (0, 0, 255))
        glPopMatrix()

    def _draw_action(self, action):
        actions = ["Text to Speech", "Descriptive Audio", "Simplify", "Tactile", "Context", 
                  "Alt Format", "Adjust Pace", "Human Help", "Comprehension Check"]
        glPushMatrix()
        glTranslatef(0, -1.5, 0)  # Lowered for space
        glColor3f(*self.red)
        self._draw_cuboid(1, 0.3, 0.05)
        glPopMatrix()
        self._render_text(f"Action: {actions[action]}", (self.width*0.45, self.height*0.25), (255, 0, 0))

        # Action Effects
        if action in [0, 1]:
            glPushMatrix()
            glTranslatef(0, 0, 0.5)
            glColor4f(0, 0, 1, 0.5)
            self._draw_circle(0.3 + 0.1 * np.sin(self.animation_timer))
            glPopMatrix()
        elif action == 3:
            glPushMatrix()
            glTranslatef(0, 0, 0.5)
            glColor3f(0.5, 0.3, 0.1)
            self._draw_cuboid(0.3, 0.3, 0.1)
            glPopMatrix()

    def _draw_reward(self):
        glPushMatrix()
        glTranslatef(0, 2.5, 0)  # Raised for space
        glColor3f(*self.green if self.last_reward > 0 else self.red)
        self._draw_cuboid(0.5, 0.3, 0.05)
        glPopMatrix()
        self._render_text(f"Reward: {self.last_reward:+.1f}", (self.width*0.45, self.height*0.75), 
                         (0, 255, 0) if self.last_reward > 0 else (255, 0, 0))

    def _draw_dashboard(self, env_state, steps, max_steps):
        glPushMatrix()
        glTranslatef(-4, -2, 0)  # Bottom left
        glColor3f(*self.device_panel_color)  # Match device panel color
        self._draw_cuboid(2, 0.1, 0.05)
        glColor3f(0.5, 0.5, 0.5)
        self._draw_cuboid(steps / max_steps * 2, 0.1, 0.05)
        glPopMatrix()
        self._render_text(f"Steps: {steps}/{max_steps}", (self.width*0.1, self.height*0.15), (100, 100, 100))

    def _draw_termination(self, steps, env_state):
        understanding, time_remaining, energy = env_state[3], env_state[4], env_state[6]
        if understanding >= 4:
            message = "Success: Student Understands!"
            color = (0, 255, 0)
            glClearColor(0.8, 1.0, 0.8, 1.0)
        elif time_remaining <= 0:
            message = "Failed: Time Ran Out!"
            color = (255, 0, 0)
            glClearColor(1.0, 0.8, 0.8, 1.0)
        elif energy <= 0:
            message = "Failed: Energy Depleted!"
            color = (255, 0, 0)
            glClearColor(1.0, 0.8, 0.8, 1.0)
        elif steps >= 20:
            message = "Failed: Max Steps Reached!"
            color = (255, 0, 0)
            glClearColor(1.0, 0.8, 0.8, 1.0)
        else:
            return
        self._render_text(message, (self.width*0.35, self.height*0.5), color, size=48)

    def _draw_ground(self):
        glPushMatrix()
        glTranslatef(0, -2.5, 0)
        glColor3f(*self.ground_color)  # Soft gray
        glBegin(GL_QUADS)
        glVertex3f(-6, 0, -6); glVertex3f(-6, 0, 6); glVertex3f(6, 0, 6); glVertex3f(6, 0, -6)
        glEnd()
        glPopMatrix()

    def _draw_cuboid(self, width, height, depth):
        w, h, d = width/2, height/2, depth/2
        glBegin(GL_QUADS)
        glVertex3f(-w, -h, d); glVertex3f(w, -h, d); glVertex3f(w, h, d); glVertex3f(-w, h, d)
        glVertex3f(-w, -h, -d); glVertex3f(-w, h, -d); glVertex3f(w, h, -d); glVertex3f(w, -h, -d)
        glVertex3f(-w, h, -d); glVertex3f(-w, h, d); glVertex3f(w, h, d); glVertex3f(w, h, -d)
        glVertex3f(-w, -h, -d); glVertex3f(w, -h, -d); glVertex3f(w, -h, d); glVertex3f(-w, -h, d)
        glVertex3f(w, -h, -d); glVertex3f(w, h, -d); glVertex3f(w, h, d); glVertex3f(w, -h, d)
        glVertex3f(-w, -h, -d); glVertex3f(-w, -h, d); glVertex3f(-w, h, d); glVertex3f(-w, h, -d)
        glEnd()

    def _draw_circle(self, radius):
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for i in range(361):
            angle = i * 3.14159 / 180
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()

    def _render_text(self, text, pos, color, size=36):
        if not self.font:
            return
        font = pygame.font.Font(None, size)
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glRasterPos2d(pos[0]/self.width*12-6, pos[1]/self.height*7-3.5)  # Adjusted for new view
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    def close(self):
        if self.isopen:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

def create_static_visualization(env=None):
    renderer = OpenGLRenderer()
    renderer.init_display()
    running = True
    env_state = None if env is None else env.reset()[0]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        renderer.render_static_scene(env_state)
    
    renderer.close()
    return True