import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pyquaternion import Quaternion

from link import Link


def funkify(v):
    #  triple quotes allow comments to span multiple lines in python
    """Returns a skew-symmetric matrix M for input vector v such that cross(v, k) = M @ k"""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


########################################################################################################################

class BaseEnv:
    def __init__(self):
        self.window = 0  # render window ID
        self.sim_time = 0  # current sim time
        self.dT = 0.003  # sim time step
        self.sim_running = True  # toggle to pause simulation
        self.RAD_TO_DEG = 180.0 / np.pi
        self.GRAVITY = -9.81
        self.anchor = np.array([0.0, 1.0, 0.0])  # fixed pendulum anchor point location in the world
        self.link_length = 0.5  # link length
        self.link_thickness = 0.04  # link width and depth
        self.link_mass = 1.0  # link mass
        self.kp = 20.0  # Baumgarte stabilization spring constant
        self.kd = 1.0  # Baumgarte stabilization damping constant
        self.cp = 4000.0  # ground penalty spring constant
        self.cd = 50.0  # ground penalty damping constant
        self.damp = 0.08  # angular velocity damping, to remove energy over time
        self.plane = True  # toggle ground plane on-and-off
        self.plane_height = 0.0
        self.links = []

        self.args = None
        self.energies = []
        self.potentials = []
        self.times = []

        # set up GLUT
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)  # display mode
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow(b"CPSC 533V Simulation Template")
        glutIdleFunc(self.inner_loop)  # callback function, repeatedly called when idle
        glutReshapeFunc(self.resize_gl_scene)  # callback when the render window is resized
        glutKeyboardFunc(self.key_pressed)  # callback for keystrokes
        glutDisplayFunc(self.render)  # callback to render
        self.init_gl(640, 480)

    #   simple stub in the base class, specifies 2-links
    def reset(self):
        self.reset_sim(2)  # reset with a given number of links

    def reset_sim(self, num_links):  # resets simulation to fixed initial conditions
        print("Simulation reset")
        self.sim_running = True
        self.sim_time = 0

        colors = ([0.8, 0.5, 0.5], [0.5, 0.8, 0.5])
        angle = np.pi / 2  # start rotated 90 degrees about the axis given below
        axis = np.array([1, 0, 1])  # axis for the initial rotation
        axis = axis / np.linalg.norm(axis)

        # clear existing links
        self.links = []
        # clear stored energies
        self.energies = []
        self.potentials = []
        self.times = []

        #  first link
        link = Link()
        link.set_cuboid(self.link_mass, self.link_thickness, self.link_length, self.link_thickness)
        link.color = colors[0]
        link.q_rot = Quaternion(axis=axis, angle=angle)  # initial orientation
        link.pos = self.anchor - link.get_r()  # initial position
        self.links.append(link)

        # remaining links
        for i in range(1, num_links):
            link = Link()  # create a link
            prev_link = self.links[i - 1]  # previous link
            link.set_cuboid(self.link_mass, self.link_thickness, self.link_length, self.link_thickness)
            link.color = colors[i % 2]
            link.q_rot = Quaternion(axis=axis, angle=angle)
            link.pos = prev_link.pos - prev_link.get_r() - link.get_r()
            self.links.append(link)

    def key_pressed(self, key, x, y):
        ch = key.decode("utf-8")
        if ch == ' ':  # toggle the simulation
            if self.sim_running:
                self.sim_running = False
            else:
                self.sim_running = True
        elif ch == chr(27):  # ESC to quit
            sys.exit()
        elif ch == 'q':  # q to quit
            sys.exit()
        elif ch == 'r':  # reset simulation
            self.reset_sim(len(self.links))
        elif ch == '+':  # more links
            n = len(self.links)
            n2 = min(8, n + 1)
            if n2 != n:
                self.reset_sim(n2)
        elif ch == '-':  # fewer links
            n = len(self.links)
            n2 = max(1, n - 1)
            if n2 != n:
                self.reset_sim(n2)
        elif ch == 'p':  # toggle ground plane
            self.plane = not self.plane

    #   provide a simple do-nothing stub in this base class
    def step(self):
        pass

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(1, 1, 3, 0, 0.5, 0, 0, 1, 0)  # eyePoint, lookatPoint, upVector

        self.draw_origin()
        if self.plane:
            self.draw_ground()
        for link in self.links:
            link.draw()

        glutSwapBuffers()

    def draw_origin(self):
        glLineWidth(3.0)

        glColor3f(1, 0.5, 0.5)  # light red x-axis
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        glEnd()

        glColor3f(0.5, 1, 0.5)  # light green y-axis
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        glEnd()

        glColor3f(0.5, 0.5, 1)  # light blue z-axis
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()

    @staticmethod
    def draw_ground():
        glLineWidth(3.0)

        size = 2.0
        glColor3f(0.85, 0.8, 0.8)  # greyish-red
        eps = 0.001
        glBegin(GL_QUADS)
        glVertex3f(-size, -eps, -size)
        glVertex3f(size, -eps, -size)
        glVertex3f(size, -eps, size)
        glVertex3f(-size, -eps, size)
        glEnd()

        # Draw the wireframe edges
        glColor3f(0.0, 0.0, 0.0)  # draw edges in black
        glLineWidth(1.0)

        x = -size
        while x <= size:
            glBegin(GL_LINE_LOOP)
            glVertex3f(x, 0, -size)  # Top Right Of The Quad (Top)
            glVertex3f(x, 0, size)  # Top Left Of The Quad (Top)
            glEnd()  # Done Drawing The Quad
            x += 1.0
        z = -size
        while z <= size:
            glBegin(GL_LINE_LOOP)
            glVertex3f(-size, 0, z)  # Top Right Of The Quad (Top)
            glVertex3f(size, 0, z)  # Top Left Of The Quad (Top)
            glEnd()  # Done Drawing The Quad
            z += 1.0

    def inner_loop(self):
        self.step()
        self.render()

    @staticmethod
    def init_gl(width, height):
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()  # Reset The Projection Matrix
        gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    @staticmethod
    def resize_gl_scene(width, height):
        if height == 0:  # Prevent A Divide By Zero If The Window Is Too Small
            height = 1
        glViewport(0, 0, width, height)  # Reset The Current Viewport And Perspective Transformation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width) / float(height), 0.1,
                       100.0)  # 45 deg horizontal field of view, aspect ratio, near, far
        glMatrixMode(GL_MODELVIEW)


########################################################################################################################

class FallingLinkEnv(BaseEnv):

    def reset(self):
        self.reset_sim(1)  # reset with a given number of links

    def step(self):
        if not self.sim_running:  # is simulation stopped?
            return
        if self.sim_time == 0:
            link = self.links[0]
            link.vel = np.array([0.1, 2.0, 0.0])  # initial velocity

        grav = np.array([0, self.GRAVITY, 0])  # gravity vector

        # pre-allocate matrix of the required size
        # TODO:  choose dimensions as needed 

        n_links = len(self.links)
        dim = 6 * n_links
        A = np.eye((dim))  # main matrix
        b = np.zeros(dim)  # right-hand-side

        psteps = 1  # number of physics time steps to take for each display time step
        for step in range(psteps):  # simulate physics time steps, before doing a draw update
            # The commented code below are for your convenience, in case you need to understand variables in each link.

            # key attributes  (see link.py)
            # link = self.links[u]
            #        link.color = [0, 0, 0]
            #        link.size = [1, 1, 1]
            #        link.mass = 1.0
            #        link.inertia = np.identity(3)   # inertia tensor, in the local frame
            #        link.q_rot = Quaternion()
            #        link.q_rot.rotation_matrix      #  get rotation matrix R
            #        link.omega = np.array([0.0, 0.0, 0.0])
            #        link.pos = np.array([0.0, 0.0, 0.0])
            #        link.vel = np.array([0.0, 0.0, 0.0])
            #        link.display_force = np.array([0.0, 0.0, 0.0])
            #        link.get_r()    # r = COM to parent hinge vector, in world coords
            # np.identity()    build 3x3 matrix
            # np.cross(a,b)    compute a cross b
            # funkify(a)       compute a tilde matrix, to compute a cross-product
            # A[2:6, 14:18] = np.identity(4)     fill in the 4x4 block in matrix, beginning at A[2,14], with a 4x4 identity matrix
            # D = E @ F        matrix-matrix or matrix-vector multiply

            # TODO:  Equations of motion solve the linear equations   A x = b
            #        You'll need to build A and b.
            #        Most quantities are already in the world frame, except for the inertia tensor

            # solve for accelerations and constraint forces
            results = np.linalg.solve(A, b)

            # TODO:  integrate the equations of motion, using explicit Euler integration
            #        i.e., update:  link.pos,  link.vel,  link.q_rot,  link.omega

            link_accs = results.reshape(n_links, 6)
            for link, link_acc in zip(self.links, link_accs):
                pass

            self.sim_time += self.dT


########################################################################################################################

class SpinningLinkEnv(BaseEnv):

    def reset(self):
        self.reset_sim(1)  # reset with a given number of links

    def step(self):
        pass


########################################################################################################################

class SingleLinkPendulumEnv(BaseEnv):

    def reset(self):
        self.reset_sim(1)  # reset with a given number of links

    def step(self):
        pass


########################################################################################################################

class MultiLinkPendulumEnv(BaseEnv):

    def reset(self):
        self.reset_sim(2)  # reset with a given number of links

    def step(self):
        pass
