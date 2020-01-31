import numpy as np
from OpenGL.GL import glMultMatrixf, glScale, glBegin, glVertex3fv, glEnd
from OpenGL.raw.GL.VERSION.GL_1_0 import glPushMatrix, glTranslatef, glColor3f, glPopMatrix, glScalef, glVertex3f, \
    glLineWidth
from OpenGL.raw.GL.VERSION.GL_1_1 import GL_LINES, GL_LINE_LOOP
from OpenGL.raw.GL.VERSION.GL_4_0 import GL_QUADS
from pyquaternion import Quaternion


class Link(object):
    def __init__(self):
        self.color = [0, 0, 0]
        self.size = [1, 1, 1]
        self.mass = 1.0
        self.inertia = np.identity(3)
        self.q_rot = Quaternion()
        self.omega = np.array([0.0, 0.0, 0.0])
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])

        self.display_force = np.array([0.0, 0.0, 0.0])

    def draw(self):
        """Render the link with OpenGL"""
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])      # move to location
        glMultMatrixf(self.q_rot.transformation_matrix.T)        # then rotate to desired orientation
        glScale(self.size[0], self.size[1], self.size[2])        # scale to the desired size
        glColor3f(self.color[0], self.color[1], self.color[2])   # set the drawing color
        self.draw_cube()
        glPopMatrix()
        glBegin(GL_LINES)
        glVertex3fv(self.pos + self.get_r())
        glVertex3fv(self.pos + self.get_r() + self.display_force)
        glEnd()

    def set_cuboid(self, mass, w, h, d):
        """Initializes link to a cuboid of the specified mass width, depth and height (x, y, z respectively)"""
        self.mass = mass
        self.inertia = mass/12 * np.array([[h**2+d**2, 0, 0], [0, w**2+d**2, 0], [0, 0, w**2+h**2]])
        self.size = np.array([w, h, d])

    def get_r(self):
        """Return the world-space vector from the link's center to its upper hinge joint"""
        # return (self.size[1]/2)*np.array([-np.sin(self.theta), np.cos(self.theta), 0])
        return self.q_rot.rotation_matrix @ np.array([0, self.size[1]/2, 0])   ##  r_world = R * r_local

    @staticmethod
    def draw_cube():
        glScalef(0.5, 0.5, 0.5)  # dimensions below are for a 2x2x2 cube, so scale it down by a half first
        glBegin(GL_QUADS)  # Start Drawing The Cube

        glVertex3f(1.0, 1.0, -1.0)  # Top Right Of The Quad (Top)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Left Of The Quad (Top)
        glVertex3f(-1.0, 1.0, 1.0)  # Bottom Left Of The Quad (Top)
        glVertex3f(1.0, 1.0, 1.0)  # Bottom Right Of The Quad (Top)

        glVertex3f(1.0, -1.0, 1.0)  # Top Right Of The Quad (Bottom)
        glVertex3f(-1.0, -1.0, 1.0)  # Top Left Of The Quad (Bottom)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Bottom)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Bottom)

        glVertex3f(1.0, 1.0, 1.0)  # Top Right Of The Quad (Front)
        glVertex3f(-1.0, 1.0, 1.0)  # Top Left Of The Quad (Front)
        glVertex3f(-1.0, -1.0, 1.0)  # Bottom Left Of The Quad (Front)
        glVertex3f(1.0, -1.0, 1.0)  # Bottom Right Of The Quad (Front)

        glVertex3f(1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Back)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Back)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Right Of The Quad (Back)
        glVertex3f(1.0, 1.0, -1.0)  # Top Left Of The Quad (Back)

        glVertex3f(-1.0, 1.0, 1.0)  # Top Right Of The Quad (Left)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Left Of The Quad (Left)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Left)
        glVertex3f(-1.0, -1.0, 1.0)  # Bottom Right Of The Quad (Left)

        glVertex3f(1.0, 1.0, -1.0)  # Top Right Of The Quad (Right)
        glVertex3f(1.0, 1.0, 1.0)  # Top Left Of The Quad (Right)
        glVertex3f(1.0, -1.0, 1.0)  # Bottom Left Of The Quad (Right)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Right)
        glEnd()  # Done Drawing The Quad

        # Draw the wireframe edges
        glColor3f(0.0, 0.0, 0.0)         # draw edges in black
        glLineWidth(1.0)

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, 1.0, -1.0)  # Top Right Of The Quad (Top)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Left Of The Quad (Top)
        glVertex3f(-1.0, 1.0, 1.0)  # Bottom Left Of The Quad (Top)
        glVertex3f(1.0, 1.0, 1.0)  # Bottom Right Of The Quad (Top)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, -1.0, 1.0)  # Top Right Of The Quad (Bottom)
        glVertex3f(-1.0, -1.0, 1.0)  # Top Left Of The Quad (Bottom)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Bottom)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Bottom)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, 1.0, 1.0)  # Top Right Of The Quad (Front)
        glVertex3f(-1.0, 1.0, 1.0)  # Top Left Of The Quad (Front)
        glVertex3f(-1.0, -1.0, 1.0)  # Bottom Left Of The Quad (Front)
        glVertex3f(1.0, -1.0, 1.0)  # Bottom Right Of The Quad (Front)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Back)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Back)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Right Of The Quad (Back)
        glVertex3f(1.0, 1.0, -1.0)  # Top Left Of The Quad (Back)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(-1.0, 1.0, 1.0)  # Top Right Of The Quad (Left)
        glVertex3f(-1.0, 1.0, -1.0)  # Top Left Of The Quad (Left)
        glVertex3f(-1.0, -1.0, -1.0)  # Bottom Left Of The Quad (Left)
        glVertex3f(-1.0, -1.0, 1.0)  # Bottom Right Of The Quad (Left)
        glEnd()  # Done Drawing The Quad

        glBegin(GL_LINE_LOOP)
        glVertex3f(1.0, 1.0, -1.0)  # Top Right Of The Quad (Right)
        glVertex3f(1.0, 1.0, 1.0)  # Top Left Of The Quad (Right)
        glVertex3f(1.0, -1.0, 1.0)  # Bottom Left Of The Quad (Right)
        glVertex3f(1.0, -1.0, -1.0)  # Bottom Right Of The Quad (Right)
        glEnd()  # Done Drawing The Quad
