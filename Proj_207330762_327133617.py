import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

class RobotArm:
    def __init__(self):
        # Robot arm parameters
        self.l1 = 0.2
        self.l2 = 1.5
        self.l3 = 1

        # Obstacle parameters
        self.obs_center = [0, 1.5, 0.1]
        self.obs_radius = 0.2

        # Joint limits
        self.lim_min = np.array([-(np.pi/2), -np.pi, -np.pi])
        self.lim_max = np.array([np.pi/2, np.pi, np.pi])

    def get_rand_conf(self):
        """Generate random joint angles."""
        rnd = np.random.random(3)
        rnd_thetas = rnd * (self.lim_max - self.lim_min) + self.lim_min
        return rnd_thetas

    def direct_kinematics(self, q):
        """Compute end-effector position from joint angles."""
        xi = np.cos(q[0]) * (self.l3 * np.cos(q[1] + q[2]) + self.l2 * np.cos(q[1]))
        yi = np.sin(q[0]) * (self.l3 * np.cos(q[1] + q[2]) + self.l2 * np.cos(q[1]))
        zi = self.l3 * np.sin(q[1] + q[2]) + self.l2 * np.sin(q[1]) + self.l1
        return np.array([xi, yi, zi])

    def collision_checker(self, p):
        """Check if the end-effector position collides with the obstacle."""
        obs_distance = np.linalg.norm(p - self.obs_center) - self.obs_radius
        distance = obs_distance
        xy_rad = np.sqrt(p[0]**2 + p[1]**2)
        if (obs_distance < 0.05) or (xy_rad > 1.995) or (xy_rad < 1.005) or (p[1] < 0) or (p[2] < 0.05):
            distance = 0
        return distance

    def QX_free(self):
        """Generate all possible configurations and filter out those that collide with obstacles."""
        X_free = []
        Q_free = []
        D_free = []

        while len(X_free) < 5000:
            q = self.get_rand_conf()
            p = self.direct_kinematics(q)
            distance = self.collision_checker(p)
            if distance > 0:
                X_free.append(p)
                Q_free.append(q)
                D_free.append(distance)

        X_free = np.array(X_free)
        Q_free = np.array(Q_free)
        D_free = np.array(D_free)

        # Plot X_free
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(X_free[:, 0], X_free[:, 1], X_free[:, 2], s=1)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.title("Free Space Plot")
        plt.show()

        return Q_free, X_free, D_free

    def inverse_kinematics(self, p):
        """Compute joint angles for a given end-effector position."""
        q = np.zeros(3)
        D = (p[0]**2 + p[1]**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        q1_1 = np.arctan2(p[1], p[0])
        q1_2 = np.arctan2(-p[1], -p[0])
        q[0] = q1_1 if (q1_1 < 0) else q1_2
        q3_1 = np.arctan2(np.sqrt(1 - D**2), D)
        q3_2 = np.arctan2(-np.sqrt(1 - D**2), D)
        q[2] = q3_1 if (q3_1 < 0) else q3_2
        q[1] = np.arctan2(p[1], p[0]) - np.arctan2(self.l3 * np.sin(q[2]), self.l2 + self.l3 * np.cos(q[2]))
        return q

    def max_dist(self):
        """Compute the path to the target point."""
        x_product = [-1.5, 0.3, 0.1]
        q_product = -self.inverse_kinematics(x_product)

        if (x_product[0] - 1.5) > 0:
            q_1 = [q_product[0]]
            q_2 = [-q_product[1]]
            q_3 = [-q_product[2]]
        elif x_product[0] == 1.5:
            q_1 = [-q_product[0]]
            q_2 = [-q_product[1]]
            q_3 = [-q_product[2]]
        else:
            q_1 = [q_product[0] + np.pi]
            q_2 = [q_product[1]]
            q_3 = [q_product[2]]

        pos_x = [x_product[0]]
        pos_y = [x_product[1]]
        pos_z = [x_product[2]]

        if (x_product[0] - 1.5) > 0:
            factor = (x_product[0] - 1.5) / 5 if (x_product[0] - 1.5) < 0.75 else (x_product[0] - 1.5) / 20
            step = 5 if (x_product[0] - 1.5) < 0.75 else 20

            for i in range(step):
                dist = []
                index = []
                for j in range(len(X_free)):
                    if 1.5 + (step - 1 - i) * factor < X_free[j][0] < 1.5 + (step - i) * factor:
                        dist.append(D_free[j])
                        index.append(j)
                if dist:
                    max_dist_index = dist.index(max(dist))
                    pos_x.append(X_free[index[max_dist_index]][0])
                    pos_y.append(X_free[index[max_dist_index]][1])
                    pos_z.append(X_free[index[max_dist_index]][2])
                    q_1.append(Q_free[index[max_dist_index]][0])
                    q_2.append(Q_free[index[max_dist_index]][1])
                    q_3.append(Q_free[index[max_dist_index]][2])

            pos_x.append(1.5)
            pos_y.append(0)
            pos_z.append(0.1)
            q_start = self.inverse_kinematics([1.5, 0, 0.1])
            q_1.append(q_start[0])
            q_2.append(q_start[1])
            q_3.append(q_start[2])

        elif (x_product[0] - 1.5) < 0:
            factor = (1.5 - x_product[0]) / 5 if (1.5 - x_product[0]) < 0.75 else (1.5 - x_product[0]) / 20
            step = 5 if (1.5 - x_product[0]) < 0.75 else 20

            for i in range(step):
                dist = []
                index = []
                for j in range(len(X_free)):
                    if 1.5 - (step - 1 - i) * factor > X_free[j][0] > 1.5 - (step - i) * factor:
                        dist.append(D_free[j])
                        index.append(j)
                if dist:
                    max_dist_index = dist.index(max(dist))
                    pos_x.append(X_free[index[max_dist_index]][0])
                    pos_y.append(X_free[index[max_dist_index]][1])
                    pos_z.append(X_free[index[max_dist_index]][2])
                    q_1.append(Q_free[index[max_dist_index]][0])
                    q_2.append(Q_free[index[max_dist_index]][1])
                    q_3.append(Q_free[index[max_dist_index]][2])

            pos_x.append(1.5)
            pos_y.append(0)
            pos_z.append(0.1)
            q_start = self.inverse_kinematics([1.5, 0, 0.1])
            q_1.append(q_start[0])
            q_2.append(q_start[1])
            q_3.append(q_start[2])

        elif (x_product[0] - 1.5) == 0:
            step = 0
            pos_x.append(1.5)
            pos_y.append(0)
            pos_z.append(0.1)
            q_start = self.inverse_kinematics([1.5, 0, 0.1])
            q_1.append(q_start[0])
            q_2.append(q_start[1])
            q_3.append(q_start[2])

        Q = np.column_stack([q_1, q_2, q_3])
        return np.array(pos_x), np.array(pos_y), np.array(pos_z), Q, np.array(D_free), Q

    def animate_arm(self):
        """Animate the robot arm motion."""
        X, Y, Z, Q_track, _, _ = self.max_dist()

        # Create figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 2.5])
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title("3D Animation of Robot Arm")

        # Initialize the line and point
        line, = ax.plot([], [], [], lw=2, color='blue')
        point, = ax.plot([], [], [], 'ok')

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point

        def animate(i):
            # Update line data
            line.set_data(X[:i], Y[:i])
            line.set_3d_properties(Z[:i])

            # Update point data
            point.set_data(X[i], Y[i])
            point.set_3d_properties(Z[i])

            return line, point

        # Create animation
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(X), interval=500, blit=True)

        # Save animation as GIF
        writer = PillowWriter(fps=2)
        anim.save('robot_arm_animation.gif', writer=writer)

        plt.show()

# Create and use the RobotArm instance
robot_arm = RobotArm()
robot_arm.animate_arm()
