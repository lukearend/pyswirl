import sys
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import matplotlib.animation as animation
matplotlib.use('MacOSX')


cart_width = 0.50   # m
cart_height = 0.30  # m
bob_radius = 0.15   # m
track_len = 4.0     # m

G = 9.8
M = 0.5
m = 0.1
L = 1.0

class PendulumCartSystem:
    def __init__(self, init_state=[0, 0, 0, 0]):
        self.state = np.array(init_state)
        self.bounced = False
        self.time_elapsed = 0

    def derivs(self, X, f):
        x, th, v, w = X
        dxdt = np.zeros_like(X)
        dxdt[0] = v
        dxdt[1] = w
        dxdt[2] = (f + (m*L*w*np.sin(th) - m*G*np.sin(th)*np.cos(th))
                   / (M + m - m*(np.cos(th))**2))
        dxdt[3] = (-f*np.cos(th) + ((M + m)*G*np.sin(th) - m*L*(w**2)*np.sin(th)*np.cos(th))
                   / L*(M + m - m*(np.cos(th)**2)))
        return dxdt

    def step(self, dt, f):
        # RK4 update method
        k1 = self.derivs(self.state, f)
        k2 = self.derivs(self.state + dt/2*k1, f)
        k3 = self.derivs(self.state + dt/2*k2, f)
        k4 = self.derivs(self.state + dt*k3, f)
        delta = dt/6*(k1 + 2*k2 + 2*k3 + k4)

        # detect bounce
        new_x = self.state[0] + delta[0]
        new_left_edge = new_x - cart_width/2
        new_right_edge = new_x + cart_width/2
        if new_left_edge < -track_len/2:
            overshoot = -(new_left_edge + track_len/2)
            new_x = -track_len/2 + cart_width/2 + overshoot
            self.bounced = True
        elif new_right_edge > track_len/2:
            overshoot = new_right_edge - track_len/2
            new_x = track_len/2 - cart_width/2 - overshoot
            self.bounced = True
        else:
            self.bounced = False

        self.state += delta
        if self.bounced:
            self.state[0] = new_x
            self.state[2] = -self.state[2]

        self.time_elapsed += dt


num_substates = 10
num_actions = 5
v_state_max = 1.0
w_state_max = np.pi
f_max = 20.0

def discretize_state(X):
    x, th, v, w = X

    s_x = math.floor((x + (track_len - cart_width)/2)/(track_len - cart_width)*num_substates)
    s_x = np.clip(s_x, 0, num_substates - 1)

    s_th = math.floor((th % (2*np.pi))/(2*np.pi)*num_substates)
    s_th = np.clip(s_th, 0, num_substates - 1)

    s_v = math.floor((v + v_state_max)/(2*v_state_max)*num_substates)
    s_v = np.clip(s_v, 0, num_substates - 1)

    s_w = math.floor((w + w_state_max)/(2*w_state_max)*num_substates)
    s_w = np.clip(s_w, 0, num_substates - 1)

    s = s_x + num_substates*s_th + (num_substates**2)*s_v + (num_substates**3)*s_w
    return s

def select_action(mu, s):
    mask = np.random.uniform(0, 1) <= np.cumsum(mu[s, :])
    action = np.nonzero(mask)[0][0]
    return action

def get_force(a):
    return np.linspace(-f_max, f_max, num_actions)[a]

def get_reward(X):
    x, th, v, w = X
    r_th = (np.cos(th) - 1) / 2
    r_x = -abs(x/(track_len/2 - cart_width/2))
    return r_th + r_x


system = PendulumCartSystem([0, np.radians(np.random.randn()), 0, 0])
dt = 1.0/30  # 30 fps

gamma = 0.9
epsilon = 0.05
alpha = 0.5

global episode, timestep, s
episode = 0
timestep = 0
s = discretize_state(system.state)

mu = np.ones((num_substates**4, num_actions)) / num_actions
Q1 = np.zeros((num_substates**4, num_actions))
Q2 = np.zeros((num_substates**4, num_actions))

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-3, 3), ylim=(-1.5, 1.5))
ax.plot([-track_len/2, track_len/2], [0, 0],
         color='k', linewidth=2, zorder=0)
ax.plot([-track_len/2, -track_len/2], [-0.05, 0.05],
         color='k', linewidth=2, zorder=0)
ax.plot([track_len/2, track_len/2], [-0.05, 0.05],
         color='k', linewidth=2, zorder=0)

rod, = ax.plot([], [], color='#a9a9a9', linewidth=2, zorder=1)
cart = plt.Rectangle((0, 0), cart_width, cart_height,
                     edgecolor='#ff2600', facecolor='#ff7e79',
                     linewidth=2, zorder=2)
bob = plt.Circle((0, 0), bob_radius,
                 edgecolor='#0433ff', facecolor='#7a80ff',
                 linewidth=2,zorder=2)
arrow, = ax.plot([], [], color='#fca503')
episode_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
timestep_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
time_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
x_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
th_text = ax.text(0.02, 0.75, '', transform=ax.transAxes)
v_text = ax.text(0.02, 0.70, '', transform=ax.transAxes)
w_text = ax.text(0.02, 0.65, '', transform=ax.transAxes)
s_text = ax.text(0.02, 0.60, '', transform=ax.transAxes)
reward_text = ax.text(0.02, 0.55, '', transform=ax.transAxes)
q_text = ax.text(0.02, 0.50, '', transform=ax.transAxes)

def init():
    ax.add_patch(cart)
    ax.add_patch(bob)
    return (rod, cart, bob, arrow,
            episode_text, timestep_text, time_text,
            x_text, th_text, v_text, w_text,
            s_text, reward_text, q_text)

def animate(i):
    global episode, timestep, s

    # select an action
    Q = (Q1[s, :] + Q2[s, :])/2
    max_inds = np.where(Q == np.max(Q))[0]
    mu[s, :] = epsilon/num_actions
    mu[s, max_inds] = (1 - epsilon)/len(max_inds) + epsilon/num_actions
    a = select_action(mu, s)
    force = get_force(a)

    # advance simulation and observe results
    system.step(dt, force)
    reward = get_reward(system.state)
    s_prime = discretize_state(system.state)

    # perform a learning step
    coin_flip = np.random.uniform(0, 1) <= 0.5;
    if system.bounced:
        Q_prime = 0
    elif coin_flip:
        max_ind = np.random.choice(np.where(Q1[s_prime, :] == np.max(Q1[s_prime, :]))[0])
        Q_prime = Q2[s_prime, max_ind]
    else:
        max_ind = np.random.choice(np.where(Q2[s_prime, :] == np.max(Q2[s_prime, :]))[0])
        Q_prime = Q1[s_prime, max_ind]

    if coin_flip:
        delta = reward + gamma*Q_prime - Q1[s, a]
        Q1[s, a] = Q1[s, a] + alpha*delta
    else:
        delta = reward + gamma*Q_prime - Q2[s, a]
        Q2[s, a] = Q2[s, a] + alpha*delta
    s = s_prime
    timestep += 1

    if system.bounced:
        system.state = [0, np.radians(np.random.randn()), 0, 0]
        system.time_elapsed = 0
        episode += 1
        timestep = 0
        s = discretize_state(system.state)

    # update graphical elements
    cart_pos = system.state[0]
    rod_angle = system.state[1]
    bob_x = cart_pos + np.sin(rod_angle)
    bob_y = np.cos(rod_angle)

    rod.set_data([cart_pos, bob_x], [0, bob_y])
    cart.set_xy([cart_pos - cart_width/2, 0 - cart_height/2])
    bob.center = bob_x, bob_y
    arrow.set_data([cart_pos, cart_pos + force / 100], [0, 0])
    episode_text.set_text("episode = {}".format(episode))
    timestep_text.set_text("timestep = {}".format(timestep))
    time_text.set_text("time = {:.2f}".format(system.time_elapsed))
    x_text.set_text("x = {:.2f}".format(system.state[0]))
    th_text.set_text("th = {:.2f}".format(system.state[1]))
    v_text.set_text("v = {:.2f}".format(system.state[2]))
    w_text.set_text("w = {:.2f}".format(system.state[3]))
    s_text.set_text("s = {}".format(s))
    reward_text.set_text("r = {:.2f}".format(reward))
    q_text.set_text("q = {:2f}".format(Q[a]))

    return (rod, cart, bob, arrow,
            episode_text, timestep_text, time_text,
            x_text, th_text, v_text, w_text,
            s_text, reward_text, q_text)


def main():
    ani = animation.FuncAnimation(fig, animate, frames=None,
                                  interval=1000*dt, blit=True, init_func=init)
    plt.show()


if __name__ == '__main__':
    main()
