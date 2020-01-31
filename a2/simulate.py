import argparse

from OpenGL.GLUT import *

from envs import BaseEnv, FallingLinkEnv, SpinningLinkEnv, SingleLinkPendulumEnv, MultiLinkPendulumEnv


def main(environment):
    if environment == 'base':
        env = BaseEnv()
    elif environment == 'fallinglink':
        env = FallingLinkEnv()
    elif environment == 'spinninglink':
        env = SpinningLinkEnv()
    elif environment == 'singlelinkpendulum':
        env = SingleLinkPendulumEnv()
    elif environment == 'multilinkpendulum':
        env = MultiLinkPendulumEnv()
    else:
        raise ValueError("Environment {} not available.".format(environment))

    # initialize the simulation
    env.reset()

    # event processing loop
    glutMainLoop()

    # NOTE: calling glutMainLoop() is very much the same thing as writing this while loop:
    #
    # while True:
    #   env.inner_loop()
    #
    # In our case, env.inner_loop() is calling env.step() and env.render(), hence simulating the world forward and
    # displaying the resulting state of the world.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default='base',
                        help="Choose between different environments. Available choices: ['base', 'fallinglink', "
                             "'spinninglink', 'singlelinkpendulum', 'multilinkpendulum']")
    args = parser.parse_args()
    print("Hit ESC/q to quit, r to reset, + and - to add or remove links (resetting the simulation).")
    main(args.environment)
