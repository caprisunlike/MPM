import time
import struct
import numpy as np
import warp as wp
from pathlib import Path
from multiprocessing.connection import Client

from grid import Grid
from particles import Particles
from simulation import Simulation
from boundary import Boundary

wp.init()

class MPM:
    def __init__(self):
        self.gridNum = 500
        self.particlesNum = 10000
        self.boundaryNum = 4
        self.dx = 1.0 / self.gridNum
        self.inv_dx = 1.0 / self.dx
        self.dt = 1e-4
        self.dim = 2
        self.gravity = wp.vec2(0.0, -9.8)
        self.b_min = 0.05
        self.b_max = 0.95

PIPE_NAME = r"\\.\pipe\mpm_warp_pipe"

def connect_pipe():
    # Omniverse receiver가 named pipe를 열 때까지 기다림
    while True:
        try:
            return Client(PIPE_NAME, family="AF_PIPE")
        except OSError:
            time.sleep(0.1)

def main():
    mpm = MPM()
    boundary = Boundary(mpm)
    grid = Grid(mpm)
    particles = Particles(mpm)
    sim = Simulation(mpm, particles, grid, boundary)

    conn = connect_pipe()

    steps = 20000
    send_every = 4

    try:
        for frame in range(steps):
            sim.simulation()

            if frame % send_every == 0:
                pos = particles.x.numpy().astype(np.float32)  # (N,2)
                z = np.zeros((pos.shape[0], 1), dtype=np.float32)
                p3 = np.concatenate([pos, z], axis=1)         # (N,3)

                conn.send((frame, p3))
        conn.send(("END", None))

    finally:
        try:
            conn.close()
        except Exception:
            pass
        print("MPM sender finished.")

if __name__ == "__main__":
    main()