import taichi as ti
import numpy as np
from grid import Grid
from particles import Particles
from simulation import Simulation
from boundary import Boundary


ti.init(arch=ti.gpu)

class MPM:
    def __init__(self):
        self.gridNum = 500
        self.particlesNum = 10000
        self.boundaryNum = 4
        self.dx = 1.0 / self.gridNum
        self.inv_dx = 1.0 / self.dx
        self.t = 0.0
        self.dt = 0.001
        self.dim = 2

        self.gravity = ti.Vector([0.0, -9.8])

        self.b_min = 0.05   # top/left boundary
        self.b_max = 0.95   # bottom/right boundary

    @ti.func
    def pos_to_idx(self, x):
        return (x * self.inv_dx - 0.5).cast(int)
    


def main():
    mpm = MPM()

    boundary = Boundary(mpm)
    grid = Grid(mpm)
    particles = Particles(mpm)
    sim = Simulation(mpm, particles, grid, boundary)

    n = 512
    gui = ti.GUI("MPM Simulation", res=(n, n))

    for mpm.t in range(5000):
        sim.simulation()
        mpm.t += mpm.dt

        pos = particles.x.to_numpy()   # 파티클 기준 좌표
        g_pos = pos * (mpm.b_max - mpm.b_min) + mpm.b_min   # gui 기준 좌표

        gui.rect(topleft=(mpm.b_min, mpm.b_min), bottomright=(mpm.b_max, mpm.b_max), color=0xFFFFFF)
        gui.circles(g_pos, radius=particles.r, color=0x66ccff)
        gui.show()
    
    print("Simulation finished.")

if __name__ == "__main__":
    main()