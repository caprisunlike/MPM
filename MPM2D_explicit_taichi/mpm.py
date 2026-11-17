import taichi as ti
import numpy as np
from grid import Grid
from particles import Particles
from simulation import Simulation
from boundary import Boundary
import imageio

ti.init(arch=ti.gpu)
USE_GGUI = True
RECORD = False
VIDEO_FPS = 50

class MPM:
    def __init__(self):
        self.gridNum = 128
        self.particlesNum = 10000
        self.boundaryNum = 4
        self.dx = 1.0 / self.gridNum
        self.inv_dx = 1.0 / self.dx
        self.t = 0.0
        self.dt = 5e-5
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
    lo, hi = mpm.b_min, mpm.b_max
    if USE_GGUI:
        win = ti.ui.Window("MPM Simulation", res=(n, n))
        canvas = win.get_canvas()
        g_pos = ti.Vector.field(2, dtype=ti.f32, shape=mpm.particlesNum)
        box = ti.Vector.field(2, dtype=ti.f32, shape=8)
        box.from_numpy(np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi], [lo, lo], [lo, hi], [hi, hi], [hi, lo]], dtype=np.float32))
    else:
        win = ti.GUI("MPM Simulation", res=(n, n))

    writer = imageio.get_writer("mpm_taichi.mp4", format="FFMPEG", fps=VIDEO_FPS) if RECORD else None

    render_interval = 50
    steps = 20000
    for s in range(steps):
        sim.simulation()
        #mpm.t += mpm.dt

        if s % render_interval == 0:
            #pos = particles.x.to_numpy()   # 파티클 기준 좌표
            pos = particles.x.to_numpy() * (hi - lo) + lo   # 화면 기준 좌표

            if USE_GGUI:   # Vulkan 기반 GPU 렌더링
                g_pos.from_numpy(pos)
                canvas.set_background_color((0, 0, 0))
                canvas.lines(box, width=0.003, color=(1.0, 1.0, 1.0))
                canvas.circles(g_pos, radius=particles.r/n, color=(0.4, 0.8, 1.0))
            else:   # CPU 기반 렌더링
                win.rect(topleft=(lo, lo), bottomright=(hi, hi), color=0xFFFFFF)
                win.circles(pos, radius=particles.r, color=0x66ccff)

            if RECORD:   # 영상 녹화
                img = win.get_image_buffer_as_numpy() if USE_GGUI else win.get_image()
                img = np.rot90(img[..., :3])
                writer.append_data((img*255).astype(np.uint8))

            win.show()

    if RECORD and writer:
        writer.close()
        print("Video saved.")

    print("Simulation finished.")

if __name__ == "__main__":
    main()