import socket
import time
import struct
import numpy as np
import warp as wp
from pathlib import Path

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

MAGIC = 0x4D504D31  # 'MPM1'
MSG_DATA = 1
MSG_END  = 2

def send_end(sock, addr, frame):
    # END packet header: (magic, msg_type, frame, 0)
    pkt = struct.pack("<IIII", MAGIC, MSG_END, frame, 0)
    sock.sendto(pkt, addr)

def main():
    mpm = MPM()
    boundary = Boundary(mpm)
    grid = Grid(mpm)
    particles = Particles(mpm)
    sim = Simulation(mpm, particles, grid, boundary)

    HOST, PORT = "127.0.0.1", 5005
    addr = (HOST, PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    steps = 20000
    send_every = 1
    CHUNK_POINTS = 4000

    for frame in range(steps):
        sim.simulation()

        if frame % send_every == 0:
            pos = particles.x.numpy().astype(np.float32)  # (N,2)
            z = np.zeros((pos.shape[0], 1), dtype=np.float32)
            p3 = np.concatenate([pos, z], axis=1)         # (N,3)

            count = p3.shape[0]
            for start in range(0, count, CHUNK_POINTS):
                end = min(count, start + CHUNK_POINTS)
                chunk = p3[start:end].tobytes(order="C")
                # DATA header: (magic, msg_type, frame, start, chunk_count)
                hdr = struct.pack("<IIIII", MAGIC, MSG_DATA, frame, start, (end - start))
                sock.sendto(hdr + chunk, addr)

    # 종료 신호 보내기
    for _ in range(10):
        send_end(sock, addr, steps)
        time.sleep(0.01)
    sock.close()
    print("Sender finished and closed.")

    # sender 끝부분(소켓 닫고 print한 다음)
    Path("C:/Users/user/Documents/graphicslab/MPM/MPM2D_explicit_warp/.sender_running.lock").unlink(missing_ok=True)

if __name__ == "__main__":
    main()