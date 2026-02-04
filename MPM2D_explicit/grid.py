from math import dist
import taichi as ti

@ti.data_oriented
class Grid:
    def __init__(self, mpm: ti.template()):
        self.mpm = mpm
        n = mpm.gridNum
        dim = mpm.dim
        
        self.pos = ti.Vector.field(n=dim, dtype=ti.f32, shape=(n, n))
        self.mass = ti.field(dtype=ti.f32, shape=(n, n))
        self.vel = ti.Vector.field(n=dim, dtype=ti.f32, shape=(n, n))
        self.force = ti.Vector.field(n=dim, dtype=ti.f32, shape=(n, n))

        self.init_grid_position()
        self.initialize()

    @ti.kernel
    def initialize(self):
        for i, j in self.mass:
            self.mass[i, j] = 0.0
            self.vel[i, j] = ti.Vector([0.0, 0.0])
            self.force[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def init_grid_position(self):
        for i, j in self.pos:
            self.pos[i, j] = ti.Vector([i, j]) * self.mpm.dx   # 파티클 기준 좌표

    @ti.func
    def compute_weights(self, fx):
        return self.kernel(fx.x) * self.kernel(fx.y)
    
    @ti.func
    def compute_gradient_weights(self, fx):
        return ti.Vector([self.mpm.inv_dx*self.d_kernel(fx.x)*self.kernel(fx.y), self.kernel(fx.x)*self.mpm.inv_dx*self.d_kernel(fx.y)])

    @ti.func
    def kernel(self, x):   # quadratic B-spline kernel
        result = 0.0
        if abs(x) < 0.5:
            result = 0.75 - x**2
        elif 0.5 <= abs(x) < 1.5:
            result = 0.5 * (1.5 - abs(x))**2
        else:
            result = 0.0
        return result
        
    @ti.func
    def d_kernel(self, x):   # derivative of quadratic B-spline kernel
        result = 0.0
        if abs(x) < 0.5:
            result = -2.0 * x
        elif 0.5 <= abs(x) < 1.5:
            if x > 0:
                result = -1.0 * (1.5 - abs(x))
            else:
                result = 1.0 * (1.5 - abs(x))
        else:
            result = 0.0
        return result