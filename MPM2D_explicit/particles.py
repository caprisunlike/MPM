import taichi as ti
#import numpy as np

@ti.data_oriented
class Particles:
    def __init__(self, mpm: ti.template()):
        self.mpm = mpm
        n = mpm.particlesNum
        dim = mpm.dim

        self.m = 0.01   # mass
        self.r = 2.0   # radius
        self.x = ti.Vector.field(n=dim, dtype=ti.f32, shape=n)   # position
        self.v = ti.Vector.field(n=dim, dtype=ti.f32, shape=n)   # velocity
        self.a = ti.Vector.field(n=dim, dtype=ti.f32, shape=n)   # acceleration
        self.vol = ti.field(dtype=ti.f32, shape=n)   # volume
        self.F = ti.Matrix.field(n=dim, m=dim, dtype=ti.f32, shape=n)   # deformation gradient
        self.B = ti.Matrix.field(n=dim, m=dim, dtype=ti.f32, shape=n)   # affine velocity field

        v = 0.25 #* (self.mpm.dx * self.mpm.dx)
        #self.D = self.calculate_Dp()
        self.D = ti.Matrix([[v, 0.0],
                            [0.0, v]])
        self.D_inv = self.D.inverse()

        self.E = 5000.0   # Young's modulus
        self.nu = 0.2   # Poisson's ratio
        self.mu = self.E / (2 * (1 + self.nu))   # Lame coefficient
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))   # Lame coefficient
        
        self.initialize()

    @ti.kernel
    def initialize(self):
        n = self.mpm.particlesNum
        dim = ti.static(self.mpm.dim)
        for i in range(n):
            self.x[i] = [(ti.random() * 0.8 + 0.1), (ti.random() * 0.8 + 0.1)]
            self.v[i] = ti.Vector([0.9, 0.0])
            self.a[i] = ti.Vector([0.0, 0.0])
            self.vol[i] = self.mpm.dx * self.mpm.dx   # 초기 부피
            self.F[i] = ti.Matrix.identity(ti.f32, dim)
            self.B[i] = ti.Matrix.zero(ti.f32, dim, dim)

    #def calculate_Dp(self):
        #return 0.25 * (self.mpm.dx)**2 * np.eye(self.mpm.dim)   # quadratic Dp = 1/4 Δx^2 I
    
    @ti.func
    def compute_first_piola_kirchhoff_stress(self, F):   # Neo-Hookean model
        J = F.determinant()
        J = ti.max(J, 1e-6)  # avoid numerical issues
        F_invT = F.inverse().transpose()
        mu, lam = self.mu, self.lam
        P = mu * (F - F_invT) + lam * ti.log(J) * F_invT   # P = mu (F - F^{-T}) + lambda * log(J) F^{-T}
        return P
    