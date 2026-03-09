import warp as wp


@wp.kernel
def _particles_initialize_kernel(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    a: wp.array(dtype=wp.vec2),
    vol: wp.array(dtype=wp.float32),
    F: wp.array(dtype=wp.mat22),
    B: wp.array(dtype=wp.mat22),
    dx: float,
    seed: int,
):
    i = wp.tid()

    state = wp.rand_init(seed, i)

    x[i] = wp.vec2(wp.randf(state) * 0.8 + 0.1, wp.randf(state) * 0.8 + 0.1)
    v[i] = wp.vec2(0.9, 0.0)
    a[i] = wp.vec2(0.0, 0.0)
    vol[i] = dx * dx

    F[i] = wp.mat22(1.0, 0.0,
                    0.0, 1.0)
    B[i] = wp.mat22(0.0, 0.0,
                    0.0, 0.0)


@wp.func
def particles_compute_first_piola_kirchhoff_stress(F: wp.mat22, mu: float, lam: float) -> wp.mat22:
    # Neo-Hookean
    J = wp.determinant(F)
    J = wp.max(J, 1e-6)

    F_invT = wp.transpose(wp.inverse(F))
    # P = mu (F - F^{-T}) + lambda * log(J) F^{-T}
    P = mu * (F - F_invT) + lam * wp.log(J) * F_invT
    return P


class Particles:
    def __init__(self, mpm):
        self.mpm = mpm
        n = mpm.particlesNum
        dim = mpm.dim  # 2

        self.m = 0.01
        self.r = 2.0

        self.x = wp.zeros(n, dtype=wp.vec2)
        self.v = wp.zeros(n, dtype=wp.vec2)
        self.a = wp.zeros(n, dtype=wp.vec2)
        self.vol = wp.zeros(n, dtype=wp.float32)

        self.F = wp.zeros(n, dtype=wp.mat22)
        self.B = wp.zeros(n, dtype=wp.mat22)

        v = 0.25
        self.D = wp.mat22(v, 0.0,
                          0.0, v)
        self.D_inv = wp.inverse(self.D)

        self.E = 5000.0
        self.nu = 0.2
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        self.initialize()

    def initialize(self):
        wp.launch(
            kernel=_particles_initialize_kernel,
            dim=self.mpm.particlesNum,
            inputs=[self.x, self.v, self.a, self.vol, self.F, self.B, float(self.mpm.dx), 2026],
        )
