import warp as wp

from grid import grid_compute_weights, grid_compute_gradient_weights
from particles import particles_compute_first_piola_kirchhoff_stress
from boundary import boundary_collision


@wp.func
def _pos_to_idx(x: wp.vec2, inv_dx: float) -> wp.vec2i:
    # (x * inv_dx - 0.5).cast(int)
    return wp.vec2i(int(x[0] * inv_dx - 0.5), int(x[1] * inv_dx - 0.5))


@wp.func
def _outer_product(a: wp.vec2, b: wp.vec2) -> wp.mat22:
    # a outer b
    return wp.mat22(
        a[0] * b[0], a[0] * b[1],
        a[1] * b[0], a[1] * b[1],
    )


@wp.kernel
def particle_to_grid(
    gridNum: int,
    inv_dx: float,
    dx: float,

    grid_mass: wp.array(dtype=wp.float32, ndim=2),
    grid_vel: wp.array(dtype=wp.vec2, ndim=2),

    p_x: wp.array(dtype=wp.vec2),
    p_v: wp.array(dtype=wp.vec2),
    p_B: wp.array(dtype=wp.mat22),
    p_m: float,
    p_D_inv: wp.mat22,
):
    p = wp.tid()
    base = _pos_to_idx(p_x[p], inv_dx)

    for i in range(3):
        for j in range(3):
            gi = base[0] + i
            gj = base[1] + j

            if gi < 0 or gi >= gridNum or gj < 0 or gj >= gridNum:
                continue

            idx_f = wp.vec2(float(gi), float(gj))
            fx = p_x[p] * inv_dx - idx_f

            w = grid_compute_weights(fx)

            # grid mass
            wp.atomic_add(grid_mass, gi, gj, w * p_m)

            # grid momentum
            affine = p_B[p] * p_D_inv * (-fx * dx)
            momentum = p_m * (p_v[p] + affine)

            # atomic add vec2 (최근 Warp에서는 지원되는 편)
            wp.atomic_add(grid_vel, gi, gj, w * momentum)


@wp.kernel
def compute_grid_velocities(
    grid_mass: wp.array(dtype=wp.float32, ndim=2),
    grid_vel: wp.array(dtype=wp.vec2, ndim=2),
):
    i, j = wp.tid()
    m = grid_mass[i, j]
    if m > 0.0:
        grid_vel[i, j] = grid_vel[i, j] / m
    else:
        grid_mass[i, j] = 0.0
        grid_vel[i, j] = wp.vec2(0.0, 0.0)


@wp.kernel
def clear_grid_force(
    grid_force: wp.array(dtype=wp.vec2, ndim=2),
):
    i, j = wp.tid()
    grid_force[i, j] = wp.vec2(0.0, 0.0)


@wp.kernel
def compute_explicit_grid_forces(
    gridNum: int,
    inv_dx: float,
    dx: float,

    grid_force: wp.array(dtype=wp.vec2, ndim=2),

    p_x: wp.array(dtype=wp.vec2),
    p_vol: wp.array(dtype=wp.float32),
    p_F: wp.array(dtype=wp.mat22),
    p_m: float,
    gravity: wp.vec2,

    mu: float,
    lam: float,
):
    p = wp.tid()
    base = _pos_to_idx(p_x[p], inv_dx)

    P = particles_compute_first_piola_kirchhoff_stress(p_F[p], mu, lam)
    F_T = wp.transpose(p_F[p])
    stress = P * F_T

    for i in range(3):
        for j in range(3):
            gi = base[0] + i
            gj = base[1] + j

            if gi < 0 or gi >= gridNum or gj < 0 or gj >= gridNum:
                continue

            idx_f = wp.vec2(float(gi), float(gj))
            fx = p_x[p] * inv_dx - idx_f

            w = grid_compute_weights(fx)
            grad_w = grid_compute_gradient_weights(fx, inv_dx)

            # 내부힘
            f_int = -p_vol[p] * (stress * grad_w)
            # 중력
            f_grav = w * p_m * gravity

            f = f_int + f_grav

            wp.atomic_add(grid_force, gi, gj, f)


@wp.kernel
def grid_velocity_update(
    grid_mass: wp.array(dtype=wp.float32, ndim=2),
    grid_vel: wp.array(dtype=wp.vec2, ndim=2),
    grid_force: wp.array(dtype=wp.vec2, ndim=2),
    grid_pos: wp.array(dtype=wp.vec2, ndim=2),

    b_pos: wp.array(dtype=wp.vec2),
    b_normal: wp.array(dtype=wp.vec2),
    b_num: int,

    dt: float,
):
    i, j = wp.tid()
    m = grid_mass[i, j]
    if m > 0.0:
        grid_vel[i, j] = grid_vel[i, j] + dt * (grid_force[i, j] / m)

    grid_vel[i, j] = boundary_collision(
        grid_pos[i, j],
        grid_vel[i, j],
        b_pos,
        b_normal,
        b_num,
        dt,
    )


@wp.kernel
def update_particle_deformation_gradient(
    gridNum: int,
    inv_dx: float,
    dx: float,
    dt: float,

    grid_vel: wp.array(dtype=wp.vec2, ndim=2),

    p_x: wp.array(dtype=wp.vec2),
    p_F: wp.array(dtype=wp.mat22),
):
    p = wp.tid()
    base = _pos_to_idx(p_x[p], inv_dx)

    new = wp.mat22(0.0, 0.0,
                   0.0, 0.0)

    for i in range(3):
        for j in range(3):
            gi = base[0] + i
            gj = base[1] + j

            if gi < 0 or gi >= gridNum or gj < 0 or gj >= gridNum:
                continue

            idx_f = wp.vec2(float(gi), float(gj))
            fx = p_x[p] * inv_dx - idx_f
            grad_w = grid_compute_gradient_weights(fx, inv_dx)

            new = new + _outer_product(grid_vel[gi, gj], grad_w)

    I = wp.mat22(1.0, 0.0,
                 0.0, 1.0)

    p_F[p] = (I + dt * new) * p_F[p]


@wp.kernel
def grid_to_particle(
    gridNum: int,
    inv_dx: float,
    dx: float,

    grid_vel: wp.array(dtype=wp.vec2, ndim=2),

    p_x: wp.array(dtype=wp.vec2),
    p_v: wp.array(dtype=wp.vec2),
    p_B: wp.array(dtype=wp.mat22),
):
    p = wp.tid()
    base = _pos_to_idx(p_x[p], inv_dx)

    v_new = wp.vec2(0.0, 0.0)
    B_new = wp.mat22(0.0, 0.0,
                     0.0, 0.0)

    for i in range(3):
        for j in range(3):
            gi = base[0] + i
            gj = base[1] + j

            if gi < 0 or gi >= gridNum or gj < 0 or gj >= gridNum:
                continue

            idx_f = wp.vec2(float(gi), float(gj))
            fx = p_x[p] * inv_dx - idx_f
            w = grid_compute_weights(fx)

            v_new = v_new + w * grid_vel[gi, gj]
            B_new = B_new + w * _outer_product(grid_vel[gi, gj], -fx * dx)

    p_v[p] = v_new
    p_B[p] = B_new


@wp.kernel
def particle_advection(
    dt: float,
    p_x: wp.array(dtype=wp.vec2),
    p_v: wp.array(dtype=wp.vec2),
):
    p = wp.tid()
    p_x[p] = p_x[p] + dt * p_v[p]


class Simulation:
    def __init__(self, mpm, particles, grid, boundary):
        self.mpm = mpm
        self.particles = particles
        self.grid = grid
        self.boundary = boundary

    def simulation(self):
        # 원본 호출 순서 그대로 :contentReference[oaicite:5]{index=5}
        self.grid.initialize()

        wp.launch(
            kernel=particle_to_grid,
            dim=self.mpm.particlesNum,
            inputs=[
                self.mpm.gridNum,
                float(self.mpm.inv_dx),
                float(self.mpm.dx),
                self.grid.mass,
                self.grid.vel,
                self.particles.x,
                self.particles.v,
                self.particles.B,
                float(self.particles.m),
                self.particles.D_inv,
            ],
        )

        wp.launch(
            kernel=compute_grid_velocities,
            dim=(self.mpm.gridNum, self.mpm.gridNum),
            inputs=[self.grid.mass, self.grid.vel],
        )

        wp.launch(
            kernel=clear_grid_force,
            dim=(self.mpm.gridNum, self.mpm.gridNum),
            inputs=[self.grid.force],
        )

        wp.launch(
            kernel=compute_explicit_grid_forces,
            dim=self.mpm.particlesNum,
            inputs=[
                self.mpm.gridNum,
                float(self.mpm.inv_dx),
                float(self.mpm.dx),
                self.grid.force,
                self.particles.x,
                self.particles.vol,
                self.particles.F,
                float(self.particles.m),
                self.mpm.gravity,
                float(self.particles.mu),
                float(self.particles.lam),
            ],
        )

        wp.launch(
            kernel=grid_velocity_update,
            dim=(self.mpm.gridNum, self.mpm.gridNum),
            inputs=[
                self.grid.mass,
                self.grid.vel,
                self.grid.force,
                self.grid.pos,
                self.boundary.b_pos,
                self.boundary.b_normal,
                int(self.boundary.b_num),
                float(self.mpm.dt),
            ],
        )

        wp.launch(
            kernel=update_particle_deformation_gradient,
            dim=self.mpm.particlesNum,
            inputs=[
                self.mpm.gridNum,
                float(self.mpm.inv_dx),
                float(self.mpm.dx),
                float(self.mpm.dt),
                self.grid.vel,
                self.particles.x,
                self.particles.F,
            ],
        )

        wp.launch(
            kernel=grid_to_particle,
            dim=self.mpm.particlesNum,
            inputs=[
                self.mpm.gridNum,
                float(self.mpm.inv_dx),
                float(self.mpm.dx),
                self.grid.vel,
                self.particles.x,
                self.particles.v,
                self.particles.B,
            ],
        )

        wp.launch(
            kernel=particle_advection,
            dim=self.mpm.particlesNum,
            inputs=[
                float(self.mpm.dt),
                self.particles.x,
                self.particles.v,
            ],
        )
