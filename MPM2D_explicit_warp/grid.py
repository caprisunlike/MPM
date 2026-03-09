import warp as wp


@wp.kernel
def _grid_initialize_kernel(
    mass: wp.array(dtype=wp.float32, ndim=2),
    vel: wp.array(dtype=wp.vec2, ndim=2),
    force: wp.array(dtype=wp.vec2, ndim=2),
):
    i, j = wp.tid()
    mass[i, j] = 0.0
    vel[i, j] = wp.vec2(0.0, 0.0)
    force[i, j] = wp.vec2(0.0, 0.0)


@wp.kernel
def _grid_init_pos_kernel(
    pos: wp.array(dtype=wp.vec2, ndim=2),
    dx: float,
):
    i, j = wp.tid()
    pos[i, j] = wp.vec2(float(i) * dx, float(j) * dx)


@wp.func
def grid_kernel(x: float) -> float:
    result = 0.0
    ax = wp.abs(x)

    if ax < 0.5:
        result = 0.75 - x * x
    elif ax < 1.5:
        t = 1.5 - ax
        result = 0.5 * t * t
    else:
        result = 0.0
    return result


@wp.func
def grid_d_kernel(x: float) -> float:
    result = 0.0
    ax = wp.abs(x)

    if ax < 0.5:
        result = -2.0 * x
    elif ax < 1.5:
        t = 1.5 - ax
        if x > 0.0:
            result = -t
        else:
            result = t
    else:
        result = 0.0
    return result


@wp.func
def grid_compute_weights(fx: wp.vec2) -> float:
    return grid_kernel(fx[0]) * grid_kernel(fx[1])


@wp.func
def grid_compute_gradient_weights(fx: wp.vec2, inv_dx: float) -> wp.vec2:
    # [inv_dx * d_kernel(fx.x)*kernel(fx.y), kernel(fx.x)*inv_dx*d_kernel(fx.y)]
    return wp.vec2(
        inv_dx * grid_d_kernel(fx[0]) * grid_kernel(fx[1]),
        grid_kernel(fx[0]) * inv_dx * grid_d_kernel(fx[1]),
    )


class Grid:
    def __init__(self, mpm):
        self.mpm = mpm
        n = mpm.gridNum
        dim = mpm.dim  # 2

        self.pos = wp.zeros((n, n), dtype=wp.vec2)
        self.mass = wp.zeros((n, n), dtype=wp.float32)
        self.vel = wp.zeros((n, n), dtype=wp.vec2)
        self.force = wp.zeros((n, n), dtype=wp.vec2)

        self.init_grid_position()
        self.initialize()

    def initialize(self):
        wp.launch(
            kernel=_grid_initialize_kernel,
            dim=(self.mpm.gridNum, self.mpm.gridNum),
            inputs=[self.mass, self.vel, self.force],
        )

    def init_grid_position(self):
        wp.launch(
            kernel=_grid_init_pos_kernel,
            dim=(self.mpm.gridNum, self.mpm.gridNum),
            inputs=[self.pos, float(self.mpm.dx)],
        )
