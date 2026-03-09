import warp as wp


@wp.kernel
def _boundary_initialize_kernel(
    b_pos: wp.array(dtype=wp.vec2),
    b_normal: wp.array(dtype=wp.vec2),
):
    # left
    b_pos[0] = wp.vec2(0.0, 0.5)
    b_normal[0] = wp.vec2(1.0, 0.0)
    # right
    b_pos[1] = wp.vec2(1.0, 0.5)
    b_normal[1] = wp.vec2(-1.0, 0.0)
    # bottom
    b_pos[2] = wp.vec2(0.5, 0.0)
    b_normal[2] = wp.vec2(0.0, 1.0)
    # top
    b_pos[3] = wp.vec2(0.5, 1.0)
    b_normal[3] = wp.vec2(0.0, -1.0)


@wp.func
def boundary_collision(
    grid_position: wp.vec2,
    grid_velocity: wp.vec2,
    b_pos: wp.array(dtype=wp.vec2),
    b_normal: wp.array(dtype=wp.vec2),
    b_num: int,
    dt: float,
) -> wp.vec2:
    grid_pos = grid_position
    grid_vel = grid_velocity

    for i in range(b_num):
        normal = b_normal[i]
        boundary_pos = b_pos[i]

        dist = wp.dot(normal, grid_pos - boundary_pos)
        trial_pos = grid_pos + dt * grid_vel
        trial_dist = wp.dot(normal, trial_pos - boundary_pos)

        dist_c = trial_dist - wp.min(dist, 0.0)
        if dist_c < 0.0:
            grid_vel = grid_vel - dist_c * normal / dt

    return grid_vel


class Boundary:
    def __init__(self, mpm):
        self.mpm = mpm
        self.b_num = mpm.boundaryNum

        self.b_pos = wp.zeros(self.b_num, dtype=wp.vec2)
        self.b_normal = wp.zeros(self.b_num, dtype=wp.vec2)

        self.initialize()

    def initialize(self):
        wp.launch(
            kernel=_boundary_initialize_kernel,
            dim=1,
            inputs=[self.b_pos, self.b_normal],
        )
