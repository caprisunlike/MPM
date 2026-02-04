import taichi as ti
import numpy as np

@ti.data_oriented
class Simulation:
    def __init__(self, mpm, particles, grid, boundary):
        self.mpm = mpm
        self.particles = particles
        self.grid = grid
        self.boundary = boundary

    def simulation(self):
        self.grid.initialize()
        self.particle_to_grid(self.mpm, self.grid, self.particles)
        self.compute_grid_velocities(self.grid)
        #self.identify_grid_degree_of_freedoms()
        self.compute_explicit_grid_forces(self.mpm, self.grid, self.particles)
        self.grid_velocity_update(self.mpm, self.grid)
        self.update_particle_deformation_gradient(self.mpm, self.grid, self.particles)
        self.grid_to_particle(self.mpm, self.grid, self.particles)
        self.particle_advection(self.mpm, self.particles)
        #print("----- Step Finished -----")

    @ti.kernel
    def particle_to_grid(self, mpm: ti.template(), grid: ti.template(), particles: ti.template()):
        ti.loop_config(block_dim=256)
        ti.block_local(grid.mass, grid.vel)
        
        for p in range(mpm.particlesNum):
            base = mpm.pos_to_idx(particles.x[p])   # 기준이 되는 인덱스
            #w_sum = 0.0
            for i in range(3):
                for j in range(3):
                    # x/dx가 3.8이면 base=3, fx=0.8, i=0,1,2 -> idx=2,3,4
                    # x/dx가 3.3이면 base=2, fx=1.3, i=0,1,2 -> idx=1,2,3
                    idx = ti.Vector([base.x + i, base.y + j])
                    if idx.x < 0 or idx.x >= mpm.gridNum or idx.y < 0 or idx.y >= mpm.gridNum:
                        continue

                    fx = particles.x[p] * mpm.inv_dx - idx.cast(ti.f32)   # fractional position : idx로부터 상대 거리   (그리드 기준좌표)
                    w = grid.compute_weights(fx)
                    #w_sum += w

                    # grid mass
                    ti.atomic_add(grid.mass[idx], w * particles.m)

                    # grid momentum
                    affine = particles.B[p] @ particles.D_inv @ (-fx*mpm.dx)
                    momentum = particles.m * (particles.v[p] + affine)
                    ti.atomic_add(grid.vel[idx].x, w * momentum.x)
                    ti.atomic_add(grid.vel[idx].y, w * momentum.y)
            #if p == 0:
            #    print("Particle 0 weight sum:", w_sum)

        return
    
    @ti.kernel
    def compute_grid_velocities(self, grid: ti.template()):
        for i, j in grid.mass:
            if grid.mass[i, j] > 0.0:
                grid.vel[i, j] /= grid.mass[i, j]
            else:
                grid.mass[i, j] = 0.0
                grid.vel[i, j] = [0.0, 0.0]    # 만약 질량이 0이 아니고 음수면 어떻게 되는거지?

        return
    
    def identify_grid_degree_of_freedoms(self):
        return
    
    @ti.kernel
    def compute_explicit_grid_forces(self, mpm: ti.template(), grid: ti.template(), particles: ti.template()):
        ti.loop_config(block_dim=256)
        ti.block_local(grid.force)
        grid.force.fill([0.0, 0.0])

        for p in range(mpm.particlesNum):
            base = mpm.pos_to_idx(particles.x[p])

            P = particles.compute_first_piola_kirchhoff_stress(particles.F[p])
            F_T = particles.F[p].transpose()
            stress = P @ F_T

            for i in range(3):
                for j in range(3):
                    idx = ti.Vector([base.x + i, base.y + j])
                    if idx.x < 0 or idx.x >= mpm.gridNum or idx.y < 0 or idx.y >= mpm.gridNum:
                        continue
                    fx = particles.x[p] * mpm.inv_dx - idx.cast(ti.f32)
                    w = grid.compute_weights(fx)
                    grad_w = grid.compute_gradient_weights(fx)

                    # 내부힘
                    f_int = -particles.vol[p] * (stress @ grad_w)
                    # 중력
                    f_grav = w * particles.m * mpm.gravity

                    f = f_int + f_grav
                    ti.atomic_add(grid.force[idx].x, f.x)
                    ti.atomic_add(grid.force[idx].y, f.y)
        return 
    
    @ti.kernel
    def grid_velocity_update(self, mpm: ti.template(), grid: ti.template()):
        for i, j in grid.mass:
            if grid.mass[i,j] > 0.0:
                grid.vel[i,j] += mpm.dt * (grid.force[i,j] / grid.mass[i,j])
            grid.vel[i,j] = self.boundary.collision(grid.pos[i,j], grid.vel[i,j])   # collision
        return
    
    @ti.kernel
    def update_particle_deformation_gradient(self, mpm: ti.template(), grid: ti.template(), particles: ti.template()):
        for p in range(mpm.particlesNum):
            base = mpm.pos_to_idx(particles.x[p])
            new  = ti.Matrix.zero(ti.f32, mpm.dim, mpm.dim)
            for i in range(3):
                for j in range(3):
                    idx = ti.Vector([base.x + i, base.y + j])
                    if idx.x < 0 or idx.x >= mpm.gridNum or idx.y < 0 or idx.y >= mpm.gridNum:
                        continue
                    fx = particles.x[p] * mpm.inv_dx - idx.cast(ti.f32)
                    grad_w = grid.compute_gradient_weights(fx)

                    new += grid.vel[idx].outer_product(grad_w)
            particles.F[p] = (ti.Matrix.identity(ti.f32, mpm.dim) + mpm.dt * new) @ particles.F[p]

        return
    
    @ti.kernel
    def grid_to_particle(self, mpm: ti.template(), grid: ti.template(), particles: ti.template()):
        for p in range(mpm.particlesNum):
            base = mpm.pos_to_idx(particles.x[p])
            #w_sum = 0.0
            v_new = ti.Vector.zero(ti.f32, mpm.dim)
            B_new = ti.Matrix.zero(ti.f32, mpm.dim, mpm.dim)
            for i in range(3):
                for j in range(3):
                    idx = ti.Vector([base.x + i, base.y + j])
                    if idx.x < 0 or idx.x >= mpm.gridNum or idx.y < 0 or idx.y >= mpm.gridNum:
                        continue
                    fx = particles.x[p] * mpm.inv_dx - idx.cast(ti.f32)
                    w = grid.compute_weights(fx)
                    #w_sum += w
                    v_new += w * grid.vel[idx]
                    B_new += w * grid.vel[idx].outer_product(-fx*mpm.dx)
            particles.v[p] = v_new
            particles.B[p] = B_new
            #if p == 0:
            #    print("Particle 0 weight sum:", w_sum)
        return
    
    @ti.kernel
    def particle_advection(self, mpm: ti.template(), particles: ti.template()):
        ti.loop_config(block_dim=256)
        for p in range(mpm.particlesNum):
            particles.x[p] += mpm.dt * particles.v[p]   # 위치 업데이트
        return
    
    @ti.kernel
    def debug_particles(self, particles: ti.template()):
        for p in range(1):  # 처음 1개만
            x = particles.x[p]
            v = particles.v[p]
            print("p =", p, "x =", x, "v =", v)


