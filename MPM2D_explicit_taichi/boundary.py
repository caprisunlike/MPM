import taichi as ti

@ti.data_oriented
class Boundary:
    def __init__(self, mpm: ti.template()):
        self.mpm = mpm
        self.b_num = mpm.boundaryNum
        self.b_pos = ti.Vector.field(n=2, dtype=ti.f32, shape=self.b_num)
        self.b_normal = ti.Vector.field(n=2, dtype=ti.f32, shape=self.b_num)

        self.mu_f = 0.2   # friction coefficient

        self.initialize()
    
    @ti.kernel
    def initialize(self):
        # left
        self.b_pos[0] = ti.Vector([0.0, 0.5])
        self.b_normal[0] = ti.Vector([1.0, 0.0])
        # right
        self.b_pos[1] = ti.Vector([1.0, 0.5])
        self.b_normal[1] = ti.Vector([-1.0, 0.0])
        # bottom
        self.b_pos[2] = ti.Vector([0.5, 0.0])
        self.b_normal[2] = ti.Vector([0.0, 1.0])
        # top
        self.b_pos[3] = ti.Vector([0.5, 1.0])
        self.b_normal[3] = ti.Vector([0.0, -1.0])

    @ti.func   # Implementation ref : Elias-Gu/MPM2D/src/border.cpp
    def collision(self, grid_position, grid_velocity):   # Separating type만 구현[type 종류 : Sticky(붙기), Separating(밀어내기), Sliding(미끄러지기)]
        grid_pos = grid_position
        grid_vel = grid_velocity
        dt = self.mpm.dt

        for i in range(self.b_num):
            normal = self.b_normal[i]
            boundary_pos = self.b_pos[i]

            dist = normal.dot(grid_pos - boundary_pos)   # 현재 위치 기준 거리 : (boundary의 normal벡터)와 (grid와 boundary의 거리)의 내적
            trial_pos = grid_pos + dt * grid_vel   # 예상되는 다음 위치
            trial_dist = normal.dot(trial_pos - boundary_pos)   # 다음 위치 기준 거리 : (boundary의 normal벡터)와 (다음 grid와 boundary의 거리)의 내적
            
            dist_c = trial_dist - ti.min(dist, 0.0)   # 현재 위치에서 이미 음수인 경우(충돌O) dist값 유지, 양수인 경우(충돌X) 0으로 변경
            if dist_c < 0.0:   # dist_c가 음수면 충돌 발생
                delta_vn = -dist_c / dt    # normal 방향 속도의 변화량, 벽이 밀어내야 할 속도
                grid_vel += delta_vn * normal
                grid_vel = self.friction(grid_vel, delta_vn, normal, self.mu_f)   # friction
        
        return grid_vel
    
        # dist_c는 다음 스텝에서 충돌이 얼마나 발생하는지(벽에 얼마나 침투하는지)만 계산
        # 현재는 충돌이 아니고 다음 스텝에서 충돌이 발생하는 경우 trial_dist만 사용하면 벽에 얼마나 침투했는지 계산 가능
        # 현재 충돌이고 다음 스텝에서도 충돌이 발생하는 경우 현재 위치에서 얼마나 더 침투했는지를 계산.
        """
        <케이스 A> : 현재는 안전 (distance=+0.2), 다음은 침투 (trial=-0.1)
        baseline = 0
        dist_c = -0.1 - 0 = -0.1
        → 벽 쪽으로 들어가는 만큼 속도 제거

        <케이스 B> : 이미 침투 (distance=-0.2), 다음은 그대로 (trial=-0.2)
        baseline = -0.2
        dist_c = -0.2 - (-0.2) = 0
        → 이번 스텝에 “더 들어간 게 없으니” 보정 0 (의도된 동작)

        <케이스 C> : 이미 침투 (distance=-0.2), 다음은 더 침투 (trial=-0.3)
        baseline = -0.2
        dist_c = -0.3 - (-0.2) = -0.1
        → 추가로 더 들어가려는 -0.1만큼만 속도 제거

        현재 충돌이든 아니든 동일한 목적(추가 침투 제거)을 달성하는 방식
        """

    @ti.func   # Implementation ref : Elias-Gu/MPM2D/src/border.cpp
    def friction(self, vel, delta_vn, normal, mu_f):   # Coulomb friction
        vt = vel - normal * vel.dot(normal)   # tangential velocity (접선 속도) : v_t = v_rel - n(v_norm)
        if vt.norm() > 1e-7:   # 접선 속도가 거의 0이면 건너뜀(미끄러지는 게 거의 없으므로)
            # 마찰 한도가 접선 속도보다 크면 접선 속도만큼만 줄여서 0으로 만들고(정지 마찰), 작으면 마찰 한도만큼만 줄임(운동 마찰)
            vel -= ti.min(vt.norm(), mu_f * delta_vn) * (vt / vt.norm())   # vel = vel - 줄일 양 * vt의 단위벡터(미끄러지는 방향)
        
        return vel

        """
        <케이스 A> : 접선 속도 3, 마찰 한도 5 (마찰 한도가 더 크므로 접선 속도만큼만 줄임)
        min = 3
        접선 속도 - min = 0
        → 접선 속도가 0이 됨 (정지 마찰)

        <케이스 B> : 접선 속도 3, 마찰 한도 1 (마찰 한도가 더 작으므로 마찰 한도만큼만 줄임)
        min = 1
        접선 속도 - min = 2
        → 접선 속도가 2가 됨 (운동 마찰)
        """