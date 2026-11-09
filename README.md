# MPM
Material Point Method
SIGGRAPH 2016 MPM Course Notes를 바탕으로 Material Point Method를 처음부터 구현한 2D 탄성체 시뮬레이터.
같은 알고리즘을 Taichi와 NVIDIA Warp 두 가지 GPU 프레임워크로 구현하였으며,
Warp 버전은 시뮬레이션 결과를 NVIDIA Omniverse(USD Composer)로 실시간 전송해 시각화.

## Features
- APIC transfer : affine 행렬(B, D⁻¹)을 이용한 Particle-to-Grid / Grid-to-Particle 전달
- Quadratic B-spline 보간 커널과 그 gradient
- Neo-Hookean 구성 모델로 1st Piola-Kirchhoff 응력 계산
- Explicit time integration : 그리드 내부력(응력)과 중력으로 그리드 속도 갱신
- 변형 그래디언트 갱신
- 경계 충돌 : 그리드 속도를 경계 법선 방향으로 사영해 침투 방지
- Omniverse 연동(Warp 버전) : named pipe로 입자 위치를 스트리밍하고 USD 'Points' prim을 실시간 갱신

## Algorithm
```
1. Grid reset
2. P2G
3. Grid velocity
4. Grid force
5. Grid update + boundary collision
6. F update
7. G2P
8. Advection
```

