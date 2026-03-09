import socket
import struct
import time
import numpy as np

import os
import subprocess
import sys
from pathlib import Path

import omni
from pxr import UsdGeom, Gf, Vt, Sdf

def launch_sender_once():
    # 프로젝트 루트
    project_dir = Path("C:\\Users\\user\\Documents\\graphicslab\\MPM\\MPM2D_explicit_warp")

    # sender 파일
    sender_path = project_dir / "mpm.py"

    # lock 파일
    lock_path = project_dir / ".sender_running.lock"

    if lock_path.exists():
        print("Sender seems already launched (lock exists).")
        return

    if not sender_path.exists():
        print("Sender file not found:", sender_path)
        return

    try:
        lock_path.write_text("running", encoding="utf-8")
    except Exception as e:
        print("Failed to create lock:", e)

    #python_exe = sys.executable
    python_exe = Path("C:\\Users\\user\\anaconda3\\python.exe")

    if not python_exe.exists():
        print("Python executable not found:", python_exe)
        return

    try:
        lock_path.write_text("running", encoding="utf-8")
    except Exception as e:
        print("Failed to create lock:", e)

    print("Launching sender:", sender_path)

    subprocess.Popen(
        [python_exe, str(sender_path)],
        cwd=str(project_dir),
        #creationflags=subprocess.CREATE_NO_WINDOW,
    )

HOST, PORT = "127.0.0.1", 5005
MAGIC = 0x4D504D31
MSG_DATA = 1
MSG_END  = 2

EXPECTED_N = 10000   # 고정 파티클 수
IDLE_TIMEOUT_SEC = 10.0   # END가 드랍돼도 10초 후 종료
COMMIT_THRESHOLD = 0.90   # 커밋 임계치

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
sock.setblocking(False)

ctx = omni.usd.get_context()
stage = ctx.get_stage()

prim_path = "/World/MPM/Particles"
prim = stage.GetPrimAtPath(prim_path)
if not prim.IsValid():
    UsdGeom.Xform.Define(stage, "/World/MPM")
    points = UsdGeom.Points.Define(stage, prim_path)
else:
    points = UsdGeom.Points(prim)

points_attr = points.GetPointsAttr()
#points.GetWidthsAttr().Set([0.003])
widths_attr = points.GetWidthsAttr()
widths_attr.Set(Vt.FloatArray([0.005]))
widths_attr.SetMetadata("interpolation", "constant")
points.GetDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.8, 1.0)])

# --------------------------------------------------------------
# 카메라
camera_path = "/World/MPM/Camera"
camera_prim = stage.GetPrimAtPath(camera_path)

if not camera_prim.IsValid():
    camera = UsdGeom.Camera.Define(stage, camera_path)
else:
    camera = UsdGeom.Camera(camera_prim)

# 카메라 위치
xform = UsdGeom.Xformable(camera.GetPrim())
xform.ClearXformOpOrder()

translate_op = xform.AddTranslateOp()
rotate_op = xform.AddRotateXYZOp()

# [0,1]x[0,1] 영역을 정면으로 보게
translate_op.Set(Gf.Vec3d(0.5, 0.5, 5.0))
rotate_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))

camera.GetFocalLengthAttr().Set(50.0)   # focal length
camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))   # clipping range

# --------------------------------------------------------------

display_pos = np.zeros((EXPECTED_N, 3), dtype=np.float32)   # 화면에 실제 표시되는 마지막 완성 프레임
frame_pos = np.zeros((EXPECTED_N, 3), dtype=np.float32)     # 현재 수신 중인 프레임 버퍼
received = np.zeros((EXPECTED_N,), dtype=np.bool_)

finished = False
sub = None

# 프레임 커밋용 상태
current_frame = None
updated_this_frame = False

last_recv_time = time.time()

def commit_stage():
    arr = Vt.Vec3fArray.FromNumpy(display_pos)  # 이미 float32
    points_attr.Set(arr)

def can_commit_frame():
    received_count = np.count_nonzero(received)
    ratio = received_count / EXPECTED_N
    return ratio >= COMMIT_THRESHOLD

def shutdown(reason: str):
    global finished, sub, sock
    if finished:
        return
    finished = True
    try:
        if sub is not None:
            sub.unsubscribe()
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass
    print(f"Receiver shut down: {reason}")

def on_update(e):
    global current_frame, updated_this_frame, last_recv_time
    global display_pos, frame_pos, received

    if finished:
        return

    now = time.time()

    if (now - last_recv_time) > IDLE_TIMEOUT_SEC:
        if updated_this_frame and can_commit_frame():
            display_pos[:] = frame_pos
            commit_stage()
        shutdown(f"idle timeout ({IDLE_TIMEOUT_SEC}s)")
        return

    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except BlockingIOError:
            break

        last_recv_time = now

        if len(data) < 16:
            continue

        magic, msg_type, frame, a = struct.unpack_from("<IIII", data, 0)
        if magic != MAGIC:
            continue

        if msg_type == MSG_END:
            if updated_this_frame and can_commit_frame():
                display_pos[:] = frame_pos
                commit_stage()
            shutdown("END received")
            return

        if msg_type != MSG_DATA or len(data) < 20:
            continue

        magic2, msg2, frame2, start, chunk_count = struct.unpack_from("<IIIII", data, 0)

        payload = data[20:]
        expected_bytes = chunk_count * 3 * 4
        if len(payload) != expected_bytes:
            continue

        # 새 프레임 시작
        if current_frame is None:
            current_frame = frame2
            frame_pos.fill(0.0)
            received[:] = False
            updated_this_frame = False

        elif frame2 != current_frame:
            if updated_this_frame and can_commit_frame():
                display_pos[:] = frame_pos
                commit_stage()

            # 새 프레임 버퍼 초기화
            current_frame = frame2
            frame_pos.fill(0.0)
            received[:] = False
            updated_this_frame = False

        chunk = np.frombuffer(payload, dtype=np.float32).reshape(chunk_count, 3)
        end = start + chunk_count

        if start < 0:
            continue
        if end > EXPECTED_N:
            chunk = chunk[: max(0, EXPECTED_N - start)]
            end = min(end, EXPECTED_N)
            if end <= start:
                continue

        frame_pos[start:end] = chunk
        received[start:end] = True
        updated_this_frame = True

sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)
print("MPM live receiver running. Listening on UDP", HOST, PORT)
print("Prim:", prim_path, "Expected N:", EXPECTED_N)
launch_sender_once()