import time
import threading
import subprocess
from pathlib import Path
from multiprocessing.connection import Listener

import numpy as np
import omni
from pxr import UsdGeom, Gf, Vt


PIPE_NAME = r"\\.\pipe\mpm_warp_pipe"
EXPECTED_N = 10000
IDLE_TIMEOUT_SEC = 10.0
sender_process = None

def launch_sender_once():
    global sender_process

    project_dir = Path("C:\\Users\\user\\Documents\\graphicslab\\MPM\\MPM2D_explicit_warp")   # 프로젝트 루트
    sender_path = project_dir / "mpm.py"   # sender 파일

    if sender_process is not None and sender_process.poll() is None:
        print("Sender is already running.")
        return

    if not sender_path.exists():
        print("Sender file not found:", sender_path)
        return

    #python_exe = sys.executable
    python_exe = Path("C:\\Users\\user\\anaconda3\\python.exe")

    if not python_exe.exists():
        print("Python executable not found:", python_exe)
        return

    print("Launching sender:", sender_path)

    sender_process = subprocess.Popen(
        [python_exe, str(sender_path)],
        cwd=str(project_dir),
        # creationflags=subprocess.CREATE_NO_WINDOW,
    )


# --------------------------------------------------------------
# USD Points 설정
# --------------------------------------------------------------
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
# 카메라 설정
# --------------------------------------------------------------
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
# Named pipe 수신
# --------------------------------------------------------------
listener = Listener(PIPE_NAME, family="AF_PIPE")
conn = None

finished = False
sub = None
last_recv_time = time.time()


def wait_connection():
    global conn, last_recv_time

    print("Waiting for MPM pipe connection:", PIPE_NAME)
    conn = listener.accept()
    last_recv_time = time.time()
    print("MPM pipe connected.")


def set_particles(pos):
    if pos is None:
        return

    if len(pos) > EXPECTED_N:
        pos = pos[:EXPECTED_N]

    points_attr.Set(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32, copy=False)))


def shutdown(reason):
    global finished, sub, conn, sender_process

    if finished:
        return

    finished = True

    try:
        if sub is not None:
            sub.unsubscribe()
    except Exception:
        pass

    try:
        if conn is not None:
            conn.close()
    except Exception:
        pass

    try:
        if sender_process is not None and sender_process.poll() is None:
            sender_process.terminate()

            try:
                sender_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                sender_process.kill()
    except Exception:
        pass

    try:
        listener.close()
    except Exception:
        pass

    print("Receiver shut down:", reason)


def on_update(e):
    global last_recv_time

    if finished:
        return

    if conn is None:
        return

    now = time.time()

    if now - last_recv_time > IDLE_TIMEOUT_SEC:
        shutdown("idle timeout")
        return

    latest_pos = None

    # 한 update 동안 쌓인 프레임이 여러 개면 모두 읽고 마지막 프레임만 화면에 표시한다.
    # 이렇게 하면 USD Points 갱신 횟수를 줄일 수 있다.
    while conn.poll():
        try:
            frame, pos = conn.recv()
        except EOFError:
            shutdown("pipe closed")
            return

        last_recv_time = now

        if frame == "END":
            shutdown("simulation finished")
            return

        latest_pos = pos

    if latest_pos is not None:
        set_particles(latest_pos)


threading.Thread(target=wait_connection, daemon=True).start()

sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)
print("MPM live receiver running.")
print("Listening on named pipe:", PIPE_NAME)
print("Prim:", prim_path, "Expected N:", EXPECTED_N)
launch_sender_once()
