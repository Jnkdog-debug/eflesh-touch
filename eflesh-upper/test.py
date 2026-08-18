import serial
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ============================================================
# Configuration
# ============================================================
COM_PORT = '/dev/ttyUSB0'   # change to your actual serial port
BAUD_RATE = 115200
NUM_SENSORS = 15             # total sensor count (S1 .. S15)
HISTORY_LENGTH = 50          # data points kept per sensor
Y_MIN, Y_MAX = -5000, 5000   # Y-axis range
REFRESH_MS = 50              # animation refresh interval

# ============================================================
# Setup serial
# ============================================================
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)

# ============================================================
# Data buffers — S1 through S{NUM_SENSORS}
# ============================================================
z_data = {
    f'S{i}': deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
    for i in range(1, NUM_SENSORS + 1)
}

# ============================================================
# Plot setup — 3 stacked subplots, 5 sensors each
# ============================================================
SENSORS_PER_ROW = 5
N_ROWS = 3
cmap = plt.cm.tab20
colors = [cmap(i % 20) for i in range(NUM_SENSORS)]

fig, axes = plt.subplots(N_ROWS, 1, figsize=(16, 12), sharex=True)
fig.suptitle(f'Tactile Sensor Array — {NUM_SENSORS} Sensors — Z Axis Realtime',
             fontsize=13, fontweight='bold')

lines = {}
for row_idx, ax in enumerate(axes):
    start = row_idx * SENSORS_PER_ROW + 1
    end = start + SENSORS_PER_ROW
    for i in range(start, end):
        s_name = f'S{i}'
        c = colors[i - 1]
        lines[s_name], = ax.plot(z_data[s_name], label=s_name, color=c,
                                 linewidth=1.2)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(0, HISTORY_LENGTH)
    ax.set_ylabel('Z Pressure')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', ncol=5, fontsize=8)

axes[-1].set_xlabel('Samples')

# ============================================================
# Regex — matches [S1:-1959.8,-2912.7,4918.2]
# ============================================================
pattern = re.compile(r'\[S(\d+):([^,]+),([^,]+),([^\]]+)\]')


def update(frame):
    """Read serial, push new readings into deques, refresh plot."""
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("FRAME:"):
                matches = pattern.findall(line)
                for match in matches:
                    s_id = f"S{match[0]}"
                    z = float(match[3])
                    if s_id in z_data:
                        z_data[s_id].append(z)
        except Exception:
            pass  # skip garbled bytes

    for s_name in z_data.keys():
        lines[s_name].set_ydata(z_data[s_name])
    return list(lines.values())


ani = animation.FuncAnimation(fig, update, interval=REFRESH_MS, blit=False)

# Maximise window (backend-dependent, best-effort)
mng = plt.get_current_fig_manager()
try:
    mng.window.showMaximized()
except Exception:
    try:
        mng.resize(*mng.window.maxsize())
    except Exception:
        pass  # not supported by this backend

plt.tight_layout()
plt.show()

ser.close()