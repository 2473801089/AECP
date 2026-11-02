#!/usr/bin/env bash
# =============================================================
# 单次消融运行器（点云场景）
# 用法: run_one.sh <scene> <config> <run_idx> [timeout_sec]
#   scene  : Maze1 | Maze2
#   config : B0-baseline | A1-ray-only | A2-density-only |
#            A3-no-connectivity | A4-no-guidance | A5-ccl-3d
#   run    : 1..N   （同场景同编号在不同配置间配对）
# 产物: $ABL_ROOT/<scene>/<config>/runNN/
# =============================================================
set -u

SCENE=${1:?scene}
CONFIG=${2:?config}
RUN=${3:?run_index}
TIMEOUT=${4:-420}
STARTUP_SEC=${STARTUP_SEC:-20}

WS=${ABL_WS:-/home/djg/FUEL}
ROOT=${ABL_ROOT:-/home/djg/Data_ablation}
DIR="$ROOT/$SCENE/$CONFIG/run$(printf %02d "$RUN")"

# 每次运行必须是干净目录：旧日志会让“结束判定”误触发
rm -rf "$DIR"
mkdir -p "$DIR"

case "$SCENE" in
  Maze1)  # office.pcd  16 x 30 x 2.9  -> 论文 Maze1 30x16x2
    PCD="$WS/src/FUEL/uav_simulator/map_generator/resource/office.pcd"
    MSX=20.0; MSY=34.0; MSZ=4.0
    BMNX=-9.0; BMNY=-16.0; BMNZ=0.0; BMXX=9.0; BMXY=16.0; BMXZ=2.0 ;;
  Maze2)  # office2.pcd 20 x 20 x 2.9 -> 论文 Maze2 20x20x2
    PCD="$WS/src/FUEL/uav_simulator/map_generator/resource/office2.pcd"
    MSX=24.0; MSY=24.0; MSZ=4.0
    BMNX=-11.0; BMNY=-11.0; BMNZ=0.0; BMXX=11.0; BMXY=11.0; BMXZ=2.0 ;;
  *)
    echo "[run] unknown scene: $SCENE" >&2; exit 2 ;;
esac

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

export ROS_LOG_DIR="$DIR/roslog"
export ROS_HOME="$DIR/roshome"
mkdir -p "$ROS_LOG_DIR" "$ROS_HOME"

if rosnode list >/dev/null 2>&1; then
  echo "[run] ROS master already running, refusing to start (stale run?)" >&2
  exit 3
fi

cd "$WS/src/FUEL" || exit 4
{ echo "commit: $(git rev-parse HEAD)"; echo "branch: $(git branch --show-current)"; } > "$DIR/git_head.txt"

DATE_BEGIN=$(date -Is)
# setsid: 让 roslaunch 成为进程组长，退出时整组回收，避免残留仿真进程
setsid roslaunch --pid="$DIR/roslaunch.pid" exploration_manager ablation_pcd.launch \
  pcd:="$PCD" \
  map_size_x:="$MSX" map_size_y:="$MSY" map_size_z:="$MSZ" \
  box_min_x:="$BMNX" box_min_y:="$BMNY" box_min_z:="$BMNZ" \
  box_max_x:="$BMXX" box_max_y:="$BMXY" box_max_z:="$BMXZ" \
  log_dir:="$DIR" > "$DIR/launch.log" 2>&1 &
LPID=$!
PGID=$LPID

cleanup() {
  kill -INT -"$PGID" 2>/dev/null
  for _ in $(seq 1 15); do
    kill -0 -"$PGID" 2>/dev/null || break
    sleep 1
  done
  kill -9 -"$PGID" 2>/dev/null
}
trap cleanup EXIT

sleep "$STARTUP_SEC"
if ! kill -0 "$LPID" 2>/dev/null; then
  echo "crashed_at_startup,0" > "$DIR/termination.txt"
  echo "[run] launch died during startup, see $DIR/launch.log" >&2
  exit 5
fi

rosparam dump "$DIR/rosparam.yaml" / 2>/dev/null || true
if [ -s "$DIR/rosparam.yaml" ]; then
  sha256sum "$DIR/rosparam.yaml" | awk '{print $1}' > "$DIR/params.sha256"
fi

# 触发探索；记录触发前的日志行数，之后的 FINISH 才有效
LOG_LINES=$(wc -l < "$DIR/launch.log")
rostopic pub -1 /waypoint_generator/waypoints nav_msgs/Path \
  "{header: {frame_id: world}, poses: [{header: {frame_id: world}, pose: {position: {x: 1.0, y: 0.0, z: 1.0}, orientation: {w: 1.0}}}]}" \
  >/dev/null 2>&1

START=$(date +%s)
TERM="unknown"
while true; do
  if tail -n +$((LOG_LINES + 1)) "$DIR/launch.log" 2>/dev/null | grep -q "finish exploration"; then
    TERM="finished"; break
  fi
  if ! kill -0 "$LPID" 2>/dev/null; then
    TERM="crashed"; break
  fi
  if [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
    TERM="timeout"; break
  fi
  sleep 2
done
END=$(date +%s)
DATE_END=$(date -Is)

printf '%s,%s\n' "$TERM" "$((END-START))" > "$DIR/termination.txt"

python3 "$(dirname "$0")/write_meta.py" \
  --dir "$DIR" --scene "$SCENE" --config "$CONFIG" --run "$RUN" \
  --termination "$TERM" --duration "$((END-START))" \
  --begin "$DATE_BEGIN" --end "$DATE_END" 2>/dev/null || true

echo "[run] $SCENE/$CONFIG/run$(printf %02d "$RUN") -> $TERM ($((END-START))s)"
