
# 2026-05-14 前沿选择与重复访问抑制优化日志

## 背景

上一版在 FUEL 原始 ATSP 代价矩阵上加入了残留债务、覆盖重叠、局部回头 closeout 和冷却机制，希望减少未知区域探索中的远距离回头补观测。但新一轮运行只进行到约 200 多轮就没有完成探索，主要现象是无人机在岔路口附近反复重规划，无法稳定判断下一步应该进入哪一条分支。

## 数据现象

本轮 `Data_debug` 中的日志显示，`frontier_track_log.csv` 约 3486 行，`local_view_log.csv` 约 1009 行，`start_cost_log.csv` 约 3486 行，总轮数约 248 轮。`stale_fraction` 基本为 0，说明很多所谓“遗忘前沿”并不是前沿自然失效导致的。

`forget_flag=1` 的前沿中，大量来自 debt-only 触发；`selected_reason=2` 的局部优先项多数 `coverage_overlap` 接近 0，且 `progress_delta` 基本为 0。这说明上一版的局部修复逻辑过于激进，把一些并没有真实路径覆盖证据的前沿提前到局部第一位，导致岔路口处反复选择同一类低进展目标。

局部日志中还出现同一稳定签名前沿多次排在第一，例如某些前沿累计 `local_pass_cnt` 很高但 `progress_delta=0`，说明无人机在局部最优附近不断尝试、冷却又不够硬，最终表现为岔路口反复重规划。

`start_cost_log.csv` 中 `path_overlap` 偏高，但目标前沿本身的覆盖重叠并不高，说明旧路径走廊的惩罚影响到了新分支选择。在岔路口场景中，进入任意新分支前都可能先经过同一段旧走廊，如果直接用整条路径 overlap 作为惩罚，会让多个分支的代价变得难以区分。

## 本次代码优化

1. 将全局 frontier-to-frontier ATSP 边上的 debt、overlap、区域切换和回返惩罚默认关闭。默认情况下，全局前沿间代价恢复为 FUEL 原始代价，避免在岔路口由实验代价扰乱主方向选择。

2. 将 start 行的 debt bonus 和 overlap penalty 默认关闭，并把 bonus 上限保持在 30% 原始运动代价。这样全局第一目标仍以运动代价、凹陷区域代价、狭窄通道代价为主，不再被局部残留债务强行牵引。

3. 删除 debt-only 遗忘触发。现在遗忘标记必须来自 stale 证据、路径覆盖证据，或者非常近距离下的局部多次经过证据。单纯 debt_score 高不再能把前沿标成遗忘区域。

4. 收紧局部 closeout 逻辑。每轮最多只允许 1 个 closeout 目标插到局部首位，并且必须满足真实 forget、覆盖重叠，或非常近距离、多次经过、低进展的小残留条件。

5. 强化 cooldown。处于冷却期的前沿默认不能再次作为 closeout，除非它有很强覆盖证据并且距离无人机非常近。这样可以防止同一个局部目标在岔路口反复被推到第一位。

6. 将路径重复惩罚改为 tail overlap，只看路径末端约 3m 的重复覆盖，并且惩罚还要乘以目标前沿自身的 `coverage_overlap`。这样旧走廊不会单独惩罚通往新分支的路径。

7. 加入 branch hysteresis。局部半径内的前沿会按无人机到前沿方向划分分支角区，一旦选中某个分支，会保持若干轮优先探索该分支，直到该分支没有候选或保持时间到期。这样岔路口不会每轮在多个分支之间来回跳。

8. 扩展日志字段。`local_view_log.csv` 新增 `branch_id` 和 `active_branch_id`，用于后续统计岔路口分支切换次数；`start_cost_log.csv` 和 `frontier_track_log.csv` 保留 debt、coverage、progress、cooldown 等字段，方便下一轮定位是否还存在局部最优。

## 当前效果

代码已重新编译通过，说明本次修改不会因为语法或链接问题导致程序无法启动。编译过程中只出现原工程已有的 unused variable、signedness 和格式化 warning，没有新的编译错误。

由于本次修改后还没有重新运行完整探索，真实探索效果需要下一轮 `Data_debug` 数据验证。预期效果是：`forget_reason=4` 的 debt-only 假遗忘应降为 0；局部第一目标中 `selected_reason=2/3` 的占比应明显下降；同一前沿连续排第一但 `progress_delta=0` 的次数应减少；岔路口处 `active_branch_id` 应保持数轮稳定，而不是每轮切换。

## 下一轮重点观察

重新运行探索后，优先检查四个指标：第一，探索是否能走出岔路口并继续推进；第二，`local_view_log.csv` 中每轮第一目标的 `branch_id` 是否频繁跳变；第三，`forget_reason=4` 是否消失；第四，同一 `sig` 的 `local_pass_cnt` 和 `attempt_cnt` 是否不再异常增长。如果仍有远距离回头，再根据新的 `branch_id`、`active_branch_id` 和 tail overlap 数据判断是分支保持不足，还是全局 ATSP 仍然缺少区域完成度约束。

# Environment Rrepresentation
<!-- - dynamic obstacle predictor: use previous observation to predict a short range trajectory (distribution).  -->
<!-- - also implement sdf in a more compact way. -->
<!-- - Env: SDFMap + ObjPrediction, evaluate EDT -->
<!-- - refine evaluateEDT -->
<!-- - kinodynamic astar using Env -->
<!-- - hash table have shits. -->
- bspline optimization using. Env

- dealing with uncertainty. -wenchao paper

- refactor pcd path finder

- pnp


ld*(-4*p2**2*t1 + 4*p2**2*t2 - 12*p2*p3*t1**2 + 12*p2*p3*t2**2 - 80*p4*p5*t1**6 + 80*p4*p5*t2**6 - 400*p5**2*t1**7/7 + 400*p5**2*t2**7/7 - t1**5*(48*p3*p5 + 144*p4**2/5) - t1**4*(20*p2*p5 + 36*p3*p4) - t1**3*(16*p2*p4 + 12*p3**2) + t2**5*(48*p3*p5 + 144*p4**2/5) + t2**4*(20*p2*p5 + 36*p3*p4) + t2**3*(16*p2*p4 + 12*p3**2)) + p0**2 + 2*p0*p1*ti + 2*p0*p2*ti**2 + 2*p0*p3*ti**3 + 2*p0*p4*ti**4 + 2*p0*p5*ti**5 - 2*p0*qi + p1**2*ti**2 + 2*p1*p2*ti**3 + 2*p1*p3*ti**4 + 2*p1*p4*ti**5 + 2*p1*p5*ti**6 - 2*p1*qi*ti + p2**2*ti**4 + 2*p2*p3*ti**5 + 2*p2*p4*ti**6 + 2*p2*p5*ti**7 - 2*p2*qi*ti**2 + p3**2*ti**6 + 2*p3*p4*ti**7 + 2*p3*p5*ti**8 - 2*p3*qi*ti**3 + p4**2*ti**8 + 2*p4*p5*ti**9 - 2*p4*qi*ti**4 + p5**2*ti**10 - 2*p5*qi*ti**5 + qi**2

2*p0 + 2*p1*ti + 2*p2*ti**2 + 2*p3*ti**3 + 2*p4*ti**4 + 2*p5*ti**5 - 2*qi

2*ti*(p0 + p1*ti + p2*ti**2 + p3*ti**3 + p4*ti**4 + p5*ti**5 - qi)

-4*ld*(2*p2*t1 - 2*p2*t2 + 3*p3*t1**2 - 3*p3*t2**2 + 4*p4*t1**3 - 4*p4*t2**3 + 5*p5*t1**4 - 5*p5*t2**4) + 2*p0*ti**2 + 2*p1*ti**3 + 2*p2*ti**4 + 2*p3*ti**5 + 2*p4*ti**6 + 2*p5*ti**7 - 2*qi*ti**2

-12*ld*(p2*t1**2 - p2*t2**2 + 2*p3*t1**3 - 2*p3*t2**3 + 3*p4*t1**4 - 3*p4*t2**4 + 4*p5*t1**5 - 4*p5*t2**5) + 2*p0*ti**3 + 2*p1*ti**4 + 2*p2*ti**5 + 2*p3*ti**6 + 2*p4*ti**7 + 2*p5*ti**8 - 2*qi*ti**3

-4*ld*(20*p2*t1**3 - 20*p2*t2**3 + 45*p3*t1**4 - 45*p3*t2**4 + 72*p4*t1**5 - 72*p4*t2**5 + 100*p5*t1**6 - 100*p5*t2**6)/5 + 2*p0*ti**4 + 2*p1*ti**5 + 2*p2*ti**6 + 2*p3*ti**7 + 2*p4*ti**8 + 2*p5*ti**9 - 2*qi*ti**4

-4*ld*(35*p2*t1**4 - 35*p2*t2**4 + 84*p3*t1**5 - 84*p3*t2**5 + 140*p4*t1**6 - 140*p4*t2**6 + 200*p5*t1**7 - 200*p5*t2**7)/7 + 2*p0*ti**5 + 2*p1*ti**6 + 2*p2*ti**7 + 2*p3*ti**8 + 2*p4*ti**9 + 2*p5*ti**10 - 2*qi*ti**5

#!/bin/bash

source ~/app/bashrc.sh;
echo "dji" | sudo -S chmod 777 /dev/ttyUSB0
# dji ros
taskset -c 1 roslaunch djiros djiros.launch & sleep 5;

# realsense
taskset -c 1 roslaunch realsense2_camera rs_camera.launch & sleep 4;
rosrun dynamic_reconfigure dynparam set /camera/Stereo_Module 'Emitter Enabled' false & sleep 1;

# vins localization
taskset -c 2-5 rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/newcalibration/realsense_stereo_imu_config.yaml & sleep 1;
# rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/VINS-Fusion/config/newcalibration/realsense_stereo_imu_config.yaml & sleep 1;

# local mapping and planning 
taskset -c 6-7 roslaunch plan_manage simulation.launch & sleep 2;

taskset -c 1 roslaunch machine_defined ctrl_md.launch


#!/bin/bash
echo "dji" | sudo -S cpufreq-set -g performance
source /home/dji/app/bashrc.sh;
roscore &
#sleep 2; #ssh -t ubuntu@tegra1 "echo 1 | sudo -S date --set \"$(date -u)\"" &
#sleep 1; #ssh -t ubuntu@tegra2 "echo 1 | sudo -S date --set \"$(date -u)\"" &
#sleep 1; taskset -c 0 roslaunch vins vins_rviz.launch &
sleep 1; taskset -c 0 roslaunch plan_manage rviz.launch &
exec bash;

add yaw control;
use fastest straight line trajectory;
add ceil to avoid flying high;


