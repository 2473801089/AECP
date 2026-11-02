#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace fast_planner
{
  class EDTEnvironment;     // EDT环境：提供距离场查询接口
  class SDFMap;             // 占据栅格地图：提供地图更新和查询接口
  class FastPlannerManager; // 规划管理器：负责路径规划和轨迹优化
  class FrontierFinder;     // 前沿搜索器：负责从地图中提取前沿信息
  class GridPlanner;         // 网格规划器：双层均匀网格分解 + 全局TSP引导
  struct ExplorationParam;  // 探索参数配置
  struct ExplorationData;   // 探索数据结构：存储前沿信息、视点信息和规划结果等

  enum EXPL_RESULT
  {
    NO_FRONTIER, // 没有前沿可供探索
    FAIL,        // 规划失败
    SUCCEED      // 规划成功
  };

  class FastExplorationManager
  {
  public:
    FastExplorationManager();
    ~FastExplorationManager();

    void initialize(ros::NodeHandle &nh); // 初始化函数：设置参数、创建模块实例和ROS接口

    int planExploreMotion(const Vector3d &pos,
                          const Vector3d &vel,
                          const Vector3d &acc,
                          const Vector3d &yaw);

    // Benchmark method, classic frontier and rapid frontier
    int classicFrontier(const Vector3d &pos,
                        const double &yaw); // 经典前沿方法

    int rapidFrontier(const Vector3d &pos,
                      const Vector3d &vel,
                      const double &yaw,
                      bool &classic); // 快速前沿方法

    // RACER-style visualization accessors
    int getActiveGridId() const { return active_grid_id_; }
    int getActiveRootGridId() const { return active_root_grid_id_; }

    shared_ptr<ExplorationData> ed_;
    shared_ptr<ExplorationParam> ep_;
    shared_ptr<FastPlannerManager> planner_manager_;
    shared_ptr<FrontierFinder> frontier_finder_;
    shared_ptr<GridPlanner> grid_planner_;
    shared_ptr<SDFMap> getSDFMap() { return sdf_map_; }
    // unique_ptr<ViewFinder> view_finder_;

  private:
    shared_ptr<EDTEnvironment> edt_environment_;
    shared_ptr<SDFMap> sdf_map_;
    bool enable_grid_select_log_ = false;
    std::string grid_select_log_path_;
    std::ofstream grid_select_log_file_;
    int grid_select_log_round_ = 0;
    bool enable_local_tsp_cost_log_ = false;
    std::string local_tsp_cost_log_path_;
    std::ofstream local_tsp_cost_log_file_;
    int local_tsp_cost_log_round_ = 0;
    bool enable_decision_trace_log_ = false;
    std::string decision_trace_log_path_;
    std::ofstream decision_trace_log_file_;
    int decision_trace_log_round_ = 0;
    double local_tsp_block_cost_ = 5000.0;
    double local_tsp_path_sample_step_ = 0.2;
    double local_tsp_first_cost_ratio_ = 1.3;
    double local_tsp_first_cost_margin_ = 1.0;
    // [MODIFIED BY CODEX]: Local LKH is restricted by the candidate set, not
    // by a post-solve distance/node-count truncation.
    int local_tsp_candidate_num_ = 10;
    int grid_decision_round_ = 0;
    int active_grid_id_ = -1;
    int active_root_grid_id_ = -1;
    int active_root_complete_rounds_ = 0;
    int active_root_incomplete_rounds_ = 0;
    int root_unreachable_rounds_ = 0;
    int active_grid_hold_until_round_ = -1;
    int root_hold_until_round_ = -1;
    int last_lead_frontier_id_ = -1;
    int lead_frontier_repeat_rounds_ = 0;
    int last_selected_frontier_id_ = -1;
    int last_selected_frontier_grid_id_ = -1;
    int selected_frontier_hold_rounds_ = 0;
    int active_root_lock_rounds_ = 0;
    int soft_released_root_id_ = -1;
    int soft_release_until_round_ = -1;
    vector<int> pending_root_grid_ids_;
    int active_root_component_region_id_ = -1;
    int active_root_component_root_id_ = -1;
    int active_root_component_initial_frontier_count_ = 0;
    int active_root_component_last_frontier_count_ = 0;
    int active_root_component_last_update_round_ = 0;
    double root_component_release_ratio_ = 0.15;

    // Fine-grid local connected-component tracking.
    int active_component_grid_id_ = -1;
    int active_component_local_id_ = -1;
    int active_component_initial_frontier_count_ = 0;
    int active_component_last_frontier_count_ = 0;
    int active_component_last_update_round_ = 0;
    double active_component_release_ratio_ = 0.15;

    // Dual-threshold grid switching (Step 2): per-coarse-root tracking
    int active_root_initial_frontier_count_ = 0;
    double root_switch_frontier_ratio_ = 0.12;
    double root_switch_coverage_ratio_ = 0.88;

    // Timeout protection (Step 3)
    ros::Time active_root_enter_time_;
    bool active_root_timed_out_ = false;
    bool last_root_ready_ = false;  // set when root meets dual-threshold before switch
    double root_timeout_seconds_ = 138.0;

    // Batch pending cleanup (Step 5)
    int consecutive_completed_roots_ = 0;
    int batch_cleanup_interval_ = 3;
    vector<int> recent_path_grid_ids_;
    int recent_path_grid_limit_ = 8;
    int consecutive_fine_grid_cleanups_ = 0;
    int fine_grid_cleanup_interval_ = 3;
    int fine_grid_cleanup_hold_rounds_ = 0;
    double fine_grid_cleanup_cov_thresh_ = 0.90;

    Vector3d coarse_momentum_dir_ = Vector3d::Zero();    // movement direction in current root
    // Oscillation history: key=min(gid1,gid2)*10000+max(gid1,gid2), val=cooldown_round
    std::unordered_map<int64_t, int> osc_cooldown_map_;
    int osc_cooldown_rounds_ = 10;

    // Grid-based global TSP: compute tour through grid centers
    void findGlobalTourViaGrids(const Vector3d& cur_pos,
                                const Vector3d& cur_vel,
                                const Vector3d& cur_yaw,
                                const vector<int>& grid_ids,
                                const vector<Vector3d>& grid_centers,
                                vector<int>& tour_indices);

    // [MODIFIED BY CODEX]: Original-FUEL-style LKH solve for the local
    // frontier ATSP. Matrix node 0 is the current predicted UAV state.
    bool solveLocalTourViaLKH(const Eigen::MatrixXd& cost_mat,
                              vector<int>& tour_indices);

    // RACER-style local frontier TSP restricted by the grid coverage path
    // (Removed — simplified to inline 3-factor scoring in planExploreMotion)

    // Refine the LKH-ordered frontier viewpoints from the current UAV state.
    void refineLocalTour(const Vector3d &cur_pos,
                         const Vector3d &cur_vel,
                         const Vector3d &cur_yaw,
                         const vector<vector<Vector3d>> &n_points,
                         const vector<vector<double>> &n_yaws,
                         vector<Vector3d> &refined_pts,
                         vector<double> &refined_yaws);

    //路径剪枝：对规划得到的路径进行剪枝，去除冗余点，提高路径质量
    void shortenPath(vector<Vector3d> &path);

    // Component-aware path checks — removed (simplified to TSP-guided selection)
    bool pathStaysInGrid(int grid_id, const vector<Vector3d>& path) const;

  public:
    typedef shared_ptr<FastExplorationManager> Ptr;
  };

} // namespace fast_planner

#endif
