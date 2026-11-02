#ifndef _FRONTIER_FINDER_H_
#define _FRONTIER_FINDER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <deque>
#include <utility>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

using Eigen::Vector3d;
using std::deque;
using std::list;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class RayCaster;

namespace fast_planner
{
  class EDTEnvironment;
  class PerceptionUtils;

  struct Viewpoint
  {
    Vector3d pos_;
    double yaw_;
    int visib_num_;
    // Existing rear-ray depth used by the frontier-level missed-area logic.
    int unknown_num_ = 0;
    // Viewpoint-level N_unknown: visible unknown voxels behind its visible
    // frontier cells. This is kept separate from the frontier-level metric.
    int visible_unknown_num_ = 0;
  };

  struct Frontier
  {
    vector<Vector3d> cells_;
    vector<Vector3d> filtered_cells_;
    Vector3d average_;
    int id_ = -1;
    vector<Viewpoint> viewpoints_;
    Vector3d box_min_, box_max_;
    list<vector<Vector3d>> paths_;
    list<double> costs_;

    double concavity_ = 0.0;
    double narrowness_ = 0.0;
    double major_scale_ = 0.0;
    double minor_scale_ = 0.0;
    double obstacle_anisotropy_ = 0.0;
    double obstacle_density_ = 0.0;

    double adaptive_pos_score_ = 0.0;
    double adaptive_obs_score_ = 0.0;
    double adaptive_size_score_ = 0.0;
    double adaptive_path_score_ = 0.0;
    double adaptive_geom_score_ = 0.0;
    double adaptive_forget_score_ = 0.0;

    double last_start_travel_cost_ = 0.0;
    double last_start_final_cost_ = 0.0;
    double last_path_length_ = 0.0; // 纯A*路径长度(米), 不含yaw
    int last_start_path_size_ = 0;

    bool ever_touched_ = false; // 持久：一旦cov>0.03就永久标记
    bool forget_flag_ = false;
    double stale_fraction_ = 0.0;
    double coverage_overlap_ = 0.0;
    double filtered_coverage_overlap_ = 0.0;
    double debt_score_ = 0.0;
    double progress_delta_ = 0.0;
    double path_overlap_ = 0.0;
    int forget_reason_ = 0;
    int region_id_ = -1;

    std::string stable_sig_;
    int persist_cnt_ = 0;
    int first_seen_round_ = 0;
    int last_seen_round_ = 0;
    bool in_local_band_ = false;
    int local_pass_cnt_ = 0;
    int attempt_cnt_ = 0;
    int cooldown_until_round_ = 0;
    // ==================== Adaptive voxel / bounded unknown pocket ====================
    int back_unknown_num_ = 0;
    double back_obstacle_ratio_ = 0.0;
    double back_leak_ratio_ = 0.0;
    double bounded_pocket_score_ = 0.0;
    bool bounded_pocket_flag_ = false;

    double persistent_coverage_overlap_ = 0.0;
    double closeout_overlap_ = 0.0;
    double adaptive_voxel_score_ = 0.0;
    bool invalid_closeout_ = false;
    int tsp_layer_ = 3;
    int closeout_type_ = 0;

    double min_pass_dist_ = 1e9;      // 历史上无人机离该前沿最近距离
    int near_pass_cnt_ = 0;           // 进入传感器范围次数
    int last_near_round_ = 0;         // 最近一次靠近轮次
    double pass_debt_score_ = 0.0;    // 基于历史近距离的残留得分
  };

  struct ExploreRegion
  {
    int id_;
    Vector3d center_;
    Vector3d min_;
    Vector3d max_;
    double completion_;
    double unfilled_cost_;
  };

  class FrontierFinder
  {
  public:
    FrontierFinder(const shared_ptr<EDTEnvironment> &edt, ros::NodeHandle &nh);
    ~FrontierFinder();

    void searchFrontiers();
    void computeFrontiersToVisit();
    void getFrontiers(vector<vector<Vector3d>> &clusters);
    void getDormantFrontiers(vector<vector<Vector3d>> &clusters);
    void getFrontierBoxes(vector<pair<Vector3d, Vector3d>> &boxes);
    void getTopViewpointsInfo(const Vector3d &cur_pos, vector<Vector3d> &points,
                              vector<double> &yaws, vector<Vector3d> &averages);
    void getViewpointsInfo(const Vector3d &cur_pos, const double &cur_yaw,
                           const vector<int> &ids,
                           const int &view_num, const double &max_decay,
                           vector<vector<Vector3d>> &points, vector<vector<double>> &yaws);
    void updateFrontierCostMatrix();
    bool getFrontierPairCost(int from_id, int to_id, double &cost,
                             vector<Vector3d> &path) const;
    void refreshFrontierTrackingForPlanning(const Vector3d &cur_pos,
                                            const Vector3d cur_yaw);
    void getPathForTour(const Vector3d &pos, const vector<int> &frontier_ids,
                        vector<Vector3d> &path);

    void setNextFrontier(const int &id);
    bool isFrontierCovered();
    // Check only the committed frontier cluster.  The global overload above is
    // kept for the original FUEL code paths; rolling execution must not be
    // interrupted merely because another visible frontier changes.
    void wrapYaw(double &yaw);

    shared_ptr<PerceptionUtils> percep_utils_;
    ros::Publisher forget_vis_pub_;
    ros::Publisher local_order_vis_pub_;

    void setGridPlanningLogContext(int frontier_stage, bool root_lock,
                                   int root_unreachable_rounds,
                                   int active_grid_id, int active_root_grid_id);
    // [MODIFIED BY CODEX]: Manager supplies post-LKH prefix metadata so the
    // viewpoint log records which nodes actually enter Dijkstra.
    void setLocalViewRefinementLogContext(
        const vector<int>& lkh_ids, const std::unordered_map<int, double>& start_path_lengths,
        const std::unordered_map<int, int>& dijkstra_orders,
        const std::unordered_map<int, std::string>& dijkstra_reasons);
    void computeForgetFlags();

    bool selectBestCurrentCloseoutViewpoint(const Frontier &f, const Eigen::Vector3d &cur_pos, Viewpoint &best_view);

    Frontier *getFrontierById(int id);

    double computePersistentCoverageOverlap(const Frontier &frontier) const;
    void updateFrontierTrackingStats(const Vector3d &cur_pos);
    void writeFrontierTrackingLogOnly(const Vector3d &cur_pos);
    void updateCoverage(const Vector3d &cur_pos, const Vector3d &cur_yaw);

  private:
    void splitLargeFrontiers(list<Frontier> &frontiers);
    bool splitHorizontally(const Frontier &frontier, list<Frontier> &splits);
    void mergeFrontiers(Frontier &ftr1, const Frontier &ftr2);
    bool isFrontierChanged(const Frontier &ft);
    bool haveOverlap(const Vector3d &min1, const Vector3d &max1, const Vector3d &min2,
                     const Vector3d &max2);
    void computeFrontierInfo(Frontier &frontier);

    std::string makeFrontierSignature(const Frontier &frontier) const;
    void updateFrontierTrackingLog(const Vector3d &cur_pos);
    double computeCoverageOverlap(const Frontier &frontier) const;
    double computeFilteredCoverageOverlap(const Frontier &frontier) const;
    double computePathOverlap(const vector<Vector3d> &path) const;
    double computeStaleFraction(const Frontier &frontier);
    void markCoverageSegment(const Vector3d &p0, const Vector3d &p1, int stamp_round);
    void publishLocalExploreOrderViz(const Vector3d &cur_pos,
                                     const vector<int> &ordered_ids);

    void downsample(const vector<Vector3d> &cluster_in, vector<Vector3d> &cluster_out);
    void sampleViewpoints(Frontier &frontier);
    int countVisibleCells(const Vector3d &pos, const double &yaw, const vector<Vector3d> &cluster);
    void countVisibleFrontierAndUnknown(const Vector3d &pos, const double &yaw,
                                        const vector<Vector3d> &cluster, int &frontier_num,
                                        int &visible_unknown_num);
    bool isNearUnknown(const Vector3d &pos);
    vector<Eigen::Vector3i> sixNeighbors(const Eigen::Vector3i &voxel);
    vector<Eigen::Vector3i> tenNeighbors(const Eigen::Vector3i &voxel);
    vector<Eigen::Vector3i> allNeighbors(const Eigen::Vector3i &voxel);
    bool isNeighborUnknown(const Eigen::Vector3i &voxel);
    void expandFrontier(const Eigen::Vector3i &first);
    int toadr(const Eigen::Vector3i &idx) const;
    bool knownfree(const Eigen::Vector3i &idx);
    bool inmap(const Eigen::Vector3i &idx);
    Eigen::Vector3i searchClearVoxel(const Eigen::Vector3i &pt);
    bool isInBoxes(const vector<pair<Vector3d, Vector3d>> &boxes, const Eigen::Vector3i &idx);
    bool canBeMerged(const Frontier &ftr1, const Frontier &ftr2);
    void findViewpoints(const Vector3d &sample, const Vector3d &ftr_avg, vector<Viewpoint> &vps);

    // ==================== Adaptive voxel forgetting metrics ====================
    double computeSpatialPositionScore(const Frontier &f,const Eigen::Vector3d &cur_pos) const;
    double computeObservationDeficitScore(const Frontier &f) const;
    double computeVoxelClusterSizeScore(const Frontier &f) const;
    double computeDistanceToRecentPath(const Eigen::Vector3d &pos) const;
    double computePathGapScore(const Frontier &f) const;
    double computeGeometryFeatureScore(const Frontier &f) const;
    void computeBackLeakPocketEvidence(Frontier &f);
    double computeAdaptiveVoxelForgetScore(const Frontier &f,const Eigen::Vector3d &cur_pos) const;
    double computePassDebtScore(
        Frontier& f,
        const Eigen::Vector3d& cur_pos);

    vector<char> frontier_flag_;
    vector<char> coverage_flag_;

    vector<int> coverage_last_round_;
    int coverage_round_ = 0;
    int coverage_memory_rounds_ = 12;
    bool coverage_mark_only_new_segment_ = true;

    deque<Vector3d> path_history_;
    static constexpr int PATH_HISTORY_MAX = 50;
    double inflate_radius_;

    list<Frontier> frontiers_, dormant_frontiers_, tmp_frontiers_;
    vector<int> removed_ids_;
    list<Frontier>::iterator first_new_ftr_;
    Frontier next_frontier_;
    std::vector<char> coverage_ever_flag_;

    double adaptive_voxel_thresh_;
    int cluster_min_;
    double cluster_size_xy_, cluster_size_z_;
    double candidate_rmax_, candidate_rmin_, candidate_dphi_, min_candidate_dist_,
        min_candidate_clearance_;
    int down_sample_;
    double min_view_finish_fraction_, resolution_;
    double feature_radius_;
    int feature_dir_num_;

    int log_frontier_stage_ = -1;
    bool log_root_lock_ = false;
    int log_root_unreachable_rounds_ = 0;
    int log_active_grid_id_ = -1;
    int log_active_root_grid_id_ = -1;

    int purge_min_cells_;
    double forget_coverage_thresh_;
    double regret_lambda1_, regret_lambda2_;
    // [MODIFIED BY CODEX]: 结构评价主开关，消融实验用；true=启用(默认)，false=关闭整个结构评价机制
    bool enable_structure_eval_ = true;
    // [ABLATION]: 受限结构评分中后向射线项权重 alpha（式(11) Cr = alpha*ray + (1-alpha)*density）。
    // 默认 0.7，与改动前硬编码的 0.70/0.30 完全一致；1.0=仅后向射线，0.0=仅邻域障碍密度。
    double structure_alpha_ = 0.7;
    double local_closeout_radius_;
    bool force_local_sweep_;
    int local_sweep_min_frontiers_;
    int local_sweep_max_frontiers_;
    int local_sweep_hold_rounds_;
    double local_sweep_max_path_length_;
    double local_sweep_max_path_stretch_;
    double local_sweep_max_yaw_jump_;
    double local_sweep_nonlocal_penalty_;
    double local_sweep_bonus_max_;
    bool force_local_cover_all_;
    double local_corner_boost_;
    double local_pocket_boost_;
    int local_target_hold_rounds_;
    double local_switch_margin_;
    double local_repeat_path_penalty_;
    double local_repeat_pass_penalty_;
    int local_strong_hold_rounds_;
    double local_strong_switch_margin_;
    bool enable_local_branch_lock_;
    double local_branch_entry_gap_;
    int local_branch_hold_rounds_;
    double local_branch_switch_margin_;
    double local_min_wall_clearance_;
    bool enable_global_chain_bias_;
    double global_chain_bonus_gain_;
    double global_chain_gap_gate_;
    int global_target_hold_rounds_;
    double global_switch_margin_;
    double global_pass_pocket_bonus_gain_;
    double global_pass_pocket_gap_gate_;
    double global_pass_pocket_near_radius_;
    double global_pass_pocket_max_overlap_;
    double global_pass_pocket_min_debt_;
    double global_pass_pocket_min_bounded_;
    bool enable_global_boundary_bias_;
    double global_boundary_bonus_gain_;
    double global_boundary_band_;
    double global_boundary_gap_gate_;
    double global_boundary_max_overlap_;
    double global_boundary_min_score_;
    int global_boundary_side_hold_rounds_;
    double global_boundary_side_switch_margin_;
    double global_boundary_cross_penalty_;
    double global_boundary_nonboundary_penalty_;
    double global_boundary_same_side_bonus_;
    bool enable_global_local_completion_gate_;
    double global_local_completion_radius_;
    double global_local_completion_gap_gate_;
    double global_local_completion_overlap_max_;
    int global_local_completion_min_count_;
    int global_local_completion_max_cells_;
    bool global_local_completion_allow_single_;
    double global_local_completion_single_score_gate_;
    int global_local_completion_single_pass_gate_;
    double global_local_completion_bonus_;
    double global_local_completion_penalty_;
    int local_sweep_until_round_ = 0;
    // TSP加法代价权重
    double cooldown_progress_thresh_, closeout_min_view_dist_;
    int cooldown_rounds_;

    // 自适应体素遗忘机制
    bool enable_frontier_track_log_;
    std::string frontier_track_log_path_;
    std::ofstream frontier_track_log_file_;
    int frontier_track_round_;
    std::unordered_map<std::string, int> sig_persist_cnt_;
    std::unordered_map<std::string, int> sig_first_seen_round_;
    std::unordered_map<std::string, int> sig_last_seen_round_;
    std::unordered_map<std::string, int> sig_prev_cell_num_;
    std::unordered_map<std::string, int> sig_local_pass_cnt_;
    std::unordered_map<std::string, int> sig_attempt_cnt_;
    std::unordered_map<std::string, int> sig_cooldown_until_round_;
    std::unordered_map<std::string, int> sig_last_selected_round_;
    std::string last_local_target_sig_;
    int last_local_target_round_ = -1000000;
    int last_local_target_id_ = -1;
    Eigen::Vector3d last_local_target_entry_ = Eigen::Vector3d::Zero();
    bool has_last_local_target_entry_ = false;
    int active_local_branch_id_ = -1;
    int active_local_branch_until_round_ = -1000000;
    std::string last_global_target_sig_;
    int last_global_target_round_ = -1000000;
    int active_global_boundary_side_ = -1;
    int active_global_boundary_until_round_ = -1000000;

    // 历史近距离经过记录：用于判断“无人机曾经靠近但前沿仍未清理”
    std::unordered_map<std::string, double> sig_min_pass_dist_;
    std::unordered_map<std::string, int> sig_near_pass_cnt_;
    std::unordered_map<std::string, int> sig_last_near_round_;

    bool enable_local_view_log_;
    std::string local_view_log_path_;
    std::ofstream local_view_log_file_;
    std::string local_view_lkh_sequence_;
    std::unordered_map<int, double> local_view_start_path_lengths_;
    std::unordered_map<int, int> local_view_dijkstra_orders_;
    std::unordered_map<int, std::string> local_view_dijkstra_reasons_;

    int min_visib_num_, candidate_rnum_;
    shared_ptr<EDTEnvironment> edt_env_;
    unique_ptr<RayCaster> raycaster_;
  };

} // namespace fast_planner
#endif
