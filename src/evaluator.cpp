/*
 * RAS 2026 — Fast Stress Score Evaluator + SA with RL-guided destroy
 *
 * Key design: O(1) delta evaluation via maintained block state.
 * Each SA step only touches the affected blocks, not all demands.
 *
 * Build (Colab):
 *   pip install pybind11
 *   g++ -O3 -shared -fPIC $(python3 -m pybind11 --includes) \
 *       src/evaluator.cpp -o src/evaluator$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <random>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <string>
#include <limits>

namespace py = pybind11;

// ── Data types ────────────────────────────────────────────────────────────────

struct Demand {
    int    idx;
    int    origin, dest;
    int    volume;
    double sp_dist;
    int    commodity;   // 0=Manifest,1=Bulk,2=Intermodal,3=Multilevel
    bool   direct_only; // Intermodal / Automobile
};

// A route is a sequence of (from, to) segments.
// route_idx = 0       → unserved
// route_idx = 1       → direct (origin→dest)
// route_idx = 2..N+1  → via hub[route_idx-2]
struct CandidateRoute {
    bool   is_unserved;
    bool   is_direct;
    int    hub;             // -1 if direct/unserved
    double transport_cost;  // volume-weighted transport + interchange cost
    double handling_cost;   // volume-weighted handling at hub
    // segments for block-key lookup
    int seg1_o, seg1_d;     // first segment (or only)
    int seg2_o, seg2_d;     // second segment (-1 if direct/unserved)
};

struct Settings {
    double block_fixed_cost;
    double unserved_penalty;  // M * sp_dist already multiplied in
    double min_block_vol_short;   // <100mi
    double min_block_vol_medium;  // 100-500mi
    double min_block_vol_long;    // >500mi
};

// Block key: (from_yard, to_yard, commodity)
struct BlockKey {
    int from, to, commodity;
    bool operator==(const BlockKey& o) const {
        return from==o.from && to==o.to && commodity==o.commodity;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& k) const {
        size_t h = std::hash<int>{}(k.from);
        h ^= std::hash<int>{}(k.to)   + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.commodity) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};

struct BlockState {
    double total_volume = 0;
    double link_dist    = 0;  // distance of this block's link (for min_vol check)
    bool   is_active    = false;
};

// ── Solver state ──────────────────────────────────────────────────────────────

class RasSolver {
public:
    std::vector<Demand>                          demands;
    std::vector<std::vector<CandidateRoute>>     candidates;  // [demand_idx][route_idx]
    std::vector<int>                             current_route; // route_idx per demand
    std::unordered_map<BlockKey, BlockState, BlockKeyHash> blocks;
    Settings                                     settings;

    // Track capacity: yard → outbound manifest/bulk block count
    std::unordered_map<int, int>    yard_tracks;      // yard → max tracks
    std::unordered_map<int, double> yard_handling_cap;
    std::unordered_map<int, double> block_dist;       // (from,to) → dist (for min vol)

    double current_score = 0.0;

    // ── Init ──────────────────────────────────────────────────────────────────

    void init(
        const std::vector<std::tuple<int,int,int,int,double,int,bool>>& dem_list,
        const std::vector<std::vector<std::tuple<bool,bool,int,double,double,int,int,int,int>>>& cand_list,
        const std::vector<int>& init_routes,
        const std::unordered_map<int,int>& yd_tracks,
        const std::unordered_map<int,double>& yd_hcap,
        const std::unordered_map<std::string,double>& seg_dist,
        Settings s
    ) {
        settings = s;
        yard_tracks = yd_tracks;
        yard_handling_cap = yd_hcap;

        // Parse demands
        demands.clear();
        for (auto& [idx,o,d,v,sp,com,donly] : dem_list) {
            demands.push_back({idx,o,d,v,sp,com,donly});
        }

        // Parse candidates
        candidates.resize(demands.size());
        for (size_t i = 0; i < cand_list.size(); i++) {
            candidates[i].clear();
            for (auto& [unserved,direct,hub,tc,hc,s1o,s1d,s2o,s2d] : cand_list[i]) {
                candidates[i].push_back({unserved,direct,hub,tc,hc,s1o,s1d,s2o,s2d});
            }
        }

        // Parse segment distances
        for (auto& [k, v] : seg_dist) {
            // key format: "from_to"
            auto pos = k.find('_');
            int f = std::stoi(k.substr(0, pos));
            int t = std::stoi(k.substr(pos+1));
            block_dist[f*100000+t] = v;
        }

        // Build initial block state
        current_route = init_routes;
        rebuild_blocks();
        current_score = compute_score();
    }

    // ── Block management ──────────────────────────────────────────────────────

    double get_seg_dist(int f, int t) const {
        auto it = block_dist.find(f*100000+t);
        return it != block_dist.end() ? it->second : 500.0;
    }

    double min_vol_for_dist(double dist) const {
        if (dist < 100) return settings.min_block_vol_short;
        if (dist < 500) return settings.min_block_vol_medium;
        return settings.min_block_vol_long;
    }

    void add_demand_to_blocks(int di, int ri) {
        const auto& dem = demands[di];
        const auto& cr  = candidates[di][ri];
        if (cr.is_unserved) return;

        // Segment 1
        BlockKey k1{cr.seg1_o, cr.seg1_d, dem.commodity};
        blocks[k1].total_volume += dem.volume;
        blocks[k1].link_dist = get_seg_dist(cr.seg1_o, cr.seg1_d);

        // Segment 2
        if (!cr.is_direct) {
            BlockKey k2{cr.seg2_o, cr.seg2_d, dem.commodity};
            blocks[k2].total_volume += dem.volume;
            blocks[k2].link_dist = get_seg_dist(cr.seg2_o, cr.seg2_d);
        }
    }

    void remove_demand_from_blocks(int di, int ri) {
        const auto& dem = demands[di];
        const auto& cr  = candidates[di][ri];
        if (cr.is_unserved) return;

        BlockKey k1{cr.seg1_o, cr.seg1_d, dem.commodity};
        blocks[k1].total_volume -= dem.volume;
        if (blocks[k1].total_volume <= 0) blocks.erase(k1);

        if (!cr.is_direct) {
            BlockKey k2{cr.seg2_o, cr.seg2_d, dem.commodity};
            blocks[k2].total_volume -= dem.volume;
            if (blocks[k2].total_volume <= 0) blocks.erase(k2);
        }
    }

    void rebuild_blocks() {
        blocks.clear();
        for (size_t i = 0; i < demands.size(); i++) {
            add_demand_to_blocks(i, current_route[i]);
        }
    }

    // ── Score computation ─────────────────────────────────────────────────────

    double compute_score() const {
        double score = 0.0;

        // Per-demand: transport + interchange + handling + unserved penalty
        for (size_t i = 0; i < demands.size(); i++) {
            const auto& cr = candidates[i][current_route[i]];
            if (cr.is_unserved) {
                score += settings.unserved_penalty * demands[i].sp_dist * demands[i].volume;
            } else {
                score += cr.transport_cost + cr.handling_cost;
            }
        }

        // Per-block: fixed cost (only if volume >= min)
        for (auto& [k, b] : blocks) {
            double minvol = min_vol_for_dist(b.link_dist);
            if (b.total_volume >= minvol) {
                score += settings.block_fixed_cost;
            } else {
                // Under-minimum: treat as stress penalty (blocks shouldn't form)
                score += settings.block_fixed_cost * 10.0;
            }
        }

        return score;
    }

    // ── O(1) Delta evaluation ─────────────────────────────────────────────────

    double delta_score(int di, int new_ri) const {
        int old_ri = current_route[di];
        if (old_ri == new_ri) return 0.0;

        const auto& dem    = demands[di];
        const auto& old_cr = candidates[di][old_ri];
        const auto& new_cr = candidates[di][new_ri];

        double delta = 0.0;

        // --- Demand-level cost change ---
        double old_dem_cost = old_cr.is_unserved
            ? settings.unserved_penalty * dem.sp_dist * dem.volume
            : old_cr.transport_cost + old_cr.handling_cost;
        double new_dem_cost = new_cr.is_unserved
            ? settings.unserved_penalty * dem.sp_dist * dem.volume
            : new_cr.transport_cost + new_cr.handling_cost;
        delta += new_dem_cost - old_dem_cost;

        // --- Block fixed cost changes ---
        // Helper lambda: block cost contribution
        auto block_cost = [&](const BlockKey& k, double vol_change) -> double {
            auto it = blocks.find(k);
            double old_vol = (it != blocks.end()) ? it->second.total_volume : 0.0;
            double new_vol = old_vol + vol_change;
            double dist    = (it != blocks.end()) ? it->second.link_dist
                                                   : get_seg_dist(k.from, k.to);
            double minvol  = min_vol_for_dist(dist);

            double old_cost = (old_vol <= 0) ? 0.0
                            : (old_vol >= minvol ? settings.block_fixed_cost
                                                 : settings.block_fixed_cost * 10.0);
            double new_cost = (new_vol <= 0) ? 0.0
                            : (new_vol >= minvol ? settings.block_fixed_cost
                                                 : settings.block_fixed_cost * 10.0);
            return new_cost - old_cost;
        };

        // Remove old route contribution
        if (!old_cr.is_unserved) {
            BlockKey k1{old_cr.seg1_o, old_cr.seg1_d, dem.commodity};
            delta += block_cost(k1, -(double)dem.volume);
            if (!old_cr.is_direct) {
                BlockKey k2{old_cr.seg2_o, old_cr.seg2_d, dem.commodity};
                delta += block_cost(k2, -(double)dem.volume);
            }
        }

        // Add new route contribution
        if (!new_cr.is_unserved) {
            // Need temporary adjusted volumes for overlapping blocks
            BlockKey k1{new_cr.seg1_o, new_cr.seg1_d, dem.commodity};
            // Check if same as old seg (would have already been decremented)
            bool k1_same_as_old_s1 = !old_cr.is_unserved && !old_cr.is_direct == false
                && k1.from == old_cr.seg1_o && k1.to == old_cr.seg1_d;
            double adj1 = k1_same_as_old_s1 ? -(double)dem.volume : 0.0;

            auto it1 = blocks.find(k1);
            double vol1 = (it1 != blocks.end() ? it1->second.total_volume : 0.0) + adj1;
            double dist1 = (it1 != blocks.end()) ? it1->second.link_dist : get_seg_dist(k1.from, k1.to);
            double minvol1 = min_vol_for_dist(dist1);
            double old_c1 = vol1 <= 0 ? 0.0 : (vol1 >= minvol1 ? settings.block_fixed_cost : settings.block_fixed_cost*10.0);
            double new_c1_vol = vol1 + dem.volume;
            double new_c1 = new_c1_vol <= 0 ? 0.0 : (new_c1_vol >= minvol1 ? settings.block_fixed_cost : settings.block_fixed_cost*10.0);
            delta += new_c1 - old_c1;

            if (!new_cr.is_direct) {
                BlockKey k2{new_cr.seg2_o, new_cr.seg2_d, dem.commodity};
                delta += block_cost(k2, +(double)dem.volume);
            }
        }

        return delta;
    }

    // ── Apply move ────────────────────────────────────────────────────────────

    void apply_move(int di, int new_ri) {
        remove_demand_from_blocks(di, current_route[di]);
        add_demand_to_blocks(di, new_ri);
        current_route[di] = new_ri;
        // Note: current_score updated by caller
    }

    // ── SA with RL destroy weights ────────────────────────────────────────────

    struct SAResult {
        std::vector<int>    best_routes;
        double              best_score;
        std::vector<double> score_history;      // every log_every iters
        std::vector<int>    selected_demands;   // which demand was picked each iter
        std::vector<int>    selected_routes;    // which route was picked
        std::vector<double> rewards;            // delta improvement per iter
    };

    SAResult sa_run(
        const std::vector<float>& rl_weights,   // per-demand selection probability
        double T0, double T_final,
        int n_iter, int log_every,
        unsigned int seed
    ) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        // Normalize rl_weights to get sampling distribution
        std::vector<double> probs(demands.size());
        double total = 0.0;
        for (size_t i = 0; i < demands.size(); i++) {
            probs[i] = std::max(1e-6f, rl_weights[i]);
            total += probs[i];
        }
        for (auto& p : probs) p /= total;
        std::discrete_distribution<int> demand_dist(probs.begin(), probs.end());

        SAResult result;
        result.best_routes = current_route;
        result.best_score  = current_score;

        double T = T0;
        double alpha = std::pow(T_final / T0, 1.0 / n_iter);

        for (int it = 0; it < n_iter; it++) {
            // Pick demand (RL-guided)
            int di = demand_dist(rng);

            // Pick random candidate route
            int n_cands = candidates[di].size();
            std::uniform_int_distribution<int> route_dist(0, n_cands - 1);
            int new_ri = route_dist(rng);

            // Delta eval O(1)
            double delta = delta_score(di, new_ri);

            // SA acceptance
            bool accept = false;
            if (delta <= 0) {
                accept = true;
            } else {
                accept = uniform(rng) < std::exp(-delta / T);
            }

            double reward = 0.0;
            if (accept) {
                apply_move(di, new_ri);
                current_score += delta;
                reward = -delta;  // positive = improvement

                if (current_score < result.best_score) {
                    result.best_score  = current_score;
                    result.best_routes = current_route;
                }
            }

            result.selected_demands.push_back(di);
            result.selected_routes.push_back(new_ri);
            result.rewards.push_back(reward);

            if (it % log_every == 0) {
                result.score_history.push_back(current_score);
            }

            T *= alpha;
        }

        return result;
    }

    // ── Getters ───────────────────────────────────────────────────────────────

    double get_score() const { return current_score; }
    std::vector<int> get_routes() const { return current_route; }

    void set_routes(const std::vector<int>& routes) {
        current_route = routes;
        rebuild_blocks();
        current_score = compute_score();
    }
};

// ── pybind11 bindings ─────────────────────────────────────────────────────────

PYBIND11_MODULE(evaluator, m) {
    m.doc() = "RAS 2026 fast stress evaluator + SA with RL-guided destroy";

    py::class_<Settings>(m, "Settings")
        .def(py::init<>())
        .def_readwrite("block_fixed_cost",      &Settings::block_fixed_cost)
        .def_readwrite("unserved_penalty",       &Settings::unserved_penalty)
        .def_readwrite("min_block_vol_short",    &Settings::min_block_vol_short)
        .def_readwrite("min_block_vol_medium",   &Settings::min_block_vol_medium)
        .def_readwrite("min_block_vol_long",     &Settings::min_block_vol_long);

    py::class_<RasSolver>(m, "RasSolver")
        .def(py::init<>())
        .def("init",       &RasSolver::init)
        .def("sa_run",     &RasSolver::sa_run)
        .def("get_score",  &RasSolver::get_score)
        .def("get_routes", &RasSolver::get_routes)
        .def("set_routes", &RasSolver::set_routes)
        .def("delta_score",&RasSolver::delta_score);

    py::class_<RasSolver::SAResult>(m, "SAResult")
        .def_readonly("best_routes",       &RasSolver::SAResult::best_routes)
        .def_readonly("best_score",        &RasSolver::SAResult::best_score)
        .def_readonly("score_history",     &RasSolver::SAResult::score_history)
        .def_readonly("selected_demands",  &RasSolver::SAResult::selected_demands)
        .def_readonly("selected_routes",   &RasSolver::SAResult::selected_routes)
        .def_readonly("rewards",           &RasSolver::SAResult::rewards);
}
