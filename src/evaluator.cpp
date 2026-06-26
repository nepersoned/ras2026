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
    bool   direct_only;
};

// route_type:
//   0 = unserved
//   1 = direct          (1 segment: seg1)
//   2 = 1-hub           (2 segments: seg1, seg2)
//   3 = 2-hub           (3 segments: seg1, seg2, seg3)
struct CandidateRoute {
    bool   is_unserved;
    bool   is_direct;
    int    hub;              // first hub (-1 if direct/unserved)
    int    hub2;             // second hub (-1 if not 2-hub)
    double transport_cost;
    double handling_cost;
    int seg1_o, seg1_d;
    int seg2_o, seg2_d;     // -1,-1 if direct/unserved
    int seg3_o, seg3_d;     // -1,-1 if not 2-hub

    int n_segs() const {
        if (is_unserved) return 0;
        if (is_direct)   return 1;
        if (seg3_o >= 0) return 3;
        return 2;
    }
};

struct Settings {
    double block_fixed_cost;
    double unserved_penalty;
    double min_block_vol_short;
    double min_block_vol_medium;
    double min_block_vol_long;
};

struct BlockKey {
    int from, to, commodity;
    bool operator==(const BlockKey& o) const {
        return from==o.from && to==o.to && commodity==o.commodity;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& k) const {
        size_t h = std::hash<int>{}(k.from);
        h ^= std::hash<int>{}(k.to)        + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.commodity) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};

struct BlockState {
    double total_volume = 0;
    double link_dist    = 0;
};

// ── Solver ────────────────────────────────────────────────────────────────────

class RasSolver {
public:
    std::vector<Demand>                              demands;
    std::vector<std::vector<CandidateRoute>>         candidates;
    std::vector<int>                                 current_route;
    std::unordered_map<BlockKey, BlockState, BlockKeyHash> blocks;
    Settings                                         settings;

    std::unordered_map<int,int>    yard_tracks;
    std::unordered_map<int,double> yard_handling_cap;
    std::unordered_map<long long,double> block_dist;

    double current_score = 0.0;

    // ── Init ──────────────────────────────────────────────────────────────────
    //
    // Python passes candidates as tuples:
    //   1-hub / direct / unserved (9-tuple, legacy):
    //     (is_unserved, is_direct, hub, tc, hc, s1o, s1d, s2o, s2d)
    //   2-hub (11-tuple, new):
    //     (is_unserved, is_direct, hub, hub2, tc, hc, s1o, s1d, s2o, s2d, s3o, s3d)
    //
    // We accept both via py::object and check tuple size.

    void init(
        const std::vector<std::tuple<int,int,int,int,double,int,bool>>& dem_list,
        const py::list& cand_list_py,   // list of list of tuples (9 or 12 elements)
        const std::vector<int>& init_routes,
        const std::unordered_map<int,int>& yd_tracks,
        const std::unordered_map<int,double>& yd_hcap,
        const std::unordered_map<std::string,double>& seg_dist,
        Settings s
    ) {
        settings = s;
        yard_tracks = yd_tracks;
        yard_handling_cap = yd_hcap;

        demands.clear();
        for (auto& [idx,o,d,v,sp,com,donly] : dem_list)
            demands.push_back({idx,o,d,v,sp,com,donly});

        // Parse segment distances
        block_dist.clear();
        for (auto& [k, v] : seg_dist) {
            auto pos = k.find('_');
            long long f = std::stoi(k.substr(0, pos));
            long long t = std::stoi(k.substr(pos+1));
            block_dist[f*100000LL+t] = v;
        }

        // Parse candidates (accept 9-tuple or 12-tuple)
        size_t n = py::len(cand_list_py);
        candidates.resize(n);
        for (size_t i = 0; i < n; i++) {
            py::list dem_cands = cand_list_py[i].cast<py::list>();
            candidates[i].clear();
            for (auto item : dem_cands) {
                py::tuple t = item.cast<py::tuple>();
                size_t sz = py::len(t);
                CandidateRoute cr;
                cr.seg3_o = -1; cr.seg3_d = -1;
                cr.hub2   = -1;
                if (sz == 9) {
                    // legacy: (unserved, direct, hub, tc, hc, s1o, s1d, s2o, s2d)
                    cr.is_unserved    = t[0].cast<bool>();
                    cr.is_direct      = t[1].cast<bool>();
                    cr.hub            = t[2].cast<int>();
                    cr.transport_cost = t[3].cast<double>();
                    cr.handling_cost  = t[4].cast<double>();
                    cr.seg1_o         = t[5].cast<int>();
                    cr.seg1_d         = t[6].cast<int>();
                    cr.seg2_o         = t[7].cast<int>();
                    cr.seg2_d         = t[8].cast<int>();
                } else {
                    // new 12-tuple: (unserved, direct, hub, hub2, tc, hc, s1o,s1d, s2o,s2d, s3o,s3d)
                    cr.is_unserved    = t[0].cast<bool>();
                    cr.is_direct      = t[1].cast<bool>();
                    cr.hub            = t[2].cast<int>();
                    cr.hub2           = t[3].cast<int>();
                    cr.transport_cost = t[4].cast<double>();
                    cr.handling_cost  = t[5].cast<double>();
                    cr.seg1_o         = t[6].cast<int>();
                    cr.seg1_d         = t[7].cast<int>();
                    cr.seg2_o         = t[8].cast<int>();
                    cr.seg2_d         = t[9].cast<int>();
                    cr.seg3_o         = t[10].cast<int>();
                    cr.seg3_d         = t[11].cast<int>();
                }
                candidates[i].push_back(cr);
            }
        }

        current_route = init_routes;
        rebuild_blocks();
        current_score = compute_score();
    }

    // ── Block management ──────────────────────────────────────────────────────

    double get_seg_dist(int f, int t) const {
        auto it = block_dist.find((long long)f*100000LL+t);
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

        auto add_seg = [&](int fo, int to) {
            BlockKey k{fo, to, dem.commodity};
            blocks[k].total_volume += dem.volume;
            blocks[k].link_dist = get_seg_dist(fo, to);
        };

        add_seg(cr.seg1_o, cr.seg1_d);
        if (!cr.is_direct) {
            add_seg(cr.seg2_o, cr.seg2_d);
            if (cr.seg3_o >= 0)
                add_seg(cr.seg3_o, cr.seg3_d);
        }
    }

    void remove_demand_from_blocks(int di, int ri) {
        const auto& dem = demands[di];
        const auto& cr  = candidates[di][ri];
        if (cr.is_unserved) return;

        auto rem_seg = [&](int fo, int to) {
            BlockKey k{fo, to, dem.commodity};
            blocks[k].total_volume -= dem.volume;
            if (blocks[k].total_volume <= 0) blocks.erase(k);
        };

        rem_seg(cr.seg1_o, cr.seg1_d);
        if (!cr.is_direct) {
            rem_seg(cr.seg2_o, cr.seg2_d);
            if (cr.seg3_o >= 0)
                rem_seg(cr.seg3_o, cr.seg3_d);
        }
    }

    void rebuild_blocks() {
        blocks.clear();
        for (size_t i = 0; i < demands.size(); i++)
            add_demand_to_blocks(i, current_route[i]);
    }

    // ── Score computation ─────────────────────────────────────────────────────

    double compute_score() const {
        double score = 0.0;
        for (size_t i = 0; i < demands.size(); i++) {
            const auto& cr = candidates[i][current_route[i]];
            if (cr.is_unserved)
                score += settings.unserved_penalty * demands[i].sp_dist * demands[i].volume;
            else
                score += cr.transport_cost + cr.handling_cost;
        }
        for (auto& [k, b] : blocks) {
            double minvol = min_vol_for_dist(b.link_dist);
            score += (b.total_volume >= minvol)
                   ? settings.block_fixed_cost
                   : settings.block_fixed_cost * 10.0;
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

        // Demand-level cost change
        double old_cost = old_cr.is_unserved
            ? settings.unserved_penalty * dem.sp_dist * dem.volume
            : old_cr.transport_cost + old_cr.handling_cost;
        double new_cost = new_cr.is_unserved
            ? settings.unserved_penalty * dem.sp_dist * dem.volume
            : new_cr.transport_cost + new_cr.handling_cost;
        delta += new_cost - old_cost;

        // Collect old segments being removed
        struct Seg { int o, d; };
        std::vector<Seg> old_segs, new_segs;
        if (!old_cr.is_unserved) {
            old_segs.push_back({old_cr.seg1_o, old_cr.seg1_d});
            if (!old_cr.is_direct) {
                old_segs.push_back({old_cr.seg2_o, old_cr.seg2_d});
                if (old_cr.seg3_o >= 0)
                    old_segs.push_back({old_cr.seg3_o, old_cr.seg3_d});
            }
        }
        if (!new_cr.is_unserved) {
            new_segs.push_back({new_cr.seg1_o, new_cr.seg1_d});
            if (!new_cr.is_direct) {
                new_segs.push_back({new_cr.seg2_o, new_cr.seg2_d});
                if (new_cr.seg3_o >= 0)
                    new_segs.push_back({new_cr.seg3_o, new_cr.seg3_d});
            }
        }

        // Build temporary volume adjustments for shared segments
        // Use a small map: key = (o*100000+d) → net vol change so far
        std::unordered_map<long long, double> vol_adj;

        // Helper: block cost given current blocks + vol_adj
        auto blk_cost = [&](int fo, int to, double dv) -> double {
            BlockKey k{fo, to, dem.commodity};
            long long kk = (long long)fo*100000LL+to;
            auto it = blocks.find(k);
            double base_vol = it != blocks.end() ? it->second.total_volume : 0.0;
            double adj = 0.0;
            auto adj_it = vol_adj.find(kk);
            if (adj_it != vol_adj.end()) adj = adj_it->second;
            double dist = it != blocks.end() ? it->second.link_dist : get_seg_dist(fo, to);
            double minvol = min_vol_for_dist(dist);

            double v0 = base_vol + adj;
            double v1 = v0 + dv;
            double c0 = v0 <= 0 ? 0.0 : (v0 >= minvol ? settings.block_fixed_cost : settings.block_fixed_cost*10.0);
            double c1 = v1 <= 0 ? 0.0 : (v1 >= minvol ? settings.block_fixed_cost : settings.block_fixed_cost*10.0);
            vol_adj[kk] = adj + dv;
            return c1 - c0;
        };

        // Remove old segments
        for (auto& seg : old_segs)
            delta += blk_cost(seg.o, seg.d, -(double)dem.volume);

        // Add new segments
        for (auto& seg : new_segs)
            delta += blk_cost(seg.o, seg.d, +(double)dem.volume);

        return delta;
    }

    // ── Apply move ────────────────────────────────────────────────────────────

    void apply_move(int di, int new_ri) {
        remove_demand_from_blocks(di, current_route[di]);
        add_demand_to_blocks(di, new_ri);
        current_route[di] = new_ri;
    }

    // ── SA ────────────────────────────────────────────────────────────────────

    struct SAResult {
        std::vector<int>    best_routes;
        double              best_score;
        std::vector<double> score_history;
        std::vector<int>    selected_demands;
        std::vector<int>    selected_routes;
        std::vector<double> rewards;
    };

    SAResult sa_run(
        const std::vector<float>& rl_weights,
        double T0, double T_final,
        int n_iter, int log_every,
        unsigned int seed
    ) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

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

        double T     = T0;
        double alpha = std::pow(T_final / T0, 1.0 / n_iter);

        for (int it = 0; it < n_iter; it++) {
            int di = demand_dist(rng);
            int n_cands = candidates[di].size();
            std::uniform_int_distribution<int> route_dist(0, n_cands - 1);
            int new_ri = route_dist(rng);

            double delta = delta_score(di, new_ri);

            bool accept = (delta <= 0) || (uniform(rng) < std::exp(-delta / T));
            double reward = 0.0;
            if (accept) {
                apply_move(di, new_ri);
                current_score += delta;
                reward = -delta;
                if (current_score < result.best_score) {
                    result.best_score  = current_score;
                    result.best_routes = current_route;
                }
            }

            result.selected_demands.push_back(di);
            result.selected_routes.push_back(new_ri);
            result.rewards.push_back(reward);
            if (it % log_every == 0)
                result.score_history.push_back(current_score);

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
    m.doc() = "RAS 2026 fast stress evaluator + SA (supports 2-hub 3-segment routes)";

    py::class_<Settings>(m, "Settings")
        .def(py::init<>())
        .def_readwrite("block_fixed_cost",      &Settings::block_fixed_cost)
        .def_readwrite("unserved_penalty",       &Settings::unserved_penalty)
        .def_readwrite("min_block_vol_short",    &Settings::min_block_vol_short)
        .def_readwrite("min_block_vol_medium",   &Settings::min_block_vol_medium)
        .def_readwrite("min_block_vol_long",     &Settings::min_block_vol_long);

    py::class_<RasSolver>(m, "RasSolver")
        .def(py::init<>())
        .def("init",        &RasSolver::init)
        .def("sa_run",      &RasSolver::sa_run)
        .def("get_score",   &RasSolver::get_score)
        .def("get_routes",  &RasSolver::get_routes)
        .def("set_routes",  &RasSolver::set_routes)
        .def("delta_score", &RasSolver::delta_score);

    py::class_<RasSolver::SAResult>(m, "SAResult")
        .def_readonly("best_routes",      &RasSolver::SAResult::best_routes)
        .def_readonly("best_score",       &RasSolver::SAResult::best_score)
        .def_readonly("score_history",    &RasSolver::SAResult::score_history)
        .def_readonly("selected_demands", &RasSolver::SAResult::selected_demands)
        .def_readonly("selected_routes",  &RasSolver::SAResult::selected_routes)
        .def_readonly("rewards",          &RasSolver::SAResult::rewards);
}
