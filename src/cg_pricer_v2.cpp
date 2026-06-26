/*
 * CG Pricer V2 — True Column Generation pricing with full hub search
 *
 * For each demand i, searches ALL yards as potential hubs (not limited to
 * precomputed top-50). Uses OpenMP for parallel demand processing.
 *
 * Build (Colab):
 *   g++ -O3 -fopenmp -shared -fPIC $(python3 -m pybind11 --includes) \
 *       src/cg_pricer_v2.cpp -o src/cg_pricer_v2$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

struct NewColumn {
    int    demand_idx;
    int    hub;        // -1 = direct, >=0 = via hub
    double cost;
    double reduced_cost;
};

/*
 * price_full_hubs:
 *
 * For each demand i, evaluate ALL yards as potential hubs and find the
 * route with minimum reduced cost = cost - dual[i].
 * Returns new columns where reduced_cost < -tol.
 *
 * od_dist: flat map "origin_dest" → distance (penalized)
 * yard_ids: all yard IDs to consider as hubs
 * yard_hc:  handling cost per yard
 * demands:  (idx, origin, dest, volume, direct_only)
 * duals:    LP dual values per demand
 * tc, M:    cost coefficients
 */
std::vector<NewColumn> price_full_hubs(
    const std::unordered_map<std::string, double>& od_dist,
    const std::vector<int>&    yard_ids,
    const std::unordered_map<int, double>& yard_hc,
    const std::vector<std::tuple<int,int,int,int,bool>>& demands,
    const std::vector<double>& duals,
    double tc,
    double M_penalty,
    double tol = 1e-4
) {
    int n = (int)demands.size();
    int n_yards = (int)yard_ids.size();

    // Thread-local results
    int nthreads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    { #pragma omp single nthreads = omp_get_num_threads(); }
#endif
    std::vector<std::vector<NewColumn>> local(nthreads);

    auto dist_key = [](int a, int b) -> std::string {
        return std::to_string(a) + "_" + std::to_string(b);
    };

    auto get_dist = [&](int o, int d) -> double {
        auto it = od_dist.find(dist_key(o, d));
        return (it != od_dist.end()) ? it->second : std::numeric_limits<double>::infinity();
    };

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int i = 0; i < n; i++) {
        auto [idx, origin, dest, volume, direct_only] = demands[i];
        double dual_i = (i < (int)duals.size()) ? duals[i] : 0.0;

        double best_rc   = -tol;
        int    best_hub  = -2;  // -2 = none found yet
        double best_cost = 0.0;

        // Direct route
        double d_direct = get_dist(origin, dest);
        if (!std::isinf(d_direct)) {
            double cost_direct = tc * volume * d_direct;
            double rc = cost_direct - dual_i;
            if (rc < best_rc) {
                best_rc   = rc;
                best_hub  = -1;
                best_cost = cost_direct;
            }
        }

        // Via each hub (skip if direct_only)
        if (!direct_only) {
            for (int yi = 0; yi < n_yards; yi++) {
                int hub = yard_ids[yi];
                if (hub == origin || hub == dest) continue;

                double d1 = get_dist(origin, hub);
                double d2 = get_dist(hub, dest);
                if (std::isinf(d1) || std::isinf(d2)) continue;

                double hc = 0.0;
                auto hc_it = yard_hc.find(hub);
                if (hc_it != yard_hc.end()) hc = hc_it->second;

                double cost_hub = tc * volume * (d1 + d2) + hc * volume;
                double rc = cost_hub - dual_i;
                if (rc < best_rc) {
                    best_rc   = rc;
                    best_hub  = hub;
                    best_cost = cost_hub;
                }
            }
        }

        if (best_hub >= -1) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            local[tid].push_back({idx, best_hub, best_cost, best_rc});
        }
    }

    std::vector<NewColumn> result;
    for (auto& v : local)
        for (auto& c : v)
            result.push_back(c);

    std::sort(result.begin(), result.end(),
              [](const NewColumn& a, const NewColumn& b){
                  return a.reduced_cost < b.reduced_cost;
              });
    return result;
}

PYBIND11_MODULE(cg_pricer_v2, m) {
    m.doc() = "CG Pricer V2 — full hub search, OpenMP parallel";

    py::class_<NewColumn>(m, "NewColumn")
        .def_readonly("demand_idx",   &NewColumn::demand_idx)
        .def_readonly("hub",          &NewColumn::hub)
        .def_readonly("cost",         &NewColumn::cost)
        .def_readonly("reduced_cost", &NewColumn::reduced_cost);

    m.def("price_full_hubs", &price_full_hubs,
          py::arg("od_dist"),
          py::arg("yard_ids"),
          py::arg("yard_hc"),
          py::arg("demands"),
          py::arg("duals"),
          py::arg("tc"),
          py::arg("M_penalty"),
          py::arg("tol") = 1e-4,
          "True CG pricing: find negative-reduced-cost routes over ALL hubs.");
}
