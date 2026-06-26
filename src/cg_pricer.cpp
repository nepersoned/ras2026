/*
 * CG Pricing Problem — C++ parallel Dijkstra for Column Generation
 *
 * For each demand i, find the route with minimum (cost - dual[i]).
 * If min reduced cost < 0, we have a new column to add to the master LP.
 *
 * Build (Colab):
 *   g++ -O3 -fopenmp -shared -fPIC $(python3 -m pybind11 --includes) \
 *       src/cg_pricer.cpp -o src/cg_pricer$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

struct PricingResult {
    int    demand_idx;
    int    route_idx;       // index into candidate list
    double reduced_cost;
};

/*
 * price_all: for each demand, find the candidate route with lowest reduced cost.
 *
 * cand_costs[i]  = list of (route_idx, cost) pairs for demand i
 * duals[i]       = dual variable for demand i (from LP)
 *
 * Returns list of (demand_idx, route_idx, reduced_cost) for all demands
 * where reduced_cost < -tol.
 */
std::vector<PricingResult> price_all(
    const std::vector<std::vector<std::pair<int,double>>>& cand_costs,
    const std::vector<double>& duals,
    double tol = 1e-6
) {
    int n = (int)cand_costs.size();
    std::vector<PricingResult> results;
    results.reserve(n / 4);

    // Thread-local result buffers to avoid lock contention
    int nthreads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif
    std::vector<std::vector<PricingResult>> local(nthreads);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int i = 0; i < n; i++) {
        const auto& cands = cand_costs[i];
        if (cands.empty()) continue;
        double dual_i = (i < (int)duals.size()) ? duals[i] : 0.0;

        double best_rc = std::numeric_limits<double>::infinity();
        int    best_ri = -1;
        for (const auto& [ri, cost] : cands) {
            double rc = cost - dual_i;
            if (rc < best_rc) {
                best_rc = rc;
                best_ri = ri;
            }
        }
        if (best_rc < -tol && best_ri >= 0) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            local[tid].push_back({i, best_ri, best_rc});
        }
    }

    for (auto& v : local)
        for (auto& r : v)
            results.push_back(r);

    // Sort by most negative reduced cost (best columns first)
    std::sort(results.begin(), results.end(),
              [](const PricingResult& a, const PricingResult& b){
                  return a.reduced_cost < b.reduced_cost;
              });
    return results;
}

PYBIND11_MODULE(cg_pricer, m) {
    m.doc() = "CG Pricing — parallel reduced-cost column finder";

    py::class_<PricingResult>(m, "PricingResult")
        .def_readonly("demand_idx",    &PricingResult::demand_idx)
        .def_readonly("route_idx",     &PricingResult::route_idx)
        .def_readonly("reduced_cost",  &PricingResult::reduced_cost);

    m.def("price_all", &price_all,
          py::arg("cand_costs"),
          py::arg("duals"),
          py::arg("tol") = 1e-6,
          "Find negative-reduced-cost columns for all demands in parallel.");
}
