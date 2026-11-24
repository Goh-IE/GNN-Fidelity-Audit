#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct VectorHash {
    std::size_t operator()(const std::vector<int>& values) const noexcept {
        std::size_t seed = values.size();
        for (const int value : values) {
            seed ^= static_cast<std::size_t>(value) + 0x9e3779b97f4a7c15ULL
                    + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

class ConcreteEnumerator {
public:
    ConcreteEnumerator(
        std::string mode,
        std::vector<std::pair<std::int64_t, std::int64_t>> edges,
        const std::int64_t root,
        std::vector<int> valid_sizes,
        const std::uint64_t max_candidates)
        : mode_(std::move(mode)),
          edges_(std::move(edges)),
          root_(root),
          valid_sizes_(std::move(valid_sizes)),
          max_candidates_(max_candidates),
          selected_edge_(edges_.size(), false) {
        std::sort(valid_sizes_.begin(), valid_sizes_.end());
        valid_sizes_.erase(std::unique(valid_sizes_.begin(), valid_sizes_.end()), valid_sizes_.end());
        max_size_ = valid_sizes_.empty() ? 0 : valid_sizes_.back();
        for (int index = 0; index < static_cast<int>(edges_.size()); ++index) {
            const auto [a, b] = edges_[index];
            incident_[a].push_back(index);
            incident_[b].push_back(index);
        }
        for (auto& [node, indices] : incident_) {
            std::sort(indices.begin(), indices.end());
        }
    }

    int run() {
        if (mode_ == "all") {
            run_all();
        } else if (mode_ == "rooted") {
            run_rooted();
        } else if (mode_ == "connected") {
            run_connected();
        } else {
            std::cerr << "unsupported mode: " << mode_ << '\n';
            return 2;
        }
        std::cout << (limit_reached_ ? "LIMIT " : "END ") << candidate_count_ << '\n';
        return 0;
    }

private:
    bool is_valid_size(const int size) const {
        return std::binary_search(valid_sizes_.begin(), valid_sizes_.end(), size);
    }

    bool emit(const std::vector<int>& indices) {
        ++candidate_count_;
        if (candidate_count_ > max_candidates_) {
            limit_reached_ = true;
            return false;
        }
        std::cout << "C " << indices.size();
        for (const int index : indices) {
            std::cout << ' ' << index;
        }
        std::cout << '\n';
        return true;
    }

    bool all_combinations(const int start, const int target_size) {
        if (static_cast<int>(selected_indices_.size()) == target_size) {
            return emit(selected_indices_);
        }
        const int needed = target_size - static_cast<int>(selected_indices_.size());
        const int last_start = static_cast<int>(edges_.size()) - needed;
        for (int index = start; index <= last_start; ++index) {
            selected_indices_.push_back(index);
            const bool keep_going = all_combinations(index + 1, target_size);
            selected_indices_.pop_back();
            if (!keep_going) {
                return false;
            }
        }
        return true;
    }

    void run_all() {
        for (const int size : valid_sizes_) {
            if (size <= 0 || size > static_cast<int>(edges_.size())) {
                continue;
            }
            if (!all_combinations(0, size)) {
                return;
            }
        }
    }

    void run_rooted() {
        const auto found = incident_.find(root_);
        if (found == incident_.end()) {
            return;
        }
        for (const int index : found->second) {
            select_edge(index);
            const bool keep_going = expand_connected(0);
            unselect_edge(index);
            if (!keep_going) {
                return;
            }
        }
    }

    void run_connected() {
        for (int index = 0; index < static_cast<int>(edges_.size()); ++index) {
            select_edge(index);
            const bool keep_going = expand_connected(index);
            unselect_edge(index);
            if (!keep_going) {
                return;
            }
        }
    }

    bool expand_connected(const int minimum_edge_index) {
        std::vector<int> key = selected_indices_;
        std::sort(key.begin(), key.end());
        if (!visited_.insert(key).second) {
            return true;
        }
        if (is_valid_size(static_cast<int>(key.size())) && !emit(key)) {
            return false;
        }
        if (static_cast<int>(key.size()) >= max_size_) {
            return true;
        }

        std::vector<int> frontier;
        for (const auto& [node, count] : selected_node_counts_) {
            if (count <= 0) {
                continue;
            }
            const auto found = incident_.find(node);
            if (found != incident_.end()) {
                frontier.insert(frontier.end(), found->second.begin(), found->second.end());
            }
        }
        std::sort(frontier.begin(), frontier.end());
        frontier.erase(std::unique(frontier.begin(), frontier.end()), frontier.end());
        for (const int index : frontier) {
            if (selected_edge_[index] || (mode_ == "connected" && index < minimum_edge_index)) {
                continue;
            }
            select_edge(index);
            const bool keep_going = expand_connected(minimum_edge_index);
            unselect_edge(index);
            if (!keep_going) {
                return false;
            }
        }
        return true;
    }

    void select_edge(const int index) {
        selected_edge_[index] = true;
        selected_indices_.push_back(index);
        const auto [a, b] = edges_[index];
        ++selected_node_counts_[a];
        ++selected_node_counts_[b];
    }

    void unselect_edge(const int index) {
        const auto [a, b] = edges_[index];
        if (--selected_node_counts_[a] == 0) {
            selected_node_counts_.erase(a);
        }
        if (--selected_node_counts_[b] == 0) {
            selected_node_counts_.erase(b);
        }
        selected_indices_.pop_back();
        selected_edge_[index] = false;
    }

    std::string mode_;
    std::vector<std::pair<std::int64_t, std::int64_t>> edges_;
    std::int64_t root_;
    std::vector<int> valid_sizes_;
    std::uint64_t max_candidates_;
    int max_size_ = 0;
    std::unordered_map<std::int64_t, std::vector<int>> incident_;
    std::unordered_map<std::int64_t, int> selected_node_counts_;
    std::vector<bool> selected_edge_;
    std::vector<int> selected_indices_;
    std::unordered_set<std::vector<int>, VectorHash> visited_;
    std::uint64_t candidate_count_ = 0;
    bool limit_reached_ = false;
};

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string protocol;
    std::string mode;
    std::size_t edge_count = 0;
    std::int64_t root = -1;
    std::size_t valid_size_count = 0;
    std::uint64_t max_candidates = 0;
    if (!(std::cin >> protocol >> mode >> edge_count >> root >> valid_size_count >> max_candidates)) {
        std::cerr << "invalid header\n";
        return 2;
    }
    if (protocol != "CONCRETE_EDGE_SUBSET_ENUM_V1") {
        std::cerr << "unsupported protocol\n";
        return 2;
    }

    std::vector<int> valid_sizes(valid_size_count);
    for (std::size_t index = 0; index < valid_size_count; ++index) {
        if (!(std::cin >> valid_sizes[index]) || valid_sizes[index] <= 0) {
            std::cerr << "invalid valid-size record\n";
            return 2;
        }
    }

    std::vector<std::pair<std::int64_t, std::int64_t>> edges;
    edges.reserve(edge_count);
    for (std::size_t index = 0; index < edge_count; ++index) {
        std::int64_t a = 0;
        std::int64_t b = 0;
        if (!(std::cin >> a >> b)) {
            std::cerr << "invalid edge record at index " << index << '\n';
            return 2;
        }
        edges.emplace_back(a, b);
    }

    ConcreteEnumerator enumerator(
        std::move(mode), std::move(edges), root, std::move(valid_sizes), max_candidates);
    return enumerator.run();
}
