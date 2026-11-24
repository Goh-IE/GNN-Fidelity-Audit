#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

class RootedComponentEnumerator {
public:
    RootedComponentEnumerator(
        std::vector<std::pair<std::int64_t, std::int64_t>> edges,
        const std::int64_t root,
        const int max_size,
        const std::uint64_t max_components)
        : edges_(std::move(edges)),
          root_(root),
          max_size_(max_size),
          max_components_(max_components),
          selected_edge_(edges_.size(), false) {
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
        if (!emit_component({})) {
            write_terminal();
            return 0;
        }

        const auto root_incident = incident_.find(root_);
        if (root_incident != incident_.end()) {
            for (const int edge_index : root_incident->second) {
                select_edge(edge_index);
                const bool keep_going = expand();
                unselect_edge(edge_index);
                if (!keep_going) {
                    break;
                }
            }
        }
        write_terminal();
        return 0;
    }

private:
    bool emit_component(const std::vector<int>& key) {
        ++component_count_;
        if (component_count_ > max_components_) {
            limit_reached_ = true;
            return false;
        }
        std::cout << "C " << key.size();
        for (const int edge_index : key) {
            std::cout << ' ' << edge_index;
        }
        std::cout << '\n';
        return true;
    }

    bool expand() {
        std::vector<int> key = selected_indices_;
        std::sort(key.begin(), key.end());
        if (!visited_.insert(key).second) {
            return true;
        }
        if (!emit_component(key)) {
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

        for (const int edge_index : frontier) {
            if (selected_edge_[edge_index]) {
                continue;
            }
            select_edge(edge_index);
            const bool keep_going = expand();
            unselect_edge(edge_index);
            if (!keep_going) {
                return false;
            }
        }
        return true;
    }

    void select_edge(const int edge_index) {
        selected_edge_[edge_index] = true;
        selected_indices_.push_back(edge_index);
        const auto [a, b] = edges_[edge_index];
        ++selected_node_counts_[a];
        ++selected_node_counts_[b];
    }

    void unselect_edge(const int edge_index) {
        const auto [a, b] = edges_[edge_index];
        if (--selected_node_counts_[a] == 0) {
            selected_node_counts_.erase(a);
        }
        if (--selected_node_counts_[b] == 0) {
            selected_node_counts_.erase(b);
        }
        selected_indices_.pop_back();
        selected_edge_[edge_index] = false;
    }

    void write_terminal() const {
        std::cout << (limit_reached_ ? "LIMIT " : "END ") << component_count_ << '\n';
    }

    std::vector<std::pair<std::int64_t, std::int64_t>> edges_;
    std::int64_t root_;
    int max_size_;
    std::uint64_t max_components_;
    std::unordered_map<std::int64_t, std::vector<int>> incident_;
    std::unordered_map<std::int64_t, int> selected_node_counts_;
    std::vector<bool> selected_edge_;
    std::vector<int> selected_indices_;
    std::unordered_set<std::vector<int>, VectorHash> visited_;
    std::uint64_t component_count_ = 0;
    bool limit_reached_ = false;
};

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string magic;
    std::size_t edge_count = 0;
    std::int64_t root = 0;
    int max_size = 0;
    std::uint64_t max_components = 0;
    if (!(std::cin >> magic >> edge_count >> root >> max_size >> max_components)) {
        std::cerr << "invalid header\n";
        return 2;
    }
    if (magic != "ROOTED_COMPONENT_ENUM_V1" || max_size < 0) {
        std::cerr << "unsupported protocol or invalid max_size\n";
        return 2;
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

    RootedComponentEnumerator enumerator(
        std::move(edges), root, max_size, max_components);
    return enumerator.run();
}
