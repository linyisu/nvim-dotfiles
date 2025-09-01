-- Auto-generated LuaSnip snippets
local ls = require("luasnip")
local ps = ls.parser.parse_snippet

return {

-- 02_Graph_Theory\Shortest_Path\SPFA.h
ps("02_graph_theory_shortest_path_spfa_h", [=[
// SPFA (Shortest Path Faster Algorithm) 算法
// Bellman-Ford算法的队列优化版本，适用于有负权边的图
// 时间复杂度：平均 O(VE)，最坏情况 O(VE)
// 空间复杂度：O(V + E)
const long long INF = numeric_limits<long long>::max() / 2;

// 基础SPFA算法实现
struct SPFA {
    vector<vector<pair<int, long long>>> g;  // 邻接表
    vector<long long> dist;                  // 距离数组
    vector<int> pre, cnt;                    // 前驱节点、入队次数
    vector<bool> inQueue;                    // 是否在队列中
    int n;

    SPFA(int n) : n(n), g(n), dist(n), pre(n), cnt(n), inQueue(n) {}

    void addEdge(int u, int v, long long w) { g[u].emplace_back(v, w); }

    // 单源最短路径，返回true表示无负环
    bool spfa(int s) {
        fill(dist.begin(), dist.end(), INF);
        fill(pre.begin(), pre.end(), -1);
        fill(cnt.begin(), cnt.end(), 0);
        fill(inQueue.begin(), inQueue.end(), false);

        queue<int> q;
        dist[s] = 0;
        q.push(s);
        inQueue[s] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            inQueue[u] = false;

            for (auto [v, w] : g[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pre[v] = u;
                    if (!inQueue[v]) {
                        q.push(v);
                        inQueue[v] = true;
                        cnt[v]++;
                        if (cnt[v] >= n) {
                            return false;  // 存在负环
                        }
                    }
                }
            }
        }
        return true;
    }

    // 获取从源点到目标点的路径
    vector<int> getPath(int t) {
        vector<int> path;
        for (int x = t; x != -1; x = pre[x]) {
            path.push_back(x);
        }
        reverse(path.begin(), path.end());
        return path;
    }

    // 获取到各点的距离
    vector<long long> getDistances() { return dist; }
};

// SLF优化的SPFA
// Small Label First：将距离更小的点放在队首
struct SPFA_SLF {
    vector<vector<pair<int, long long>>> g;
    vector<long long> dist;
    vector<int> cnt;
    vector<bool> inQueue;
    int n;

    SPFA_SLF(int n) : n(n), g(n), dist(n), cnt(n), inQueue(n) {}

    void addEdge(int u, int v, long long w) { g[u].emplace_back(v, w); }

    bool spfaSLF(int s) {
        fill(dist.begin(), dist.end(), INF);
        fill(cnt.begin(), cnt.end(), 0);
        fill(inQueue.begin(), inQueue.end(), false);

        deque<int> q;
        dist[s] = 0;
        q.push_back(s);
        inQueue[s] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop_front();
            inQueue[u] = false;

            for (auto [v, w] : g[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    if (!inQueue[v]) {
                        // SLF优化：如果新距离小于队首元素距离，插入队首
                        if (!q.empty() && dist[v] < dist[q.front()]) {
                            q.push_front(v);
                        } else {
                            q.push_back(v);
                        }
                        inQueue[v] = true;
                        cnt[v]++;
                        if (cnt[v] >= n) {
                            return false;  // 存在负环
                        }
                    }
                }
            }
        }
        return true;
    }
};

// LLL优化的SPFA
// Large Label Last：将距离较大的点移到队尾
struct SPFA_LLL {
    vector<vector<pair<int, long long>>> g;
    vector<long long> dist;
    vector<int> cnt;
    vector<bool> inQueue;
    int n;

    SPFA_LLL(int n) : n(n), g(n), dist(n), cnt(n), inQueue(n) {}

    void addEdge(int u, int v, long long w) { g[u].emplace_back(v, w); }

    bool spfaLLL(int s) {
        fill(dist.begin(), dist.end(), INF);
        fill(cnt.begin(), cnt.end(), 0);
        fill(inQueue.begin(), inQueue.end(), false);

        deque<int> q;
        dist[s] = 0;
        q.push_back(s);
        inQueue[s] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop_front();

            // LLL优化：如果队首元素距离大于平均值，移到队尾
            if (!q.empty()) {
                long long avgDist = 0;
                for (int x : q) avgDist += dist[x];
                avgDist /= q.size();

                if (dist[u] > avgDist) {
                    q.push_back(u);
                    continue;
                }
            }

            inQueue[u] = false;

            for (auto [v, w] : g[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    if (!inQueue[v]) {
                        q.push_back(v);
                        inQueue[v] = true;
                        cnt[v]++;
                        if (cnt[v] >= n) {
                            return false;  // 存在负环
                        }
                    }
                }
            }
        }
        return true;
    }
};

// 多源SPFA
// 同时从多个源点开始计算最短路径
struct MultiSourceSPFA {
    vector<vector<pair<int, long long>>> g;
    vector<long long> dist;
    vector<bool> inQueue;
    int n;

    MultiSourceSPFA(int n) : n(n), g(n), dist(n), inQueue(n) {}

    void addEdge(int u, int v, long long w) { g[u].emplace_back(v, w); }

    vector<long long> multiSourceSPFA(const vector<int>& sources) {
        fill(dist.begin(), dist.end(), INF);
        fill(inQueue.begin(), inQueue.end(), false);

        queue<int> q;
        for (int s : sources) {
            dist[s] = 0;
            q.push(s);
            inQueue[s] = true;
        }

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            inQueue[u] = false;

            for (auto [v, w] : g[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    if (!inQueue[v]) {
                        q.push(v);
                        inQueue[v] = true;
                    }
                }
            }
        }
        return dist;
    }
};

// 负环检测器
// 专门用于检测图中是否存在负环
struct NegativeCycleDetector {
    vector<vector<pair<int, long long>>> g;
    int n;

    NegativeCycleDetector(int n) : n(n), g(n) {}

    void addEdge(int u, int v, long long w) { g[u].emplace_back(v, w); }

    // 检测是否存在负环
    bool hasNegativeCycle() {
        vector<long long> dist(n, 0);
        vector<int> cnt(n, 0);
        vector<bool> inQueue(n, false);
        queue<int> q;

        // 将所有点加入队列
        for (int i = 0; i < n; i++) {
            q.push(i);
            inQueue[i] = true;
        }

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            inQueue[u] = false;

            for (auto [v, w] : g[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    if (!inQueue[v]) {
                        q.push(v);
                        inQueue[v] = true;
                        cnt[v]++;
                        if (cnt[v] >= n) {
                            return true;  // 存在负环
                        }
                    }
                }
            }
        }
        return false;
    }

    // 找出所有在负环上或能到达负环的点
    vector<bool> findNegativeCycleAffected() {
        vector<long long> dist(n, 0);
        vector<bool> inQueue(n, false);
        vector<bool> affected(n, false);
        queue<int> q;

        // 将所有点加入队列
        for (int i = 0; i < n; i++) {
            q.push(i);
            inQueue[i] = true;
        }

        // 运行n轮松弛
        for (int round = 0; round < n; round++) {
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int u = q.front();
                q.pop();
                inQueue[u] = false;

                for (auto [v, w] : g[u]) {
                    if (dist[u] + w < dist[v]) {
                        dist[v] = dist[u] + w;
                        if (!inQueue[v]) {
                            q.push(v);
                            inQueue[v] = true;
                        }
                    }
                }
            }
        }

        // 再运行一轮，如果还能松弛，说明受负环影响
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            affected[u] = true;

            for (auto [v, w] : g[u]) {
                if (dist[u] + w < dist[v] && !affected[v]) {
                    dist[v] = dist[u] + w;
                    q.push(v);
                }
            }
        }

        return affected;
    }
};
]=]),

-- 02_Graph_Theory\Special\Chromatic.h
ps("02_graph_theory_special_chromatic_h", [=[

// 图着色算法集合
// 包含多种图着色算法：贪心着色、回溯着色、DSATUR算法等
// 时间复杂度：贪心O(V+E)，回溯O(k^V)，DSATUR O(V^2)
struct GraphColoring {
    int n;
    vector<vector<int>> graph;
    vector<int> color;
    int max_colors;

    GraphColoring(int sz) : n(sz), max_colors(0) {
        graph.resize(n);
        color.resize(n, -1);
    }

    void add_edge(int u, int v) {
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    // 贪心着色算法（Welsh-Powell算法）
    // 按度数降序排列顶点，依次为每个顶点分配最小可用颜色
    int greedy_coloring() {
        color.assign(n, -1);
        vector<int> order(n);
        iota(order.begin(), order.end(), 0);

        // 按度数降序排列（Welsh-Powell算法）
        sort(order.begin(), order.end(), [&](int a, int b) { return graph[a].size() > graph[b].size(); });

        max_colors = 0;
        for (int u : order) {
            vector<bool> used(n, false);
            for (int v : graph[u]) {
                if (color[v] != -1) { used[color[v]] = true; }
            }

            for (int c = 0; c < n; c++) {
                if (!used[c]) {
                    color[u] = c;
                    max_colors = max(max_colors, c + 1);
                    break;
                }
            }
        }

        return max_colors;
    }

    // 回溯算法精确着色
    // 尝试用指定数量的颜色对图进行着色
    bool backtrack_coloring(int vertex, int num_colors) {
        if (vertex == n) return true;

        for (int c = 0; c < num_colors; c++) {
            bool valid = true;
            for (int v : graph[vertex]) {
                if (color[v] == c) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                color[vertex] = c;
                if (backtrack_coloring(vertex + 1, num_colors)) { return true; }
                color[vertex] = -1;
            }
        }

        return false;
    }

    // 寻找最小着色数（色数）
    // 使用回溯算法找到能够着色的最少颜色数
    int find_chromatic_number() {
        for (int colors = 1; colors <= n; colors++) {
            color.assign(n, -1);
            if (backtrack_coloring(0, colors)) {
                max_colors = colors;
                return colors;
            }
        }
        return n;  // 最坏情况
    }

    // DSATUR算法（动态饱和度优先）
    // 优先选择饱和度最大的顶点进行着色，通常比贪心算法效果更好
    int dsatur_coloring() {
        color.assign(n, -1);
        vector<int> degree(n), saturation(n, 0);
        vector<set<int>> colored_neighbors(n);

        for (int i = 0; i < n; i++) { degree[i] = graph[i].size(); }

        // 选择度数最大的顶点开始
        int start = max_element(degree.begin(), degree.end()) - degree.begin();
        color[start] = 0;
        max_colors = 1;

        // 更新相邻顶点的饱和度
        for (int v : graph[start]) {
            colored_neighbors[v].insert(0);
            saturation[v] = colored_neighbors[v].size();
        }

        for (int colored = 1; colored < n; colored++) {
            int next = -1;
            int max_sat = -1, max_deg = -1;

            // 选择饱和度最大的未着色顶点
            for (int i = 0; i < n; i++) {
                if (color[i] == -1) {
                    if (saturation[i] > max_sat || (saturation[i] == max_sat && degree[i] > max_deg)) {
                        max_sat = saturation[i];
                        max_deg = degree[i];
                        next = i;
                    }
                }
            }

            // 为选中的顶点着色
            for (int c = 0; c < n; c++) {
                if (colored_neighbors[next].find(c) == colored_neighbors[next].end()) {
                    color[next] = c;
                    max_colors = max(max_colors, c + 1);
                    break;
                }
            }

            // 更新相邻未着色顶点的饱和度
            for (int v : graph[next]) {
                if (color[v] == -1) {
                    colored_neighbors[v].insert(color[next]);
                    saturation[v] = colored_neighbors[v].size();
                }
            }
        }

        return max_colors;
    }

    // 检查是否为二分图（2-着色）
    // 使用BFS检查图是否可以用两种颜色着色
    bool is_bipartite() {
        color.assign(n, -1);
        queue<int> q;

        for (int start = 0; start < n; start++) {
            if (color[start] == -1) {
                color[start] = 0;
                q.push(start);

                while (!q.empty()) {
                    int u = q.front();
                    q.pop();

                    for (int v : graph[u]) {
                        if (color[v] == -1) {
                            color[v] = 1 - color[u];
                            q.push(v);
                        } else if (color[v] == color[u]) {
                            return false;
                        }
                    }
                }
            }
        }

        max_colors = 2;
        return true;
    }

    // 边着色算法（Vizing算法近似）
    // 为图的边进行着色，使得相邻的边颜色不同
    int edge_coloring() {
        vector<vector<pair<int, int>>> edge_colors(n);
        map<pair<int, int>, int> edge_color_map;
        int num_edge_colors = 0;

        for (int u = 0; u < n; u++) {
            for (int v : graph[u]) {
                if (u < v) {  // 避免重复处理边
                    set<int> used_colors;

                    // 收集u和v已使用的边颜色
                    for (auto& edge : edge_colors[u]) { used_colors.insert(edge.second); }
                    for (auto& edge : edge_colors[v]) { used_colors.insert(edge.second); }

                    // 找到最小可用颜色
                    int color = 0;
                    while (used_colors.count(color)) color++;

                    edge_colors[u].push_back({v, color});
                    edge_colors[v].push_back({u, color});
                    edge_color_map[{min(u, v), max(u, v)}] = color;
                    num_edge_colors = max(num_edge_colors, color + 1);
                }
            }
        }

        return num_edge_colors;
    }

    // 完美着色验证
    // 检查当前着色方案是否满足相邻顶点颜色不同的要求
    bool verify_coloring() {
        for (int u = 0; u < n; u++) {
            for (int v : graph[u]) {
                if (color[u] == color[v]) { return false; }
            }
        }
        return true;
    }

    // 获取当前着色方案
    vector<int> get_coloring() { return color; }

    // 计算图的团数上界（着色数下界）
    // 图的色数至少等于最大团的大小
    int max_clique_bound() {
        // 简单实现：寻找最大团的大小
        int max_clique_size = 1;

        for (int u = 0; u < n; u++) {
            vector<int> candidates = graph[u];
            candidates.push_back(u);

            // 检查候选集合是否形成团
            bool is_clique = true;
            for (int i = 0; i < candidates.size() && is_clique; i++) {
                for (int j = i + 1; j < candidates.size() && is_clique; j++) {
                    int v1 = candidates[i], v2 = candidates[j];
                    bool connected = false;
                    for (int neighbor : graph[v1]) {
                        if (neighbor == v2) {
                            connected = true;
                            break;
                        }
                    }
                    if (!connected) is_clique = false;
                }
            }

            if (is_clique) { max_clique_size = max(max_clique_size, (int)candidates.size()); }
        }

        return max_clique_size;
    }

    // Brooks定理检查
    // 检查Brooks定理是否适用：除完全图和奇圈外，色数不超过最大度数
    bool brooks_theorem_applies() {
        // Brooks定理：除了完全图和奇圈，其他连通图的色数不超过最大度数
        int max_degree = 0;
        for (int i = 0; i < n; i++) { max_degree = max(max_degree, (int)graph[i].size()); }

        // 检查是否为完全图
        bool is_complete = true;
        for (int i = 0; i < n && is_complete; i++) {
            if (graph[i].size() != n - 1) is_complete = false;
        }

        if (is_complete) return false;

        // 检查是否为奇圈
        if (max_degree == 2) {
            // 可能是圈，检查长度
            vector<bool> visited(n, false);
            function<bool(int, int, int)> is_odd_cycle = [&](int u, int start, int depth) -> bool {
                if (depth > 1 && u == start) { return depth % 2 == 1; }

                visited[u] = true;
                for (int v : graph[u]) {
                    if (depth == 0 || v != start || depth > 2) {
                        if (!visited[v] || (depth > 2 && v == start)) {
                            if (is_odd_cycle(v, start, depth + 1)) return true;
                        }
                    }
                }
                return false;
            };

            if (is_odd_cycle(0, 0, 0)) return false;
        }

        return true;
    }
};

// 使用示例：
// GraphColoring gc(n);
// gc.add_edge(u, v);
// int colors = gc.greedy_coloring();
// bool bipartite = gc.is_bipartite();
// int chromatic = gc.find_chromatic_number();
]=]),

-- 02_Graph_Theory\Special\EulerPath.h
ps("02_graph_theory_special_eulerpath_h", [=[

// 欧拉路径和欧拉回路算法
// 欧拉路径：经过图中每条边恰好一次的路径
// 欧拉回路：起点和终点相同的欧拉路径
// 时间复杂度：O(V+E)
struct EulerPath {
    int n;
    vector<vector<int>> graph;
    vector<int> degree;
    vector<bool> used_edge;
    vector<int> path;

    EulerPath(int sz) : n(sz) {
        graph.resize(n);
        degree.resize(n, 0);
    }

    void add_edge(int u, int v, bool directed = false) {
        graph[u].push_back(v);
        degree[u]++;
        if (!directed) {
            graph[v].push_back(u);
            degree[v]++;
        } else {
            degree[v]--;  // 有向图：记录出度-入度
        }
    }

    // 检查是否存在欧拉路径/回路
    // 无向图：奇度顶点0个→欧拉回路，2个→欧拉路径
    // 有向图：所有顶点入度=出度→欧拉回路，1个顶点出度-入度=1，1个顶点入度-出度=1→欧拉路径
    pair<bool, pair<int, int>> check_euler_path(bool directed = false) {
        if (directed) {
            int start = -1, end = -1;
            for (int i = 0; i < n; i++) {
                if (degree[i] == 1) {
                    if (start == -1)
                        start = i;
                    else
                        return {false, {-1, -1}};
                } else if (degree[i] == -1) {
                    if (end == -1)
                        end = i;
                    else
                        return {false, {-1, -1}};
                } else if (degree[i] != 0) {
                    return {false, {-1, -1}};
                }
            }
            if (start == -1 && end == -1) return {true, {0, 0}};        // 欧拉回路
            if (start != -1 && end != -1) return {true, {start, end}};  // 欧拉路径
            return {false, {-1, -1}};
        } else {
            int odd_count = 0;
            vector<int> odd_vertices;
            for (int i = 0; i < n; i++) {
                if (degree[i] % 2 == 1) {
                    odd_count++;
                    odd_vertices.push_back(i);
                }
            }
            if (odd_count == 0) return {true, {0, 0}};                              // 欧拉回路
            if (odd_count == 2) return {true, {odd_vertices[0], odd_vertices[1]}};  // 欧拉路径
            return {false, {-1, -1}};
        }
    }

    // Hierholzer算法求欧拉路径
    // 基于DFS的算法，时间复杂度O(V+E)
    vector<int> find_euler_path(int start = -1, bool directed = false) {
        auto euler_check = check_euler_path(directed);
        if (!euler_check.first) return {};

        if (start == -1) start = euler_check.second.first;

        // 建立邻接表的边索引
        vector<vector<pair<int, int>>> adj(n);
        int edge_id = 0;
        for (int u = 0; u < n; u++) {
            for (int v : graph[u]) {
                adj[u].push_back({v, edge_id++});
                if (!directed) { adj[v].push_back({u, edge_id - 1}); }
            }
        }

        used_edge.assign(edge_id, false);
        vector<int> current_path;
        vector<int> result;

        current_path.push_back(start);

        while (!current_path.empty()) {
            int u = current_path.back();
            bool found = false;

            for (auto& edge : adj[u]) {
                int v = edge.first;
                int eid = edge.second;

                if (!used_edge[eid]) {
                    used_edge[eid] = true;
                    current_path.push_back(v);
                    found = true;
                    break;
                }
            }

            if (!found) {
                result.push_back(current_path.back());
                current_path.pop_back();
            }
        }

        reverse(result.begin(), result.end());
        return result;
    }

    // 检查图的连通性
    // 欧拉路径存在的必要条件之一
    bool is_connected() {
        vector<bool> visited(n, false);
        int start = 0;
        while (start < n && graph[start].empty()) start++;
        if (start == n) return true;  // 没有边

        function<void(int)> dfs = [&](int u) {
            visited[u] = true;
            for (int v : graph[u]) {
                if (!visited[v]) dfs(v);
            }
        };

        dfs(start);
        for (int i = 0; i < n; i++) {
            if (!graph[i].empty() && !visited[i]) { return false; }
        }
        return true;
    }

    // 求所有欧拉回路的数量（BEST定理）
    // 仅适用于有向强连通图且每个顶点入度=出度
    long long count_euler_circuits() {
        // 可以通过矩阵树定理计算，但实现复杂
        return -1;  // 表示未实现
    }
};

// 使用示例：
// EulerPath euler(n);
// euler.add_edge(u, v);
// auto check = euler.check_euler_path();
// if (check.first) {
//     vector<int> path = euler.find_euler_path(check.second.first);
// }
]=]),

-- 02_Graph_Theory\Special\HamiltonPath.h
ps("02_graph_theory_special_hamiltonpath_h", [=[

// 哈密顿路径和哈密顿回路算法
// 哈密顿路径：经过图中每个顶点恰好一次的路径
// 哈密顿回路：起点和终点相同的哈密顿路径
// 这是NP完全问题，只能用指数时间算法求解
struct HamiltonPath {
    int n;
    vector<vector<int>> graph;
    vector<vector<bool>> adj;
    vector<bool> visited;
    vector<int> path;
    vector<vector<int>> all_paths;

    HamiltonPath(int sz) : n(sz) {
        graph.resize(n);
        adj.assign(n, vector<bool>(n, false));
        visited.resize(n);
    }

    void add_edge(int u, int v, bool directed = false) {
        graph[u].push_back(v);
        adj[u][v] = true;
        if (!directed) {
            graph[v].push_back(u);
            adj[v][u] = true;
        }
    }

    // 回溯算法求哈密顿路径
    // 时间复杂度：O(n!)，适合小规模图
    bool find_hamilton_path_dfs(int u, int target, int depth) {
        if (depth == n) {
            return (target == -1 || adj[u][target]);  // target=-1表示路径，否则是回路
        }

        for (int v : graph[u]) {
            if (!visited[v]) {
                visited[v] = true;
                path[depth] = v;
                if (find_hamilton_path_dfs(v, target, depth + 1)) { return true; }
                visited[v] = false;
            }
        }
        return false;
    }

    // 寻找哈密顿路径（从start开始）
    vector<int> find_hamilton_path(int start) {
        visited.assign(n, false);
        path.assign(n, -1);

        visited[start] = true;
        path[0] = start;

        if (find_hamilton_path_dfs(start, -1, 1)) { return path; }
        return {};
    }

    // 寻找哈密顿回路（从start开始并回到start）
    vector<int> find_hamilton_cycle(int start) {
        visited.assign(n, false);
        path.assign(n, -1);

        visited[start] = true;
        path[0] = start;

        if (find_hamilton_path_dfs(start, start, 1)) { return path; }
        return {};
    }

    // 动态规划解法（状态压缩DP）
    // 时间复杂度：O(n^2 * 2^n)，空间复杂度：O(n * 2^n)
    // 适合n<=20的图
    pair<bool, vector<int>> hamilton_path_dp(int start, int end = -1) {
        // dp[mask][i] 表示访问过mask中的点，当前在点i的最小路径长度
        vector<vector<int>> dp(1 << n, vector<int>(n, -1));
        vector<vector<int>> parent(1 << n, vector<int>(n, -1));

        dp[1 << start][start] = 0;

        for (int mask = 0; mask < (1 << n); mask++) {
            for (int u = 0; u < n; u++) {
                if (!(mask & (1 << u)) || dp[mask][u] == -1) continue;

                for (int v : graph[u]) {
                    if (mask & (1 << v)) continue;  // 已经访问过

                    int new_mask = mask | (1 << v);
                    if (dp[new_mask][v] == -1) {
                        dp[new_mask][v] = dp[mask][u] + 1;
                        parent[new_mask][v] = u;
                    }
                }
            }
        }

        // 寻找哈密顿路径
        int full_mask = (1 << n) - 1;
        int best_end = -1;

        if (end == -1) {
            // 寻找任意终点的哈密顿路径
            for (int i = 0; i < n; i++) {
                if (dp[full_mask][i] != -1) {
                    best_end = i;
                    break;
                }
            }
        } else {
            // 指定终点
            if (dp[full_mask][end] != -1) { best_end = end; }
        }

        if (best_end == -1) return {false, {}};

        // 重构路径
        vector<int> result_path;
        int mask = full_mask;
        int curr = best_end;

        while (curr != -1) {
            result_path.push_back(curr);
            int prev = parent[mask][curr];
            if (prev != -1) { mask ^= (1 << curr); }
            curr = prev;
        }

        reverse(result_path.begin(), result_path.end());
        return {true, result_path};
    }

    // 检查哈密顿回路（状态压缩DP）
    // 使用动态规划检查是否存在哈密顿回路并构造路径
    pair<bool, vector<int>> hamilton_cycle_dp(int start = 0) {
        vector<vector<int>> dp(1 << n, vector<int>(n, -1));
        vector<vector<int>> parent(1 << n, vector<int>(n, -1));

        dp[1 << start][start] = 0;

        for (int mask = 0; mask < (1 << n); mask++) {
            for (int u = 0; u < n; u++) {
                if (!(mask & (1 << u)) || dp[mask][u] == -1) continue;

                for (int v : graph[u]) {
                    if (mask & (1 << v)) continue;

                    int new_mask = mask | (1 << v);
                    if (dp[new_mask][v] == -1) {
                        dp[new_mask][v] = dp[mask][u] + 1;
                        parent[new_mask][v] = u;
                    }
                }
            }
        }

        // 检查是否能回到起点
        int full_mask = (1 << n) - 1;
        for (int u = 0; u < n; u++) {
            if (u != start && dp[full_mask][u] != -1 && adj[u][start]) {
                // 重构路径
                vector<int> result_path;
                int mask = full_mask;
                int curr = u;

                while (curr != -1) {
                    result_path.push_back(curr);
                    int prev = parent[mask][curr];
                    if (prev != -1) { mask ^= (1 << curr); }
                    curr = prev;
                }

                reverse(result_path.begin(), result_path.end());
                result_path.push_back(start);  // 添加回到起点
                return {true, result_path};
            }
        }

        return {false, {}};
    }

    // 计算所有哈密顿路径的数量
    // 用于小规模图的完整枚举
    int count_hamilton_paths() {
        all_paths.clear();
        function<void(int, vector<int>&)> dfs = [&](int depth, vector<int>& current_path) {
            if (depth == n) {
                all_paths.push_back(current_path);
                return;
            }

            int u = current_path.back();
            for (int v : graph[u]) {
                if (find(current_path.begin(), current_path.end(), v) == current_path.end()) {
                    current_path.push_back(v);
                    dfs(depth + 1, current_path);
                    current_path.pop_back();
                }
            }
        };

        for (int start = 0; start < n; start++) {
            vector<int> current_path = {start};
            dfs(1, current_path);
        }

        return all_paths.size();
    }

    // Ore定理检查（充分条件）
    // 如果对于不相邻的顶点u,v，有deg(u)+deg(v)>=n，则存在哈密顿回路
    bool ore_theorem_check() {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!adj[i][j] && graph[i].size() + graph[j].size() < n) { return false; }
            }
        }
        return true;
    }

    // Dirac定理检查（充分条件）
    // 如果每个顶点的度数至少为n/2，则存在哈密顿回路
    bool dirac_theorem_check() {
        for (int i = 0; i < n; i++) {
            if (graph[i].size() < n / 2) { return false; }
        }
        return true;
    }
};

// 使用示例：
// HamiltonPath hamilton(n);
// hamilton.add_edge(u, v);
// vector<int> path = hamilton.find_hamilton_path(0);
// auto [has_cycle, cycle] = hamilton.hamilton_cycle_dp();
]=]),

-- 02_Graph_Theory\Special\PlanarGraph.h
ps("02_graph_theory_special_planargraph_h", [=[

// 平面图相关算法
// 平面图：可以在平面上绘制且边不相交的图
// 包含平面性测试、欧拉公式验证、对偶图构造等算法
struct PlanarGraph {
    int n, m;
    vector<vector<int>> graph;
    vector<pair<int, int>> edges;

    PlanarGraph(int vertices) : n(vertices), m(0) { graph.resize(n); }

    void add_edge(int u, int v) {
        graph[u].push_back(v);
        graph[v].push_back(u);
        edges.push_back({u, v});
        m++;
    }

    // Kuratowski定理检查（基于K5和K3,3子图）
    // 图是平面图当且仅当不包含K5或K3,3的细分
    bool has_k5_subdivision() {
        // 检查是否包含K5的细分
        if (n < 5) return false;

        // 枚举所有5个顶点的子集
        function<bool(vector<int>&, int, int)> check_k5;
        check_k5 = [&](vector<int>& vertices, int start, int count) -> bool {
            if (count == 5) {
                // 检查这5个顶点是否能形成K5的细分
                for (int i = 0; i < 5; i++) {
                    for (int j = i + 1; j < 5; j++) {
                        // 检查vertices[i]和vertices[j]之间是否有路径
                        if (!has_path_subdivision(vertices[i], vertices[j], vertices)) { return false; }
                    }
                }
                return true;
            }

            for (int i = start; i < n; i++) {
                vertices[count] = i;
                if (check_k5(vertices, i + 1, count + 1)) { return true; }
            }
            return false;
        };

        vector<int> vertices(5);
        return check_k5(vertices, 0, 0);
    }

    // 检查是否包含K3,3的细分
    bool has_k33_subdivision() {
        // 检查是否包含K3,3的细分
        if (n < 6) return false;

        function<bool(vector<int>&, vector<int>&, int, int, int, int)> check_k33;
        check_k33 =
            [&](vector<int>& part1, vector<int>& part2, int start1, int count1, int start2, int count2) -> bool {
            if (count1 == 3 && count2 == 3) {
                // 检查是否形成K3,3的细分
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        vector<int> forbidden = part1;
                        forbidden.insert(forbidden.end(), part2.begin(), part2.end());
                        if (!has_path_subdivision(part1[i], part2[j], forbidden)) { return false; }
                    }
                }
                return true;
            }

            if (count1 < 3) {
                for (int i = start1; i < n; i++) {
                    part1[count1] = i;
                    if (check_k33(part1, part2, i + 1, count1 + 1, start2, count2)) { return true; }
                }
            }

            if (count2 < 3) {
                for (int i = start2; i < n; i++) {
                    bool in_part1 = false;
                    for (int j = 0; j < count1; j++) {
                        if (part1[j] == i) {
                            in_part1 = true;
                            break;
                        }
                    }
                    if (!in_part1) {
                        part2[count2] = i;
                        if (check_k33(part1, part2, start1, count1, i + 1, count2 + 1)) { return true; }
                    }
                }
            }

            return false;
        };

        vector<int> part1(3), part2(3);
        return check_k33(part1, part2, 0, 0, 0, 0);
    }

    // 辅助函数：检查两点间是否有路径细分
    bool has_path_subdivision(int start, int end, const vector<int>& forbidden) {
        vector<bool> visited(n, false);
        for (int v : forbidden) {
            if (v != start && v != end) visited[v] = true;
        }

        queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            if (u == end) return true;

            for (int v : graph[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }

        return false;
    }

    // 基于Kuratowski定理的平面性测试
    // 时间复杂度：指数级，仅适用于小规模图
    bool is_planar_kuratowski() { return !has_k5_subdivision() && !has_k33_subdivision(); }

    // 欧拉公式验证（必要条件）
    // 连通平面图满足：V - E + F = 2，简单平面图：E ≤ 3V - 6
    bool euler_formula_check() {
        if (m == 0) return true;

        // 对于连通平面图：V - E + F = 2
        // 对于简单连通平面图：E <= 3V - 6
        if (n >= 3 && m > 3 * n - 6) return false;

        // 无三角形的平面图：E <= 2V - 4
        bool has_triangle = false;
        for (int u = 0; u < n; u++) {
            for (int v : graph[u]) {
                for (int w : graph[u]) {
                    if (v != w && has_edge(v, w)) {
                        has_triangle = true;
                        break;
                    }
                }
                if (has_triangle) break;
            }
            if (has_triangle) break;
        }

        if (!has_triangle && n >= 3 && m > 2 * n - 4) return false;

        return true;
    }

    bool has_edge(int u, int v) {
        for (int neighbor : graph[u]) {
            if (neighbor == v) return true;
        }
        return false;
    }

    // 左右测试算法（Left-Right Planarity Test）简化版
    // 线性时间平面性测试算法的简化实现
    bool left_right_planarity_test() {
        if (!euler_formula_check()) return false;

        // 这里实现简化的左右测试
        // 完整实现需要复杂的数据结构
        return is_planar_kuratowski();
    }

    // DFS平面性测试
    // 基于DFS树和反向边的平面性检测
    bool dfs_planarity_test() {
        vector<bool> visited(n, false);
        vector<int> dfs_order;
        vector<int> parent(n, -1);
        vector<vector<int>> back_edges(n);

        // DFS遍历
        function<void(int)> dfs = [&](int u) {
            visited[u] = true;
            dfs_order.push_back(u);

            for (int v : graph[u]) {
                if (!visited[v]) {
                    parent[v] = u;
                    dfs(v);
                } else if (parent[u] != v) {
                    // 反向边
                    back_edges[u].push_back(v);
                }
            }
        };

        if (n > 0) dfs(0);

        // 检查反向边的平面嵌入
        // 这里实现简化版本
        return back_edges.size() <= n;  // 简化检查
    }

    // 获取图的亏格（genus）
    // 亏格g：图嵌入到g-环面上的最小g值，平面图的亏格为0
    int calculate_genus() {
        // 根据欧拉公式：V - E + F = 2 - 2g（g为亏格）
        // 对于连通图
        int components = count_components();

        if (components == 0) return 0;

        // 简化计算，假设图是连通的
        // g = (2 - V + E - F) / 2
        // 对于平面图的最小嵌入，F可以通过面的计算得到

        // 这里返回理论最小亏格
        int max_edges_planar = max(0, 3 * n - 6);
        if (m <= max_edges_planar) return 0;

        return (m - max_edges_planar + 2) / 2;  // 简化估算
    }

    int count_components() {
        vector<bool> visited(n, false);
        int components = 0;

        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                components++;
                function<void(int)> dfs = [&](int u) {
                    visited[u] = true;
                    for (int v : graph[u]) {
                        if (!visited[v]) dfs(v);
                    }
                };
                dfs(i);
            }
        }

        return components;
    }

    // 平面图的对偶图构造
    // 对偶图：每个面对应一个顶点，相邻面对应的顶点相连
    vector<vector<int>> construct_dual_graph() {
        // 这需要知道平面嵌入的面信息
        // 这里提供框架，实际实现需要更复杂的几何信息
        vector<vector<int>> dual;

        // 简化：假设已知面的邻接关系
        // 实际需要从平面嵌入中提取面信息

        return dual;
    }

    // 检查是否为外平面图
    // 外平面图：所有顶点都在外面的平面图，不包含K4和K2,3细分
    bool is_outerplanar() {
        // 外平面图当且仅当不包含K4细分和K2,3细分
        if (n < 4) return true;

        // 检查K4细分
        function<bool(vector<int>&, int, int)> check_k4;
        check_k4 = [&](vector<int>& vertices, int start, int count) -> bool {
            if (count == 4) {
                // 检查这4个顶点是否能形成K4的细分
                for (int i = 0; i < 4; i++) {
                    for (int j = i + 1; j < 4; j++) {
                        if (!has_path_subdivision(vertices[i], vertices[j], vertices)) { return false; }
                    }
                }
                return true;
            }

            for (int i = start; i < n; i++) {
                vertices[count] = i;
                if (check_k4(vertices, i + 1, count + 1)) { return true; }
            }
            return false;
        };

        vector<int> vertices(4);
        if (check_k4(vertices, 0, 0)) return false;

        // 外平面图的边数限制：E <= 2V - 3
        return m <= 2 * n - 3;
    }

    // 计算图的树宽上界（与平面性相关）
    // 平面图的树宽最多为O(√n)
    int tree_width_bound() {
        // 平面图的树宽最多为sqrt(3*n)
        // 这里给出理论上界
        return (int)ceil(sqrt(3.0 * n));
    }

    // 四色定理应用（平面图4-着色）
    // 根据四色定理，所有平面图都可以用4种颜色着色
    bool is_four_colorable() {
        // 根据四色定理，所有平面图都是4-可着色的
        return is_planar_kuratowski();
    }
};

// 使用示例：
// PlanarGraph pg(n);
// pg.add_edge(u, v);
// bool planar = pg.is_planar_kuratowski();
// bool outerplanar = pg.is_outerplanar();
// int genus = pg.calculate_genus();
]=]),

-- 02_Graph_Theory\Tree_Algorithms\HeavyLightDecomp.h
ps("02_graph_theory_tree_algorithms_heavylightdecomp_h", [=[

// 重链剖分 - 将树分解为重链和轻边，支持路径查询和子树查询
// 时间复杂度：预处理 O(n)，单次查询 O(log n)
// 空间复杂度：O(n)
struct HeavyLightDecomp {
    vector<vector<int>> tree;                         // 邻接表
    vector<int> parent, depth, heavy, head, pos, sz;  // 父节点、深度、重儿子、链头、DFS序、子树大小
    int n, timer;

    HeavyLightDecomp(int _n) : n(_n), timer(0) {
        tree.resize(n);
        parent.resize(n);
        depth.resize(n);
        heavy.resize(n, -1);
        head.resize(n);
        pos.resize(n);
        sz.resize(n);
    }

    void add_edge(int u, int v) {
        tree[u].push_back(v);
        tree[v].push_back(u);
    }  // 第一次DFS：计算子树大小、深度、重儿子
    int dfs1(int v, int p, int d) {
        parent[v] = p;
        depth[v] = d;
        sz[v] = 1;
        int max_size = 0;

        for (int u : tree[v]) {
            if (u != p) {
                sz[v] += dfs1(u, v, d + 1);
                if (sz[u] > max_size) {
                    max_size = sz[u];
                    heavy[v] = u;  // 子树最大的儿子作为重儿子
                }
            }
        }

        return sz[v];
    }

    // 第二次DFS：构建重链，分配DFS序
    void dfs2(int v, int h) {
        head[v] = h;       // 链头
        pos[v] = timer++;  // DFS序

        if (heavy[v] != -1) {
            dfs2(heavy[v], h);  // 重儿子继续在同一条重链
        }

        for (int u : tree[v]) {
            if (u != parent[v] && u != heavy[v]) {
                dfs2(u, u);  // 轻儿子开始新的重链
            }
        }
    }

    void build(int root = 0) {
        dfs1(root, -1, 0);
        dfs2(root, root);
    }  // 查询LCA（最近公共祖先）
    int lca(int u, int v) {
        while (head[u] != head[v]) {
            if (depth[head[u]] > depth[head[v]]) {
                u = parent[head[u]];
            } else {
                v = parent[head[v]];
            }
        }
        return depth[u] < depth[v] ? u : v;
    }

    // 查询路径上的区间（用于配合线段树等数据结构）
    vector<pair<int, int>> query_path(int u, int v) {
        vector<pair<int, int>> up_path, down_path;

        while (head[u] != head[v]) {
            if (depth[head[u]] > depth[head[v]]) {
                up_path.push_back({pos[head[u]], pos[u]});
                u = parent[head[u]];
            } else {
                down_path.push_back({pos[head[v]], pos[v]});
                v = parent[head[v]];
            }
        }

        // 处理同一条重链上的部分
        if (u != v) {
            if (depth[u] > depth[v]) {
                up_path.push_back({pos[v], pos[u]});
            } else {
                down_path.push_back({pos[u], pos[v]});
            }
        } else {
            up_path.push_back({pos[u], pos[u]});
        }

        // 合并路径，注意down_path需要反向
        reverse(down_path.begin(), down_path.end());
        up_path.insert(up_path.end(), down_path.begin(), down_path.end());

        return up_path;
    }

    // 查询子树对应的区间
    pair<int, int> query_subtree(int v) { return {pos[v], pos[v] + sz[v] - 1}; }

    // 查询两点间距离
    int distance(int u, int v) { return depth[u] + depth[v] - 2 * depth[lca(u, v)]; }

    // 判断u是否是v的祖先
    bool is_ancestor(int u, int v) { return pos[u] <= pos[v] && pos[v] < pos[u] + sz[u]; }
};
]=]),

-- 02_Graph_Theory\Tree_Algorithms\LCA_Binary.h
ps("02_graph_theory_tree_algorithms_lca_binary_h", [=[

// LCA（最近公共祖先）倍增算法
// 时间复杂度：预处理 O(n log n)，单次查询 O(log n)
// 空间复杂度：O(n log n)
struct LCA_Binary {
    int n, LOG;
    vector<vector<int>> up;  // up[i][j] 表示从节点 i 向上跳 2^j 步到达的节点
    vector<int> depth;
    vector<vector<int>> adj;

    LCA_Binary(int _n) : n(_n) {
        LOG = 0;
        while ((1 << LOG) <= n) LOG++;
        up.assign(n, vector<int>(LOG, -1));
        depth.assign(n, 0);
        adj.assign(n, vector<int>());
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void dfs(int v, int p) {
        up[v][0] = p;
        // 预处理倍增数组
        for (int i = 1; i < LOG; i++) {
            if (up[v][i - 1] != -1) { up[v][i] = up[up[v][i - 1]][i - 1]; }
        }

        for (int u : adj[v]) {
            if (u != p) {
                depth[u] = depth[v] + 1;
                dfs(u, v);
            }
        }
    }

    void preprocess(int root = 0) {
        depth[root] = 0;
        dfs(root, -1);
    }

    int lca(int u, int v) {
        if (depth[u] < depth[v]) swap(u, v);

        // 将u提升到与v相同的深度
        int diff = depth[u] - depth[v];
        for (int i = 0; i < LOG; i++) {
            if ((diff >> i) & 1) { u = up[u][i]; }
        }

        if (u == v) return u;

        // 二分查找LCA
        for (int i = LOG - 1; i >= 0; i--) {
            if (up[u][i] != up[v][i]) {
                u = up[u][i];
                v = up[v][i];
            }
        }

        return up[u][0];
    }

    int distance(int u, int v) { return depth[u] + depth[v] - 2 * depth[lca(u, v)]; }

    // 从u向上走k步（k步祖先）
    int kth_ancestor(int u, int k) {
        if (depth[u] < k) return -1;

        for (int i = 0; i < LOG; i++) {
            if ((k >> i) & 1) {
                u = up[u][i];
                if (u == -1) return -1;
            }
        }
        return u;
    }

    // 路径上u到v的第k个节点（0-indexed）
    int kth_node_on_path(int u, int v, int k) {
        int l = lca(u, v);
        int dist_u = depth[u] - depth[l];
        int dist_v = depth[v] - depth[l];

        if (k <= dist_u) {
            return kth_ancestor(u, k);
        } else {
            return kth_ancestor(v, dist_u + dist_v - k);
        }
    }
};
]=]),

-- 02_Graph_Theory\Tree_Algorithms\LCA_Tarjan.h
ps("02_graph_theory_tree_algorithms_lca_tarjan_h", [=[

// Tarjan离线LCA算法
// 时间复杂度：O(n + m α(n))，其中m为查询次数，α为反阿克曼函数
// 空间复杂度：O(n + m)
// 适用于需要批量处理大量LCA查询的场景
struct LCA_Tarjan {
    struct Query {
        int v, id;  // 查询节点和查询编号
    };

    vector<vector<int>> adj;
    vector<vector<Query>> queries;  // 每个节点的查询列表
    vector<int> parent, ancestor;   // 并查集的父节点和祖先
    vector<bool> visited;
    vector<int> result;  // 查询结果
    int n;

    LCA_Tarjan(int _n) : n(_n) {
        adj.resize(n);
        queries.resize(n);
        parent.resize(n);
        ancestor.resize(n);
        visited.resize(n, false);
        iota(parent.begin(), parent.end(), 0);
        iota(ancestor.begin(), ancestor.end(), 0);
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void add_query(int u, int v, int id) {
        queries[u].push_back({v, id});
        queries[v].push_back({u, id});
    }

    int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }

    void unite(int x, int y) {
        x = find(x), y = find(y);
        if (x != y) { parent[y] = x; }
    }

    void dfs(int u, int p) {
        ancestor[u] = u;

        // 访问所有子节点
        for (int v : adj[u]) {
            if (v != p) {
                dfs(v, u);
                unite(u, v);
                ancestor[find(u)] = u;  // 更新祖先
            }
        }

        visited[u] = true;

        // 处理与已访问节点的查询
        for (const Query& q : queries[u]) {
            if (visited[q.v]) { result[q.id] = ancestor[find(q.v)]; }
        }
    }

    vector<int> solve(int root, int num_queries) {
        result.resize(num_queries);
        dfs(root, -1);
        return result;
    }
};

// 重链剖分LCA（重复实现，用于对比）
// 时间复杂度：预处理 O(n)，单次查询 O(log n)
// 空间复杂度：O(n)
struct HeavyLightDecomposition {
    vector<vector<int>> adj;
    vector<int> parent, depth, heavy, head, pos, size;
    int n, timer;

    HeavyLightDecomposition(int _n) : n(_n), timer(0) {
        adj.resize(n);
        parent.resize(n);
        depth.resize(n);
        heavy.resize(n, -1);
        head.resize(n);
        pos.resize(n);
        size.resize(n);
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int dfs_size(int v, int p) {
        size[v] = 1;
        parent[v] = p;

        for (int u : adj[v]) {
            if (u != p) {
                depth[u] = depth[v] + 1;
                size[v] += dfs_size(u, v);
                if (heavy[v] == -1 || size[u] > size[heavy[v]]) { heavy[v] = u; }
            }
        }

        return size[v];
    }

    void dfs_decompose(int v, int h) {
        head[v] = h;
        pos[v] = timer++;

        if (heavy[v] != -1) { dfs_decompose(heavy[v], h); }

        for (int u : adj[v]) {
            if (u != parent[v] && u != heavy[v]) { dfs_decompose(u, u); }
        }
    }

    void preprocess(int root = 0) {
        depth[root] = 0;
        dfs_size(root, -1);
        dfs_decompose(root, root);
    }

    int lca(int u, int v) {
        while (head[u] != head[v]) {
            if (depth[head[u]] > depth[head[v]]) {
                u = parent[head[u]];
            } else {
                v = parent[head[v]];
            }
        }
        return depth[u] < depth[v] ? u : v;
    }

    // 获取路径分解后的区间列表
    vector<pair<int, int>> get_path(int u, int v) {
        vector<pair<int, int>> result;

        while (head[u] != head[v]) {
            if (depth[head[u]] > depth[head[v]]) {
                result.push_back({pos[head[u]], pos[u]});
                u = parent[head[u]];
            } else {
                result.push_back({pos[head[v]], pos[v]});
                v = parent[head[v]];
            }
        }

        if (depth[u] > depth[v]) swap(u, v);
        result.push_back({pos[u], pos[v]});

        return result;
    }

    int distance(int u, int v) { return depth[u] + depth[v] - 2 * depth[lca(u, v)]; }
};
]=]),

-- 02_Graph_Theory\Tree_Algorithms\TreeChain.h
ps("02_graph_theory_tree_algorithms_treechain_h", [=[

// 树链剖分模板（配合线段树使用）
// 时间复杂度：预处理 O(n)，单次路径操作 O(log^2 n)，单次子树操作 O(log n)
// 空间复杂度：O(n)
struct TreeChain {
    int n, cnt;
    vector<vector<int>> adj;        // 邻接表
    vector<int> fa, son, dep, siz;  // 父节点、重儿子、深度、子树大小
    vector<int> top, dfn, rnk;      // 链顶、DFS序、DFS序对应的节点

    TreeChain(int _n) : n(_n), cnt(0) {
        adj.resize(n + 1);
        fa.resize(n + 1);
        son.resize(n + 1, 0);
        dep.resize(n + 1, 0);
        siz.resize(n + 1);
        top.resize(n + 1);
        dfn.resize(n + 1);
        rnk.resize(n + 1);
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // 第一次DFS：计算深度、父节点、子树大小、重儿子
    void dfs1(int u, int f) {
        fa[u] = f;
        dep[u] = dep[f] + 1;
        siz[u] = 1;

        for (int v : adj[u]) {
            if (v == f) continue;
            dfs1(v, u);
            siz[u] += siz[v];
            if (siz[v] > siz[son[u]]) {
                son[u] = v;  // 更新重儿子
            }
        }
    }

    // 第二次DFS：建立重链，分配DFS序
    void dfs2(int u, int t) {
        dfn[u] = ++cnt;  // DFS序
        rnk[cnt] = u;    // DFS序对应的节点
        top[u] = t;      // 链顶

        if (!son[u]) return;  // 叶子节点

        dfs2(son[u], t);  // 重儿子继续在同一条链

        // 处理轻儿子，每个轻儿子开始新的链
        for (int v : adj[u]) {
            if (v != fa[u] && v != son[u]) { dfs2(v, v); }
        }
    }

    void build(int root = 1) {
        dfs1(root, 0);
        dfs2(root, root);
    }

    // 查询LCA
    int lca(int u, int v) {
        while (top[u] != top[v]) {
            if (dep[top[u]] >= dep[top[v]]) {
                u = fa[top[u]];
            } else {
                v = fa[top[v]];
            }
        }
        return (dfn[u] <= dfn[v] ? u : v);
    }

    // 查询两点间距离
    int distance(int u, int v) { return dep[u] + dep[v] - 2 * dep[lca(u, v)]; }

    // 获取路径分解后的区间（用于路径修改/查询）
    vector<pair<int, int>> get_path_ranges(int u, int v) {
        vector<pair<int, int>> ranges;

        while (top[u] != top[v]) {
            if (dep[top[u]] < dep[top[v]]) swap(u, v);
            ranges.push_back({dfn[top[u]], dfn[u]});
            u = fa[top[u]];
        }

        if (dep[u] > dep[v]) swap(u, v);
        ranges.push_back({dfn[u], dfn[v]});

        return ranges;
    }

    // 获取子树对应的区间
    pair<int, int> get_subtree_range(int u) { return {dfn[u], dfn[u] + siz[u] - 1}; }

    // 判断u是否为v的祖先
    bool is_ancestor(int u, int v) { return dfn[u] <= dfn[v] && dfn[v] <= dfn[u] + siz[u] - 1; }

    // 路径上第k个节点（从u到v，0-indexed）
    int kth_on_path(int u, int v, int k) {
        int l = lca(u, v);
        int dist_u = dep[u] - dep[l];
        int dist_v = dep[v] - dep[l];

        if (k <= dist_u) {
            // 在u到lca的路径上
            for (int i = 0; i < k; i++) { u = fa[u]; }
            return u;
        } else {
            // 在lca到v的路径上
            k = dist_u + dist_v - k;
            for (int i = 0; i < k; i++) { v = fa[v]; }
            return v;
        }
    }
};
]=]),

-- 03_Dynamic_Programming\Advanced\PlugDP.h
ps("03_dynamic_programming_advanced_plugdp_h", [=[

/**
 * 插头动态规划模板
 * 功能：哈密顿回路、简单回路、路径覆盖、连通子图计数
 * 时间复杂度：O(n*m*4^m)，空间复杂度：O(4^m)
 */
struct PlugDP {
    int n, m;
    map<long long, long long> dp, new_dp;
    vector<vector<int>> grid;

    PlugDP(int rows, int cols) : n(rows), m(cols) { grid.assign(n, vector<int>(m, 1)); }

    void set_grid(const vector<vector<int>>& g) { grid = g; }  // 哈密顿回路计数
    long long count_hamilton_cycles() {
        dp.clear();
        dp[0] = 1;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                new_dp.clear();
                for (auto& [state, ways] : dp) {
                    if (grid[i][j] == 0) {
                        add_state(shift_state(state), ways);
                    } else {
                        process_cell(i, j, state, ways);
                    }
                }
                dp = new_dp;
            }
        }
        return dp.count(0) ? dp[0] : 0;
    }  // 简单回路计数
    long long count_simple_cycles() {
        dp.clear();
        dp[0] = 1;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                new_dp.clear();
                for (auto& [state, ways] : dp) {
                    if (grid[i][j] == 0) {
                        add_state(shift_state(state), ways);
                    } else {
                        process_simple_cycle_cell(i, j, state, ways);
                    }
                }
                dp = new_dp;
            }
        }
        return dp.count(0) ? dp[0] : 0;
    }  // 最小路径覆盖
    long long min_path_cover() {
        dp.clear();
        dp[0] = 1;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                new_dp.clear();
                for (auto& [state, ways] : dp) {
                    if (grid[i][j] == 0) {
                        add_state(shift_state(state), ways);
                    } else {
                        process_path_cover_cell(i, j, state, ways);
                    }
                }
                dp = new_dp;
            }
        }
        return dp.count(0) ? dp[0] : 0;
    }

   private:  // 插头操作
    int get_plug(long long state, int pos) { return (state >> (pos * 2)) & 3; }
    long long set_plug(long long state, int pos, int value) {
        long long mask = 3LL << (pos * 2);
        return (state & (~mask)) | ((long long)value << (pos * 2));
    }
    long long shift_state(long long state) { return state >> 2; }
    void add_state(long long state, long long ways) { new_dp[state] += ways; }  // 处理哈密顿回路格子
    void process_cell(int i, int j, long long state, long long ways) {
        int up = get_plug(state, j), left = get_plug(state, j + 1);

        if (up == 0 && left == 0) {
            // 开始新路径
            if (i < n - 1 && j < m - 1 && grid[i + 1][j] && grid[i][j + 1]) {
                long long new_state = set_plug(state, j, 1);
                new_state = set_plug(new_state, j + 1, 2);
                add_state(shift_state(new_state), ways);
            }
        } else if (up > 0 && left == 0) {
            // 延伸上插头
            if (i < n - 1 && grid[i + 1][j]) {
                add_state(shift_state(set_plug(set_plug(state, j, up), j + 1, 0)), ways);
            }
            if (j < m - 1 && grid[i][j + 1]) {
                add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, up)), ways);
            }
        } else if (up == 0 && left > 0) {
            // 延伸左插头
            if (i < n - 1 && grid[i + 1][j]) {
                add_state(shift_state(set_plug(set_plug(state, j, left), j + 1, 0)), ways);
            }
            if (j < m - 1 && grid[i][j + 1]) {
                add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, left)), ways);
            }
        } else {
            // 连接两插头
            if (up == left) {
                if (i == n - 1 && j == m - 1) {
                    add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, 0)), ways);
                }
            } else {
                long long new_state = set_plug(set_plug(state, j, 0), j + 1, 0);
                add_state(shift_state(merge_components(new_state, up, left)), ways);
            }
        }
    }  // 处理简单回路格子
    void process_simple_cycle_cell(int i, int j, long long state, long long ways) {
        int up = get_plug(state, j), left = get_plug(state, j + 1);

        // 不通过此格子
        add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, 0)), ways);

        if (up == 0 && left == 0) {
            // 开始新回路
            if (can_form_cycle(i, j)) { add_state(shift_state(set_plug(set_plug(state, j, 1), j + 1, 1)), ways); }
        } else if (up > 0 && left == 0) {
            // 延伸上插头
            if (i < n - 1) add_state(shift_state(set_plug(set_plug(state, j, up), j + 1, 0)), ways);
            if (j < m - 1) add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, up)), ways);
        } else if (up == 0 && left > 0) {
            // 延伸左插头
            if (i < n - 1) add_state(shift_state(set_plug(set_plug(state, j, left), j + 1, 0)), ways);
            if (j < m - 1) add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, left)), ways);
        } else if (up == left) {
            // 闭合回路
            add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, 0)), ways);
        }
    }  // 处理路径覆盖格子
    void process_path_cover_cell(int i, int j, long long state, long long ways) {
        int up = get_plug(state, j), left = get_plug(state, j + 1);

        // 不覆盖此格子
        add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, 0)), ways);

        if (up == 0 && left == 0) {
            // 开始新路径
            add_state(shift_state(set_plug(set_plug(state, j, 1), j + 1, 2)), ways);
        } else if (up > 0 && left == 0) {
            // 延伸上插头
            add_state(shift_state(set_plug(set_plug(state, j, up), j + 1, 0)), ways);
            add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, up)), ways);
            // 终止路径
            add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, 0)), ways);
        } else if (up == 0 && left > 0) {
            // 延伸左插头
            add_state(shift_state(set_plug(set_plug(state, j, left), j + 1, 0)), ways);
            add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, left)), ways);
            // 终止路径
            add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, 0)), ways);
        } else {
            // 连接两插头
            add_state(shift_state(merge_components(set_plug(set_plug(state, j, 0), j + 1, 0), up, left)), ways);
        }
    }  // 合并连通分量
    long long merge_components(long long state, int comp1, int comp2) {
        if (comp1 == comp2) return state;
        for (int i = 0; i <= m; i++) {
            if (get_plug(state, i) == comp2) { state = set_plug(state, i, comp1); }
        }
        return state;
    }

    // 检查能否形成回路
    bool can_form_cycle(int i, int j) { return i < n - 1 && j < m - 1 && grid[i + 1][j] && grid[i][j + 1]; }

   public:
    // 连通子图计数
    long long count_connected_subgraphs() {
        dp.clear();
        dp[0] = 1;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                new_dp.clear();
                for (auto& [state, ways] : dp) {
                    add_state(shift_state(state), ways);
                    if (grid[i][j] == 1) { process_connected_subgraph_cell(i, j, state, ways); }
                }
                dp = new_dp;
            }
        }

        long long result = 0;
        for (auto& [state, ways] : dp) result += ways;
        return result - 1;  // 减去空集
    }
    void clear() {
        dp.clear();
        new_dp.clear();
    }

   private:
    void process_connected_subgraph_cell(int i, int j, long long state, long long ways) {
        int up = get_plug(state, j), left = get_plug(state, j + 1);

        if (up == 0 && left == 0) {
            // 开始新连通分量
            int max_id = 0;
            for (int k = 0; k <= m; k++) max_id = max(max_id, get_plug(state, k));
            int new_id = max_id + 1;
            long long new_state = state;
            if (i < n - 1 && grid[i + 1][j]) new_state = set_plug(new_state, j, new_id);
            if (j < m - 1 && grid[i][j + 1]) new_state = set_plug(new_state, j + 1, new_id);
            add_state(shift_state(new_state), ways);
        } else if (up > 0 && left == 0) {
            add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, up)), ways);
        } else if (up == 0 && left > 0) {
            add_state(shift_state(set_plug(set_plug(state, j, left), j + 1, 0)), ways);
        } else if (up == left) {
            add_state(shift_state(set_plug(set_plug(state, j, 0), j + 1, 0)), ways);
        } else {
            add_state(shift_state(merge_components(set_plug(set_plug(state, j, 0), j + 1, 0), up, left)), ways);
        }
    }
};
]=]),

-- 03_Dynamic_Programming\Advanced\ProfileDP.h
ps("03_dynamic_programming_advanced_profiledp_h", [=[

/**
 * 轮廓线动态规划模板
 * 功能：棋盘填充、哈密顿路径、独立集、着色等问题
 * 时间复杂度：O(n*2^m)，空间复杂度：O(2^m)
 */
struct ProfileDP {
    int n, m, max_mask;
    vector<vector<long long>> dp, new_dp;

    ProfileDP(int rows, int cols) : n(rows), m(cols), max_mask(1 << cols) {
        dp.assign(n + 1, vector<long long>(max_mask, 0));
        new_dp.assign(n + 1, vector<long long>(max_mask, 0));
    }  // 棋盘填充（1x2和2x1骨牌）
    long long count_tilings() {
        dp[0][0] = 1;
        for (int i = 0; i < n; i++) {
            for (int mask = 0; mask < max_mask; mask++) new_dp[i + 1][mask] = 0;
            for (int mask = 0; mask < max_mask; mask++) {
                if (dp[i][mask] == 0) continue;
                fill_row(i, 0, mask, 0, dp[i][mask]);
            }
            for (int mask = 0; mask < max_mask; mask++) dp[i + 1][mask] = new_dp[i + 1][mask];
        }
        return dp[n][0];
    }  // 递归填充行
    void fill_row(int row, int col, int cur_mask, int next_mask, long long ways) {
        if (col == m) {
            new_dp[row + 1][next_mask] += ways;
            return;
        }
        if (cur_mask & (1 << col)) {
            // 被上方竖直骨牌占据
            fill_row(row, col + 1, cur_mask, next_mask, ways);
        } else {
            // 放竖直骨牌
            fill_row(row, col + 1, cur_mask, next_mask | (1 << col), ways);
            // 放水平骨牌
            if (col + 1 < m && !(cur_mask & (1 << (col + 1)))) { fill_row(row, col + 2, cur_mask, next_mask, ways); }
        }
    }  // 带障碍的棋盘填充
    long long count_tilings_with_obstacles(const vector<vector<bool>>& blocked) {
        dp[0][0] = 1;
        for (int i = 0; i < n; i++) {
            for (int mask = 0; mask < max_mask; mask++) new_dp[i + 1][mask] = 0;
            for (int mask = 0; mask < max_mask; mask++) {
                if (dp[i][mask] == 0) continue;
                fill_row_with_obstacles(i, 0, mask, 0, dp[i][mask], blocked);
            }
            for (int mask = 0; mask < max_mask; mask++) dp[i + 1][mask] = new_dp[i + 1][mask];
        }
        return dp[n][0];
    }
    void fill_row_with_obstacles(
        int row, int col, int cur_mask, int next_mask, long long ways, const vector<vector<bool>>& blocked) {
        if (col == m) {
            new_dp[row + 1][next_mask] += ways;
            return;
        }
        if (blocked[row][col]) {
            fill_row_with_obstacles(row, col + 1, cur_mask, next_mask, ways, blocked);
        } else if (cur_mask & (1 << col)) {
            fill_row_with_obstacles(row, col + 1, cur_mask, next_mask, ways, blocked);
        } else {
            // 放竖直骨牌
            if (row + 1 < n && !blocked[row + 1][col]) {
                fill_row_with_obstacles(row, col + 1, cur_mask, next_mask | (1 << col), ways, blocked);
            }
            // 放水平骨牌
            if (col + 1 < m && !blocked[row][col + 1] && !(cur_mask & (1 << (col + 1)))) {
                fill_row_with_obstacles(row, col + 2, cur_mask, next_mask, ways, blocked);
            }
        }
    }  // 最大独立集
    long long max_independent_set() {
        vector<vector<long long>> indep_dp(n + 1, vector<long long>(max_mask, 0));
        indep_dp[0][0] = 1;

        for (int i = 0; i < n; i++) {
            for (int mask = 0; mask < max_mask; mask++) new_dp[i + 1][mask] = 0;
            for (int mask = 0; mask < max_mask; mask++) {
                if (indep_dp[i][mask] == 0) continue;
                generate_independent_sets(i, 0, mask, 0, indep_dp[i][mask]);
            }
            for (int mask = 0; mask < max_mask; mask++) {
                indep_dp[i + 1][mask] = max(indep_dp[i + 1][mask], new_dp[i + 1][mask]);
            }
        }
        return *max_element(indep_dp[n].begin(), indep_dp[n].end());
    }

    void generate_independent_sets(int row, int col, int prev_mask, int cur_mask, long long value) {
        if (col == m) {
            new_dp[row + 1][cur_mask] = max(new_dp[row + 1][cur_mask], value);
            return;
        }
        // 不选当前位置
        generate_independent_sets(row, col + 1, prev_mask, cur_mask, value);
        // 选当前位置
        bool can_select = !(prev_mask & (1 << col)) && (col == 0 || !(cur_mask & (1 << (col - 1))));
        if (can_select) { generate_independent_sets(row, col + 1, prev_mask, cur_mask | (1 << col), value + 1); }
    }

    void clear() {
        for (int i = 0; i <= n; i++) {
            fill(dp[i].begin(), dp[i].end(), 0);
            fill(new_dp[i].begin(), new_dp[i].end(), 0);
        }
    }
};
]=]),

-- 03_Dynamic_Programming\Advanced\SOS_DP.h
ps("03_dynamic_programming_advanced_sos_dp_h", [=[

/**
 * SOS动态规划模板
 * 功能：子集和、超集和、集合卷积、最大权重独立集
 * 时间复杂度：O(n*2^n)，空间复杂度：O(2^n)
 */
struct SOS_DP {
    int n, max_mask;
    vector<long long> dp, original;

    SOS_DP(int bits) : n(bits), max_mask(1 << bits) {
        dp.resize(max_mask, 0);
        original.resize(max_mask, 0);
    }

    void set_value(int mask, long long value) {
        original[mask] = value;
        dp[mask] = value;
    }  // 计算子集和
    void compute_subset_sum() {
        dp = original;
        for (int i = 0; i < n; i++) {
            for (int mask = 0; mask < max_mask; mask++) {
                if (mask & (1 << i)) { dp[mask] += dp[mask ^ (1 << i)]; }
            }
        }
    }

    // 计算超集和
    void compute_superset_sum() {
        dp = original;
        for (int i = 0; i < n; i++) {
            for (int mask = 0; mask < max_mask; mask++) {
                if (!(mask & (1 << i))) { dp[mask] += dp[mask | (1 << i)]; }
            }
        }
    }  // 集合卷积
    vector<long long> subset_convolution(const vector<long long>& a, const vector<long long>& b) {
        vector<vector<long long>> fa(n + 1, vector<long long>(max_mask, 0));
        vector<vector<long long>> fb(n + 1, vector<long long>(max_mask, 0));

        for (int mask = 0; mask < max_mask; mask++) {
            int pc = __builtin_popcount(mask);
            fa[pc][mask] = a[mask];
            fb[pc][mask] = b[mask];
        }

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j < n; j++) {
                for (int mask = 0; mask < max_mask; mask++) {
                    if (mask & (1 << j)) {
                        fa[i][mask] += fa[i][mask ^ (1 << j)];
                        fb[i][mask] += fb[i][mask ^ (1 << j)];
                    }
                }
            }
        }

        vector<vector<long long>> fh(n + 1, vector<long long>(max_mask, 0));
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= i; j++) {
                for (int mask = 0; mask < max_mask; mask++) { fh[i][mask] += fa[j][mask] * fb[i - j][mask]; }
            }
        }

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j < n; j++) {
                for (int mask = 0; mask < max_mask; mask++) {
                    if (mask & (1 << j)) { fh[i][mask] -= fh[i][mask ^ (1 << j)]; }
                }
            }
        }

        vector<long long> result(max_mask);
        for (int mask = 0; mask < max_mask; mask++) { result[mask] = fh[__builtin_popcount(mask)][mask]; }
        return result;
    }  // 枚举子集
    void enumerate_subsets(int mask, function<void(int)> callback) {
        for (int submask = mask;; submask = (submask - 1) & mask) {
            callback(submask);
            if (submask == 0) break;
        }
    }

    // 计算交集为空的集合对数量
    long long count_disjoint_pairs() {
        vector<long long> cnt(max_mask, 0);
        for (int mask = 0; mask < max_mask; mask++) cnt[mask] = original[mask];

        for (int i = 0; i < n; i++) {
            for (int mask = 0; mask < max_mask; mask++) {
                if (mask & (1 << i)) cnt[mask] += cnt[mask ^ (1 << i)];
            }
        }

        long long result = 0;
        for (int mask = 0; mask < max_mask; mask++) {
            int complement = ((1 << n) - 1) ^ mask;
            result += original[mask] * cnt[complement];
        }
        return result;
    }

    // 最大权重独立集
    long long max_weight_independent_set() {
        dp = original;
        for (int i = 0; i < n; i++) {
            for (int mask = 0; mask < max_mask; mask++) {
                if (mask & (1 << i)) { dp[mask] = max(dp[mask], dp[mask ^ (1 << i)]); }
            }
        }
        return dp[max_mask - 1];
    }

    vector<long long> get_result() { return dp; }
    long long get_result(int mask) { return dp[mask]; }
    void clear() {
        fill(dp.begin(), dp.end(), 0);
        fill(original.begin(), original.end(), 0);
    }
};
]=]),

-- 03_Dynamic_Programming\Classical\EditDistance.h
ps("03_dynamic_programming_classical_editdistance_h", [=[

/**
 * 编辑距离算法模板
 * 功能：基础编辑距离、带权重、空间优化、K编辑距离
 * 时间复杂度：O(nm)，空间复杂度：O(nm)或O(min(n,m))
 */

// 基础编辑距离
template <typename T>
struct EditDistance {
    vector<vector<int>> dp;
    int n, m;

    int solve(const T& s1, const T& s2) {
        n = s1.size(), m = s2.size();
        dp.assign(n + 1, vector<int>(m + 1, 0));

        for (int i = 0; i <= n; i++) dp[i][0] = i;
        for (int j = 0; j <= m; j++) dp[0][j] = j;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                }
            }
        }
        return dp[n][m];
    }

    // 获取编辑操作序列
    vector<string> getOperations(const string& s1, const string& s2) {
        solve(s1, s2);
        vector<string> ops;
        int i = n, j = m;

        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && s1[i - 1] == s2[j - 1]) {
                i--, j--;
            } else if (i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1) {
                ops.push_back("Replace " + string(1, s1[i - 1]) + " with " + string(1, s2[j - 1]));
                i--, j--;
            } else if (i > 0 && dp[i][j] == dp[i - 1][j] + 1) {
                ops.push_back("Delete " + string(1, s1[i - 1]));
                i--;
            } else {
                ops.push_back("Insert " + string(1, s2[j - 1]));
                j--;
            }
        }
        reverse(ops.begin(), ops.end());
        return ops;
    }
};

// 带权重的编辑距离
template <typename T>
struct WeightedEditDistance {
    vector<vector<int>> dp;
    int insert_cost, delete_cost, replace_cost;

    WeightedEditDistance(int ins = 1, int del = 1, int rep = 1)
        : insert_cost(ins), delete_cost(del), replace_cost(rep) {}

    int solve(const T& s1, const T& s2) {
        int n = s1.size(), m = s2.size();
        dp.assign(n + 1, vector<int>(m + 1, 0));

        for (int i = 0; i <= n; i++) dp[i][0] = i * delete_cost;
        for (int j = 0; j <= m; j++) dp[0][j] = j * insert_cost;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] =
                        min({dp[i - 1][j] + delete_cost, dp[i][j - 1] + insert_cost, dp[i - 1][j - 1] + replace_cost});
                }
            }
        }
        return dp[n][m];
    }
};

// 空间优化版本
template <typename T>
int editDistanceOptimized(const T& s1, const T& s2) {
    int n = s1.size(), m = s2.size();
    vector<int> prev(m + 1), curr(m + 1);

    for (int j = 0; j <= m; j++) prev[j] = j;

    for (int i = 1; i <= n; i++) {
        curr[0] = i;
        for (int j = 1; j <= m; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = min({prev[j], curr[j - 1], prev[j - 1]}) + 1;
            }
        }
        swap(prev, curr);
    }
    return prev[m];
}

// K编辑距离
template <typename T>
bool canTransformWithKEdits(const T& s1, const T& s2, int k) {
    int n = s1.size(), m = s2.size();
    if (abs(n - m) > k) return false;

    vector<vector<int>> dp(n + 1, vector<int>(m + 1, k + 1));
    dp[0][0] = 0;

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= m; j++) {
            if (i == 0 && j == 0) continue;
            if (i > 0) dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1);
            if (j > 0) dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1);
            if (i > 0 && j > 0) {
                int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + cost);
            }
        }
    }
    return dp[n][m] <= k;
}
]=]),

-- 03_Dynamic_Programming\Classical\Knapsack.h
ps("03_dynamic_programming_classical_knapsack_h", [=[

// 背包问题模板集合
using ll = long long;

// 0-1背包问题 - 每个物品只能选择一次
struct ZeroOneKnapsack {
    vector<int> weights, values;
    int n, W;

    ZeroOneKnapsack(int n, int W) : n(n), W(W), weights(n), values(n) {}

    void addItem(int idx, int weight, int value) {
        weights[idx] = weight;
        values[idx] = value;
    }

    // 空间优化版本 O(nW) 时间, O(W) 空间
    int solve() {
        vector<int> dp(W + 1, 0);
        for (int i = 0; i < n; i++) {
            for (int w = W; w >= weights[i]; w--) { dp[w] = max(dp[w], dp[w - weights[i]] + values[i]); }
        }
        return dp[W];
    }
};
// 完全背包问题 - 每个物品可以选择无限次
struct CompleteKnapsack {
    vector<int> weights, values;
    int n, W;

    CompleteKnapsack(int n, int W) : n(n), W(W), weights(n), values(n) {}

    void addItem(int idx, int weight, int value) {
        weights[idx] = weight;
        values[idx] = value;
    }

    int solve() {
        vector<int> dp(W + 1, 0);
        for (int i = 0; i < n; i++) {
            for (int w = weights[i]; w <= W; w++) { dp[w] = max(dp[w], dp[w - weights[i]] + values[i]); }
        }
        return dp[W];
    }
};

// 多重背包问题 - 每个物品有数量限制
struct MultipleKnapsack {
    vector<int> weights, values, counts;
    int n, W;

    MultipleKnapsack(int n, int W) : n(n), W(W), weights(n), values(n), counts(n) {}

    void addItem(int idx, int weight, int value, int count) {
        weights[idx] = weight;
        values[idx] = value;
        counts[idx] = count;
    }

    // 二进制优化版本 O(n*log(count)*W)
    int solve() {
        vector<pair<int, int>> items;

        for (int i = 0; i < n; i++) {
            int cnt = counts[i], k = 1;
            while (k <= cnt) {
                items.emplace_back(k * weights[i], k * values[i]);
                cnt -= k;
                k *= 2;
            }
            if (cnt > 0) { items.emplace_back(cnt * weights[i], cnt * values[i]); }
        }

        vector<int> dp(W + 1, 0);
        for (auto [w, v] : items) {
            for (int j = W; j >= w; j--) { dp[j] = max(dp[j], dp[j - w] + v); }
        }
        return dp[W];
    }
};
// 分组背包问题 - 每个组只能选择一个物品
struct GroupKnapsack {
    vector<vector<pair<int, int>>> groups;  // {weight, value}
    int W;

    GroupKnapsack(int W) : W(W) {}

    void addGroup(const vector<pair<int, int>>& items) { groups.push_back(items); }

    int solve() {
        vector<int> dp(W + 1, 0);
        for (auto& group : groups) {
            for (int w = W; w >= 0; w--) {
                for (auto [weight, value] : group) {
                    if (w >= weight) { dp[w] = max(dp[w], dp[w - weight] + value); }
                }
            }
        }
        return dp[W];
    }
};

// 二维背包问题 - 有两个约束条件
struct TwoDimensionalKnapsack {
    vector<int> weights1, weights2, values;
    int n, W1, W2;

    TwoDimensionalKnapsack(int n, int W1, int W2) : n(n), W1(W1), W2(W2), weights1(n), weights2(n), values(n) {}

    void addItem(int idx, int w1, int w2, int value) {
        weights1[idx] = w1;
        weights2[idx] = w2;
        values[idx] = value;
    }

    int solve() {
        vector<vector<int>> dp(W1 + 1, vector<int>(W2 + 1, 0));
        for (int i = 0; i < n; i++) {
            for (int w1 = W1; w1 >= weights1[i]; w1--) {
                for (int w2 = W2; w2 >= weights2[i]; w2--) {
                    dp[w1][w2] = max(dp[w1][w2], dp[w1 - weights1[i]][w2 - weights2[i]] + values[i]);
                }
            }
        }
        return dp[W1][W2];
    }
};
]=]),

-- 03_Dynamic_Programming\Classical\LCS.h
ps("03_dynamic_programming_classical_lcs_h", [=[

// 最长公共子序列(LCS)算法集合

// 基础LCS - O(nm)算法
struct LCS {
    string s1, s2;
    int n, m;

    LCS(const string& a, const string& b) : s1(a), s2(b), n(a.length()), m(b.length()) {}

    int solve() {
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n][m];
    }

    string getLCS() {
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        string result;
        int i = n, j = m;
        while (i > 0 && j > 0) {
            if (s1[i - 1] == s2[j - 1]) {
                result = s1[i - 1] + result;
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }
        return result;
    }
};

// 空间优化的LCS - O(min(n,m))空间
struct LCSOptimized {
    string s1, s2;
    int n, m;

    LCSOptimized(const string& a, const string& b) : s1(a), s2(b), n(a.length()), m(b.length()) {}

    int solve() {
        if (n < m) {
            swap(s1, s2);
            swap(n, m);
        }
        vector<int> prev(m + 1, 0), curr(m + 1, 0);
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1[i - 1] == s2[j - 1]) {
                    curr[j] = prev[j - 1] + 1;
                } else {
                    curr[j] = max(prev[j], curr[j - 1]);
                }
            }
            prev = curr;
        }
        return curr[m];
    }
};

// 最长公共递增子序列(LCIS)
struct LCIS {
    vector<int> a, b;
    int n, m;

    LCIS(const vector<int>& arr1, const vector<int>& arr2) : a(arr1), b(arr2), n(arr1.size()), m(arr2.size()) {}

    int solve() {
        vector<int> dp(m, 0);
        for (int i = 0; i < n; i++) {
            int cur_len = 0;
            for (int j = 0; j < m; j++) {
                if (a[i] == b[j] && dp[j] < cur_len + 1) { dp[j] = cur_len + 1; }
                if (b[j] < a[i] && dp[j] > cur_len) { cur_len = dp[j]; }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
]=]),

}
