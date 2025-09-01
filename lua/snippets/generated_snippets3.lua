-- Auto-generated LuaSnip snippets
local ls = require("luasnip")
local ps = ls.parser.parse_snippet

return {

-- 02_Graph_Theory\Connectivity\TarjanCutVertex.h
ps("02_graph_theory_connectivity_tarjancutvertex_h", [=[

// Tarjan算法求割点
struct TarjanCutVertex {
    int n, time_stamp;
    vector<vector<int>> graph;
    vector<int> dfn, low;
    vector<bool> is_cut;
    vector<int> cut_vertices;

    TarjanCutVertex(int sz) : n(sz), time_stamp(0) {
        graph.resize(n);
        dfn.resize(n, -1);
        low.resize(n, -1);
        is_cut.resize(n, false);
    }

    void add_edge(int u, int v) {
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    void tarjan(int u, int parent) {
        dfn[u] = low[u] = time_stamp++;
        int children = 0;

        for (int v : graph[u]) {
            if (v == parent) continue;

            if (dfn[v] == -1) {
                children++;
                tarjan(v, u);
                low[u] = min(low[u], low[v]);

                // 根节点的判断
                if (parent == -1 && children > 1) { is_cut[u] = true; }
                // 非根节点的判断
                if (parent != -1 && low[v] >= dfn[u]) { is_cut[u] = true; }
            } else {
                low[u] = min(low[u], dfn[v]);
            }
        }
    }

    void run() {
        for (int i = 0; i < n; i++) {
            if (dfn[i] == -1) { tarjan(i, -1); }
        }

        for (int i = 0; i < n; i++) {
            if (is_cut[i]) { cut_vertices.push_back(i); }
        }
    }

    vector<int> get_cut_vertices() { return cut_vertices; }

    bool is_cut_vertex(int u) { return is_cut[u]; }

    // 构建点双连通分量
    vector<vector<int>> get_vertex_bcc() {
        vector<vector<int>> bcc;
        vector<int> stk;

        function<void(int, int)> dfs = [&](int u, int parent) {
            dfn[u] = low[u] = time_stamp++;
            stk.push_back(u);

            for (int v : graph[u]) {
                if (v == parent) continue;

                if (dfn[v] == -1) {
                    int stk_size = stk.size();
                    dfs(v, u);
                    low[u] = min(low[u], low[v]);

                    if (low[v] >= dfn[u]) {
                        vector<int> component;
                        while (stk.size() > stk_size) {
                            component.push_back(stk.back());
                            stk.pop_back();
                        }
                        component.push_back(u);
                        bcc.push_back(component);
                    }
                } else if (dfn[v] < dfn[u]) {
                    low[u] = min(low[u], dfn[v]);
                }
            }
        };

        // 重置时间戳
        time_stamp = 0;
        fill(dfn.begin(), dfn.end(), -1);
        fill(low.begin(), low.end(), -1);

        for (int i = 0; i < n; i++) {
            if (dfn[i] == -1) {
                dfs(i, -1);
                if (!stk.empty()) {
                    bcc.push_back(stk);
                    stk.clear();
                }
            }
        }

        return bcc;
    }
};

// 使用示例：
// TarjanCutVertex cut(n);
// cut.add_edge(u, v);
// cut.run();
// vector<int> cut_vertices = cut.get_cut_vertices();
// vector<vector<int>> bcc = cut.get_vertex_bcc();
]=]),

-- 02_Graph_Theory\Connectivity\TarjanSCC.h
ps("02_graph_theory_connectivity_tarjanscc_h", [=[

// Tarjan算法求强连通分量
struct TarjanSCC {
    int n, scc_cnt, time_stamp;
    vector<vector<int>> graph;
    vector<int> dfn, low, stk, in_stack, scc_id, scc_size;

    TarjanSCC(int sz) : n(sz), scc_cnt(0), time_stamp(0) {
        graph.resize(n);
        dfn.resize(n, -1);
        low.resize(n, -1);
        in_stack.resize(n, 0);
        scc_id.resize(n, -1);
        scc_size.resize(n, 0);
    }

    void add_edge(int u, int v) { graph[u].push_back(v); }

    void tarjan(int u) {
        dfn[u] = low[u] = time_stamp++;
        stk.push_back(u);
        in_stack[u] = 1;

        for (int v : graph[u]) {
            if (dfn[v] == -1) {
                tarjan(v);
                low[u] = min(low[u], low[v]);
            } else if (in_stack[v]) {
                low[u] = min(low[u], dfn[v]);
            }
        }

        if (dfn[u] == low[u]) {
            int v;
            do {
                v = stk.back();
                stk.pop_back();
                in_stack[v] = 0;
                scc_id[v] = scc_cnt;
                scc_size[scc_cnt]++;
            } while (v != u);
            scc_cnt++;
        }
    }

    void run() {
        for (int i = 0; i < n; i++) {
            if (dfn[i] == -1) { tarjan(i); }
        }
        // 重新调整大小
        scc_size.resize(scc_cnt);
    }

    // 构建缩点后的DAG
    vector<vector<int>> build_dag() {
        vector<vector<int>> dag(scc_cnt);
        vector<set<int>> edge_set(scc_cnt);

        for (int u = 0; u < n; u++) {
            for (int v : graph[u]) {
                int su = scc_id[u], sv = scc_id[v];
                if (su != sv && edge_set[su].find(sv) == edge_set[su].end()) {
                    dag[su].push_back(sv);
                    edge_set[su].insert(sv);
                }
            }
        }
        return dag;
    }

    bool is_strongly_connected() { return scc_cnt == 1; }

    int get_scc_count() { return scc_cnt; }
    int get_scc_id(int u) { return scc_id[u]; }
    int get_scc_size(int scc) { return scc_size[scc]; }
};

// 使用示例：
// TarjanSCC scc(n);
// scc.add_edge(u, v);
// scc.run();
// vector<vector<int>> dag = scc.build_dag();
]=]),

-- 02_Graph_Theory\Matching\BipartiteMatching.h
ps("02_graph_theory_matching_bipartitematching_h", [=[

// 二分图匹配模板

// 匈牙利算法 - 最大匹配
// 时间复杂度: O(V * E)
struct Hungarian {
    vector<vector<int>> g;
    vector<int> match;
    vector<bool> used;
    int n, m;  // 左部n个点，右部m个点

    Hungarian(int n, int m) : n(n), m(m), g(n), match(m, -1), used(n) {}

    void addEdge(int u, int v) { g[u].push_back(v); }

    bool dfs(int v) {
        if (used[v]) return false;
        used[v] = true;

        for (int to : g[v]) {
            if (match[to] == -1 || dfs(match[to])) {
                match[to] = v;
                return true;
            }
        }
        return false;
    }

    int maxMatching() {
        int result = 0;
        for (int v = 0; v < n; v++) {
            fill(used.begin(), used.end(), false);
            if (dfs(v)) result++;
        }
        return result;
    }

    // 获取匹配
    vector<pair<int, int>> getMatching() {
        vector<pair<int, int>> result;
        for (int i = 0; i < m; i++) {
            if (match[i] != -1) { result.emplace_back(match[i], i); }
        }
        return result;
    }
};

// Kuhn-Munkres算法 - 最大权匹配/最小权匹配
// 时间复杂度: O(V^3)
struct KuhnMunkres {
    vector<vector<int>> cost;
    vector<int> lx, ly, match;
    vector<bool> visx, visy;
    int n, slack;

    KuhnMunkres(int n) : n(n), cost(n, vector<int>(n, 0)), lx(n), ly(n), match(n, -1), visx(n), visy(n) {}

    void setCost(int u, int v, int w) { cost[u][v] = w; }

    bool dfs(int u) {
        visx[u] = true;
        for (int v = 0; v < n; v++) {
            if (visy[v]) continue;
            int tmp = lx[u] + ly[v] - cost[u][v];
            if (tmp == 0) {
                visy[v] = true;
                if (match[v] == -1 || dfs(match[v])) {
                    match[v] = u;
                    return true;
                }
            } else {
                slack = min(slack, tmp);
            }
        }
        return false;
    }

    int maxWeightMatching() {
        // 初始化顶标
        for (int i = 0; i < n; i++) {
            lx[i] = *max_element(cost[i].begin(), cost[i].end());
            ly[i] = 0;
        }

        for (int i = 0; i < n; i++) {
            while (true) {
                fill(visx.begin(), visx.end(), false);
                fill(visy.begin(), visy.end(), false);
                slack = INT_MAX;

                if (dfs(i)) break;

                // 修改顶标
                for (int j = 0; j < n; j++) {
                    if (visx[j]) lx[j] -= slack;
                    if (visy[j]) ly[j] += slack;
                }
            }
        }

        int result = 0;
        for (int i = 0; i < n; i++) {
            if (match[i] != -1) { result += cost[match[i]][i]; }
        }
        return result;
    }

    vector<pair<int, int>> getMatching() {
        vector<pair<int, int>> result;
        for (int i = 0; i < n; i++) {
            if (match[i] != -1) { result.emplace_back(match[i], i); }
        }
        return result;
    }
};

// Hopcroft-Karp算法 - O(E√V) 最大匹配
// ?????????????
struct HopcroftKarp {
    vector<vector<int>> g;
    vector<int> pairU, pairV, dist;
    int n, m, nil;

    HopcroftKarp(int n, int m) : n(n), m(m), nil(0), g(n + 1), pairU(n + 1), pairV(m + 1), dist(n + 1) {}

    void addEdge(int u, int v) {
        g[u + 1].push_back(v + 1);  // 1-indexed
    }

    bool bfs() {
        queue<int> q;
        for (int u = 1; u <= n; u++) {
            if (pairU[u] == nil) {
                dist[u] = 0;
                q.push(u);
            } else {
                dist[u] = INT_MAX;
            }
        }
        dist[nil] = INT_MAX;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            if (dist[u] < dist[nil]) {
                for (int v : g[u]) {
                    if (dist[pairV[v]] == INT_MAX) {
                        dist[pairV[v]] = dist[u] + 1;
                        q.push(pairV[v]);
                    }
                }
            }
        }
        return dist[nil] != INT_MAX;
    }

    bool dfs(int u) {
        if (u != nil) {
            for (int v : g[u]) {
                if (dist[pairV[v]] == dist[u] + 1) {
                    if (dfs(pairV[v])) {
                        pairV[v] = u;
                        pairU[u] = v;
                        return true;
                    }
                }
            }
            dist[u] = INT_MAX;
            return false;
        }
        return true;
    }

    int maxMatching() {
        fill(pairU.begin(), pairU.end(), nil);
        fill(pairV.begin(), pairV.end(), nil);

        int matching = 0;
        while (bfs()) {
            for (int u = 1; u <= n; u++) {
                if (pairU[u] == nil && dfs(u)) { matching++; }
            }
        }
        return matching;
    }

    vector<pair<int, int>> getMatching() {
        vector<pair<int, int>> result;
        for (int u = 1; u <= n; u++) {
            if (pairU[u] != nil) {
                result.emplace_back(u - 1, pairU[u] - 1);  // ??0-indexed
            }
        }
        return result;
    }
};

// Edmonds-Karp算法 - O(E^2V) 最大匹配
// ????? - ???Blossom??
struct Blossom {
    vector<vector<int>> g;
    vector<int> match, pre, base;
    vector<bool> used, blossom;
    queue<int> q;
    int n;

    Blossom(int n) : n(n), g(n), match(n, -1), pre(n), base(n), used(n), blossom(n) {}

    void addEdge(int u, int v) {
        g[u].push_back(v);
        g[v].push_back(u);
    }

    int lca(int u, int v) {
        fill(used.begin(), used.end(), false);
        while (true) {
            u = base[u];
            used[u] = true;
            if (match[u] == -1) break;
            u = pre[match[u]];
        }
        while (true) {
            v = base[v];
            if (used[v]) return v;
            v = pre[match[v]];
        }
    }

    void markPath(int v, int b, int children) {
        while (base[v] != b) {
            blossom[base[v]] = blossom[base[match[v]]] = true;
            pre[v] = children;
            children = match[v];
            v = pre[match[v]];
        }
    }

    void shrinkBlossom(int u, int v) {
        int b = lca(u, v);
        fill(blossom.begin(), blossom.end(), false);
        markPath(u, b, v);
        markPath(v, b, u);
        for (int i = 0; i < n; i++) {
            if (blossom[base[i]]) {
                base[i] = b;
                if (!used[i]) {
                    used[i] = true;
                    q.push(i);
                }
            }
        }
    }

    bool augment(int s) {
        fill(used.begin(), used.end(), false);
        fill(pre.begin(), pre.end(), -1);
        for (int i = 0; i < n; i++) base[i] = i;

        used[s] = true;
        q = queue<int>();
        q.push(s);

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int v : g[u]) {
                if (base[u] == base[v] || match[u] == v) continue;
                if (v == s || (match[v] != -1 && pre[match[v]] != -1)) {
                    shrinkBlossom(u, v);
                } else if (pre[v] == -1) {
                    pre[v] = u;
                    if (match[v] == -1) {
                        // 找到增广路径
                        int cur = v, next;
                        while (cur != -1) {
                            next = match[pre[cur]];
                            match[cur] = pre[cur];
                            match[pre[cur]] = cur;
                            cur = next;
                        }
                        return true;
                    } else {
                        used[match[v]] = true;
                        q.push(match[v]);
                    }
                }
            }
        }
        return false;
    }

    int maxMatching() {
        int result = 0;
        for (int i = 0; i < n; i++) {
            if (match[i] == -1 && augment(i)) { result++; }
        }
        return result;
    }

    vector<pair<int, int>> getMatching() {
        vector<pair<int, int>> result;
        for (int i = 0; i < n; i++) {
            if (match[i] != -1 && i < match[i]) { result.emplace_back(i, match[i]); }
        }
        return result;
    }
};

// 最小点覆盖 - König定理
// 最小点覆盖 = 最大匹配
struct MinimumVertexCover {
    Hungarian hungarian;
    vector<vector<int>> g;
    vector<bool> visitedL, visitedR;
    int n, m;

    MinimumVertexCover(int n, int m) : n(n), m(m), hungarian(n, m), g(n), visitedL(n), visitedR(m) {}

    void addEdge(int u, int v) {
        g[u].push_back(v);
        hungarian.addEdge(u, v);
    }

    void dfs(int u) {
        visitedL[u] = true;
        for (int v : g[u]) {
            if (!visitedR[v] && hungarian.match[v] != u) {
                visitedR[v] = true;
                if (hungarian.match[v] != -1) { dfs(hungarian.match[v]); }
            }
        }
    }

    vector<int> getMinVertexCover() {
        int maxMatch = hungarian.maxMatching();

        fill(visitedL.begin(), visitedL.end(), false);
        fill(visitedR.begin(), visitedR.end(), false);

        // 反向DFS找未覆盖点
        for (int i = 0; i < n; i++) {
            bool matched = false;
            for (int j = 0; j < m; j++) {
                if (hungarian.match[j] == i) {
                    matched = true;
                    break;
                }
            }
            if (!matched) { dfs(i); }
        }

        vector<int> cover;
        // 左部未访问点 + 右部已访问点
        for (int i = 0; i < n; i++) {
            if (!visitedL[i]) {
                cover.push_back(i);  // 左部点
            }
        }
        for (int i = 0; i < m; i++) {
            if (visitedR[i]) {
                cover.push_back(n + i);  // 右部点编号加n
            }
        }
        return cover;
    }
};

// 最大独立集
// 最大独立集 = 顶点数 - 最小点覆盖
struct MaximumIndependentSet {
    MinimumVertexCover mvc;
    int n, m;

    MaximumIndependentSet(int n, int m) : n(n), m(m), mvc(n, m) {}

    void addEdge(int u, int v) { mvc.addEdge(u, v); }

    vector<int> getMaxIndependentSet() {
        auto cover = mvc.getMinVertexCover();
        vector<bool> inCover(n + m, false);
        for (int v : cover) { inCover[v] = true; }

        vector<int> independent;
        for (int i = 0; i < n + m; i++) {
            if (!inCover[i]) { independent.push_back(i); }
        }
        return independent;
    }
};
]=]),

-- 02_Graph_Theory\Maximum_Flow\Dinic.h
ps("02_graph_theory_maximum_flow_dinic_h", [=[

// Dinic最大流算法 - 时间复杂度: O(V^2 * E)
// 使用分层图和当前弧优化，适用于大多数最大流问题
struct Dinic {
    struct Edge {
        int to, cap, flow;
    };

    vector<Edge> edges;
    vector<vector<int>> g;
    vector<int> d, ptr;
    int n;

    Dinic(int _n) : n(_n) {
        g.resize(n);
        d.resize(n);
        ptr.resize(n);
    }
    void add_edge(int from, int to, int cap) {
        g[from].push_back(edges.size());
        edges.push_back({to, cap, 0});
        g[to].push_back(edges.size());
        edges.push_back({from, 0, 0});  // 反向边
    }
    bool bfs(int s, int t) {
        fill(d.begin(), d.end(), -1);
        d[s] = 0;
        queue<int> q;
        q.push(s);

        while (!q.empty()) {
            int v = q.front();
            q.pop();

            for (int id : g[v]) {
                if (d[edges[id].to] == -1 && edges[id].flow < edges[id].cap) {
                    d[edges[id].to] = d[v] + 1;
                    q.push(edges[id].to);
                }
            }
        }

        return d[t] != -1;  // 返回是否能到达汇点
    }

    int dfs(int v, int t, int pushed) {
        if (v == t || pushed == 0) { return pushed; }

        for (int& cid = ptr[v]; cid < g[v].size(); cid++) {
            int id = g[v][cid];
            int to = edges[id].to;

            if (d[v] + 1 != d[to] || edges[id].cap <= edges[id].flow) { continue; }

            int tr = dfs(to, t, min(pushed, edges[id].cap - edges[id].flow));
            if (tr > 0) {
                edges[id].flow += tr;
                edges[id ^ 1].flow -= tr;
                return tr;
            }
        }

        return 0;
    }

    int max_flow(int s, int t) {
        int flow = 0;

        while (bfs(s, t)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (int pushed = dfs(s, t, INT_MAX)) { flow += pushed; }
        }

        return flow;
    }

    // 获取最小割
    vector<bool> min_cut(int s) {
        vector<bool> cut(n, false);
        function<void(int)> dfs_cut = [&](int v) {
            cut[v] = true;
            for (int id : g[v]) {
                if (!cut[edges[id].to] && edges[id].flow < edges[id].cap) { dfs_cut(edges[id].to); }
            }
        };
        dfs_cut(s);
        return cut;
    }
};
]=]),

-- 02_Graph_Theory\Maximum_Flow\ISAP.h
ps("02_graph_theory_maximum_flow_isap_h", [=[

// ISAP (Improved Shortest Augmenting Path) 最大流算法
// 时间复杂度: O(V^2 * E)，在某些图上比Dinic更快
struct ISAP {
    struct Edge {
        int to, cap, flow, rev;
    };

    vector<vector<Edge>> graph;
    vector<int> level, iter, gap;
    int n;

    ISAP(int _n) : n(_n) {
        graph.resize(n);
        level.resize(n);
        iter.resize(n);
        gap.resize(n);
    }

    void add_edge(int from, int to, int cap) {
        graph[from].push_back({to, cap, 0, (int)graph[to].size()});
        graph[to].push_back({from, 0, 0, (int)graph[from].size() - 1});
    }

    void bfs(int t) {
        fill(level.begin(), level.end(), -1);
        fill(gap.begin(), gap.end(), 0);

        queue<int> q;
        level[t] = 0;
        gap[0] = 1;
        q.push(t);

        while (!q.empty()) {
            int v = q.front();
            q.pop();

            for (const Edge& e : graph[v]) {
                if (level[e.to] == -1) {
                    level[e.to] = level[v] + 1;
                    gap[level[e.to]]++;
                    q.push(e.to);
                }
            }
        }
    }

    int dfs(int v, int t, int pushed) {
        if (v == t) return pushed;

        int res = 0;
        for (int& i = iter[v]; i < graph[v].size(); i++) {
            Edge& e = graph[v][i];

            if (e.cap > e.flow && level[v] == level[e.to] + 1) {
                int flow = dfs(e.to, t, min(pushed, e.cap - e.flow));
                if (flow > 0) {
                    e.flow += flow;
                    graph[e.to][e.rev].flow -= flow;
                    res += flow;
                    pushed -= flow;
                    if (pushed == 0) break;
                }
            }
        }

        if (res == 0) {
            gap[level[v]]--;
            if (gap[level[v]] == 0) {
                for (int i = 0; i < n; i++) {
                    if (i != v && level[i] > level[v] && level[i] < n) { level[i] = n; }
                }
            }
            level[v] = n;
            for (const Edge& e : graph[v]) {
                if (e.cap > e.flow && level[e.to] + 1 < level[v]) { level[v] = level[e.to] + 1; }
            }
            gap[level[v]]++;
        }

        return res;
    }

    int max_flow(int s, int t) {
        bfs(t);
        if (level[s] == -1) return 0;

        int flow = 0;
        while (level[s] < n) {
            fill(iter.begin(), iter.end(), 0);
            flow += dfs(s, t, INT_MAX);
        }

        return flow;
    }

    // 获取最小割
    vector<bool> min_cut(int s) {
        vector<bool> cut(n, false);
        function<void(int)> dfs_cut = [&](int v) {
            cut[v] = true;
            for (const Edge& e : graph[v]) {
                if (!cut[e.to] && e.flow < e.cap) { dfs_cut(e.to); }
            }
        };
        dfs_cut(s);
        return cut;
    }
};
]=]),

-- 02_Graph_Theory\Maximum_Flow\MCMF_SPFA.h
ps("02_graph_theory_maximum_flow_mcmf_spfa_h", [=[

constexpr int INF = 0x3f3f3f3f;

// 最小费用最大流 (MCMF) 使用SPFA
// 时间复杂度: O(V * E * F)，其中F为最大流量
// 使用Johnson势能函数优化，避免负权边
struct MCMF_SPFA {
    struct Edge {
        int to, cap, cost, flow, rev;
    };

    vector<vector<Edge>> graph;
    vector<int> dist, parent_v, parent_e, h;
    int n;
    const int INF = 1e9;

    MCMF_SPFA(int _n) : n(_n) {
        graph.resize(n);
        dist.resize(n);
        parent_v.resize(n);
        parent_e.resize(n);
        h.resize(n);
    }

    void add_edge(int from, int to, int cap, int cost) {
        graph[from].push_back({to, cap, cost, 0, (int)graph[to].size()});
        graph[to].push_back({from, 0, -cost, 0, (int)graph[from].size() - 1});
    }

    bool spfa(int s, int t) {
        fill(dist.begin(), dist.end(), INF);
        vector<bool> inq(n, false);
        queue<int> q;

        dist[s] = 0;
        q.push(s);
        inq[s] = true;

        while (!q.empty()) {
            int v = q.front();
            q.pop();
            inq[v] = false;

            for (int i = 0; i < graph[v].size(); i++) {
                const Edge& e = graph[v][i];
                if (e.cap > e.flow && dist[e.to] > dist[v] + e.cost) {
                    dist[e.to] = dist[v] + e.cost;
                    parent_v[e.to] = v;
                    parent_e[e.to] = i;
                    if (!inq[e.to]) {
                        q.push(e.to);
                        inq[e.to] = true;
                    }
                }
            }
        }

        return dist[t] != INF;
    }

    bool dijkstra(int s, int t) {
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

        dist[s] = 0;
        pq.push({0, s});

        while (!pq.empty()) {
            auto [d, v] = pq.top();
            pq.pop();

            if (d > dist[v]) continue;

            for (int i = 0; i < graph[v].size(); i++) {
                const Edge& e = graph[v][i];
                int cost = e.cost + h[v] - h[e.to];
                if (e.cap > e.flow && dist[e.to] > dist[v] + cost) {
                    dist[e.to] = dist[v] + cost;
                    parent_v[e.to] = v;
                    parent_e[e.to] = i;
                    pq.push({dist[e.to], e.to});
                }
            }
        }
        return dist[t] != INF;
    }

    pair<int, int> min_cost_max_flow(int s, int t, int max_flow_limit = 1000000000) {
        int flow = 0, cost = 0;

        // 使用SPFA计算初始势能
        fill(h.begin(), h.end(), 0);

        while (flow < max_flow_limit) {
            bool found;
            if (flow == 0) {
                found = spfa(s, t);
            } else {
                // 更新势能
                for (int v = 0; v < n; v++) {
                    if (dist[v] != INF) { h[v] += dist[v]; }
                }
                found = dijkstra(s, t);
            }
            if (!found) break;

            // 找到增广路径上的最小容量
            int path_flow = max_flow_limit - flow;
            int v = t;
            while (v != s) {
                const Edge& e = graph[parent_v[v]][parent_e[v]];
                path_flow = min(path_flow, e.cap - e.flow);
                v = parent_v[v];
            }

            // 更新流量
            v = t;
            while (v != s) {
                Edge& e = graph[parent_v[v]][parent_e[v]];
                e.flow += path_flow;
                graph[v][e.rev].flow -= path_flow;
                v = parent_v[v];
            }

            flow += path_flow;
            cost += path_flow * (h[t] - h[s]);
        }

        return {flow, cost};
    }

    // 只求最小费用最大流
    pair<int, int> solve(int s, int t) { return min_cost_max_flow(s, t); }
};
]=]),

-- 02_Graph_Theory\Minimum_Spanning_Tree\Boruvka.h
ps("02_graph_theory_minimum_spanning_tree_boruvka_h", [=[
// Borůvka最小生成树算法的独立实现 - 时间复杂度: O(E log V)
// 并行友好的算法，每轮同时为所有连通分量选择最小出边
struct BoruvkaMST {
    struct Edge {
        int u, v, weight;
        int id;  // 边的编号

        bool operator<(const Edge& other) const { return weight < other.weight; }
    };

    struct UnionFind {
        vector<int> parent, rank;
        int components;

        UnionFind(int n) : parent(n), rank(n, 0), components(n) { iota(parent.begin(), parent.end(), 0); }

        int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }

        bool unite(int x, int y) {
            x = find(x), y = find(y);
            if (x == y) return false;

            if (rank[x] < rank[y]) swap(x, y);
            parent[y] = x;
            if (rank[x] == rank[y]) rank[x]++;
            components--;
            return true;
        }

        bool connected(int x, int y) { return find(x) == find(y); }
    };

    vector<Edge> edges;
    int n;

    BoruvkaMST(int _n) : n(_n) {}

    void add_edge(int u, int v, int w) { edges.push_back({u, v, w, (int)edges.size()}); }

    // 返回最小生成树的权值和选择的边
    pair<long long, vector<Edge>> solve() {
        UnionFind uf(n);
        vector<Edge> mst_edges;
        long long total_weight = 0;

        while (uf.components > 1) {
            vector<int> min_edge(n, -1);

            // 为每个连通分量找最小出边
            for (int i = 0; i < edges.size(); i++) {
                int u = uf.find(edges[i].u);
                int v = uf.find(edges[i].v);

                if (u != v) {
                    // 更新u所在分量的最小出边
                    if (min_edge[u] == -1 || edges[i].weight < edges[min_edge[u]].weight) { min_edge[u] = i; }
                    // 更新v所在分量的最小出边
                    if (min_edge[v] == -1 || edges[i].weight < edges[min_edge[v]].weight) { min_edge[v] = i; }
                }
            }

            // 添加找到的最小出边
            vector<bool> added(edges.size(), false);
            for (int i = 0; i < n; i++) {
                if (min_edge[i] != -1 && !added[min_edge[i]]) {
                    const Edge& e = edges[min_edge[i]];
                    if (uf.unite(e.u, e.v)) {
                        mst_edges.push_back(e);
                        total_weight += e.weight;
                        added[min_edge[i]] = true;
                    }
                }
            }
        }

        return {total_weight, mst_edges};
    }

    // 只返回最小生成树的权值
    long long get_weight() { return solve().first; }

    // 检查图是否连通
    bool is_connected() {
        UnionFind uf(n);
        for (const Edge& e : edges) { uf.unite(e.u, e.v); }
        return uf.components == 1;
    }

    // 获取最小生成树的边数
    int get_mst_edge_count() { return solve().second.size(); }
};
]=]),

-- 02_Graph_Theory\Minimum_Spanning_Tree\Kruskal.h
ps("02_graph_theory_minimum_spanning_tree_kruskal_h", [=[
// Kruskal最小生成树算法 - 时间复杂度: O(E log E)
// 适用于稀疏图，基于边的贪心策略
struct Kruskal {
    struct Edge {
        int u, v, w;
        bool operator<(const Edge& other) const { return w < other.w; }
    };

    struct DSU {
        vector<int> fa, rank;

        DSU(int n) : fa(n), rank(n, 0) { iota(fa.begin(), fa.end(), 0); }

        int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }

        bool unite(int x, int y) {
            x = find(x), y = find(y);
            if (x == y) return false;
            if (rank[x] < rank[y]) swap(x, y);
            fa[y] = x;
            if (rank[x] == rank[y]) rank[x]++;
            return true;
        }
    };

    vector<Edge> edges;
    int n;

    Kruskal(int _n) : n(_n) {}

    void add_edge(int u, int v, int w) { edges.push_back({u, v, w}); }

    // 返回最小生成树的权值和，选择的边存储在result中
    long long solve(vector<Edge>& result) {
        sort(edges.begin(), edges.end());
        DSU dsu(n);
        long long total_weight = 0;
        result.clear();

        for (const Edge& e : edges) {
            if (dsu.unite(e.u, e.v)) {
                total_weight += e.w;
                result.push_back(e);
                if (result.size() == n - 1) break;
            }
        }

        return result.size() == n - 1 ? total_weight : -1;  // -1表示图不连通
    }

    // 只返回最小生成树的权值和
    long long solve() {
        vector<Edge> result;
        return solve(result);
    }
};

// Prim最小生成树算法 - 时间复杂度: O(E log V)
// 适用于稠密图，基于点的贪心策略
struct Prim {
    struct Edge {
        int to, w;
    };

    vector<vector<Edge>> graph;
    int n;

    Prim(int _n) : n(_n), graph(_n) {}

    void add_edge(int u, int v, int w) {
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
    }

    // 返回最小生成树的权值和
    long long solve(int start = 0) {
        vector<bool> in_mst(n, false);
        vector<int> min_cost(n, INT_MAX);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

        min_cost[start] = 0;
        pq.push({0, start});
        long long total_weight = 0;
        int edges_added = 0;

        while (!pq.empty() && edges_added < n) {
            auto [cost, u] = pq.top();
            pq.pop();

            if (in_mst[u]) continue;

            in_mst[u] = true;
            total_weight += cost;
            edges_added++;

            for (const Edge& e : graph[u]) {
                if (!in_mst[e.to] && e.w < min_cost[e.to]) {
                    min_cost[e.to] = e.w;
                    pq.push({e.w, e.to});
                }
            }
        }

        return edges_added == n ? total_weight : -1;  // -1表示图不连通
    }

    // 返回最小生成树的边
    vector<pair<int, int>> get_mst_edges(int start = 0) {
        vector<bool> in_mst(n, false);
        vector<int> min_cost(n, INT_MAX);
        vector<int> parent(n, -1);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

        min_cost[start] = 0;
        pq.push({0, start});
        vector<pair<int, int>> mst_edges;

        while (!pq.empty()) {
            auto [cost, u] = pq.top();
            pq.pop();

            if (in_mst[u]) continue;

            in_mst[u] = true;
            if (parent[u] != -1) { mst_edges.push_back({parent[u], u}); }

            for (const Edge& e : graph[u]) {
                if (!in_mst[e.to] && e.w < min_cost[e.to]) {
                    min_cost[e.to] = e.w;
                    parent[e.to] = u;
                    pq.push({e.w, e.to});
                }
            }
        }

        return mst_edges;
    }
};

// Borůvka最小生成树算法（适用于稠密图）- 时间复杂度: O(E log V)
// 并行友好的最小生成树算法，每轮同时选择多条边
struct Boruvka {
    struct Edge {
        int u, v, w;
    };

    struct DSU {
        vector<int> fa, rank;

        DSU(int n) : fa(n), rank(n, 0) { iota(fa.begin(), fa.end(), 0); }

        int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }

        bool unite(int x, int y) {
            x = find(x), y = find(y);
            if (x == y) return false;
            if (rank[x] < rank[y]) swap(x, y);
            fa[y] = x;
            if (rank[x] == rank[y]) rank[x]++;
            return true;
        }
    };

    vector<Edge> edges;
    int n;

    Boruvka(int _n) : n(_n) {}

    void add_edge(int u, int v, int w) { edges.push_back({u, v, w}); }

    long long solve() {
        DSU dsu(n);
        long long total_weight = 0;
        int components = n;

        while (components > 1) {
            vector<int> min_edge(n, -1);

            // 找到每个连通分量的最小出边
            for (int i = 0; i < edges.size(); i++) {
                int u = dsu.find(edges[i].u);
                int v = dsu.find(edges[i].v);

                if (u != v) {
                    if (min_edge[u] == -1 || edges[i].w < edges[min_edge[u]].w) { min_edge[u] = i; }
                    if (min_edge[v] == -1 || edges[i].w < edges[min_edge[v]].w) { min_edge[v] = i; }
                }
            }

            // 添加最小出边
            for (int i = 0; i < n; i++) {
                if (min_edge[i] != -1) {
                    const Edge& e = edges[min_edge[i]];
                    if (dsu.unite(e.u, e.v)) {
                        total_weight += e.w;
                        components--;
                    }
                }
            }
        }

        return components == 1 ? total_weight : -1;
    }
};
]=]),

-- 02_Graph_Theory\Minimum_Spanning_Tree\Prim.h
ps("02_graph_theory_minimum_spanning_tree_prim_h", [=[
// Prim最小生成树算法的独立实现 - 时间复杂度: O(E log V)
// 基于优先队列的实现，适用于稠密图
struct PrimMST {
    struct Edge {
        int to, weight;
        bool operator>(const Edge& other) const { return weight > other.weight; }
    };

    vector<vector<Edge>> adj;
    int n;

    PrimMST(int _n) : n(_n), adj(_n) {}

    void add_edge(int u, int v, int w) {
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // 返回最小生成树的权值和和选择的边
    pair<long long, vector<pair<int, int>>> solve(int start = 0) {
        vector<bool> visited(n, false);
        vector<int> key(n, INT_MAX);
        vector<int> parent(n, -1);
        priority_queue<Edge, vector<Edge>, greater<Edge>> pq;

        key[start] = 0;
        pq.push({start, 0});
        long long total_weight = 0;
        vector<pair<int, int>> mst_edges;

        while (!pq.empty()) {
            int u = pq.top().to;
            pq.pop();

            if (visited[u]) continue;
            visited[u] = true;

            if (parent[u] != -1) {
                mst_edges.push_back({parent[u], u});
                total_weight += key[u];
            }

            for (const Edge& edge : adj[u]) {
                int v = edge.to;
                int weight = edge.weight;

                if (!visited[v] && weight < key[v]) {
                    key[v] = weight;
                    parent[v] = u;
                    pq.push({v, weight});
                }
            }
        }

        // 检查是否所有顶点都被访问（图是否连通）
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                return {-1, {}};  // 图不连通
            }
        }

        return {total_weight, mst_edges};
    }

    // 只返回最小生成树的权值
    long long get_weight(int start = 0) { return solve(start).first; }

    // 检查图是否连通
    bool is_connected() { return get_weight() != -1; }
};
]=]),

-- 02_Graph_Theory\Representation\AdjacencyList.h
ps("02_graph_theory_representation_adjacencylist_h", [=[

// 邻接表 - 图的基本表示方法
// 空间复杂度: O(V + E)
template <typename T = int>
struct AdjacencyList {
    struct Edge {
        int to;
        T weight;
        int id;  // 边的编号

        Edge() : to(0), weight(0), id(0) {}
        Edge(int t, T w = 0, int i = 0) : to(t), weight(w), id(i) {}
    };

    int n, edge_count;
    vector<vector<Edge>> adj;

    AdjacencyList(int size) : n(size), edge_count(0) { adj.resize(n); }

    void add_edge(int from, int to, T weight = 0, bool directed = true) {
        adj[from].emplace_back(to, weight, edge_count);
        if (!directed) { adj[to].emplace_back(from, weight, edge_count); }
        edge_count++;
    }

    void add_weighted_edge(int from, int to, T weight, bool directed = true) { add_edge(from, to, weight, directed); }

    void add_unweighted_edge(int from, int to, bool directed = true) { add_edge(from, to, 1, directed); }

    vector<Edge>& operator[](int u) { return adj[u]; }

    const vector<Edge>& operator[](int u) const { return adj[u]; }

    int size() const { return n; }
    int edges() const { return edge_count; }

    // DFS 遍历
    void dfs(int u, vector<bool>& visited, function<void(int)> process = nullptr) {
        visited[u] = true;
        if (process) process(u);

        for (const Edge& e : adj[u]) {
            if (!visited[e.to]) { dfs(e.to, visited, process); }
        }
    }

    // BFS 遍历
    void bfs(int start, function<void(int)> process = nullptr) {
        vector<bool> visited(n, false);
        queue<int> q;

        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            if (process) process(u);

            for (const Edge& e : adj[u]) {
                if (!visited[e.to]) {
                    visited[e.to] = true;
                    q.push(e.to);
                }
            }
        }
    }

    // 检查连通性
    bool is_connected() {
        vector<bool> visited(n, false);
        dfs(0, visited);

        for (int i = 0; i < n; i++) {
            if (!visited[i]) return false;
        }
        return true;
    }

    // 获取所有边
    vector<tuple<int, int, T>> get_edges() {
        vector<tuple<int, int, T>> edges;
        for (int u = 0; u < n; u++) {
            for (const Edge& e : adj[u]) { edges.emplace_back(u, e.to, e.weight); }
        }
        return edges;
    }

    // 获取度数
    int degree(int u) const { return adj[u].size(); }

    // 清空图
    void clear() {
        for (int i = 0; i < n; i++) { adj[i].clear(); }
        edge_count = 0;
    }

    // 删除边
    void remove_edge(int from, int to) {
        adj[from].erase(remove_if(adj[from].begin(), adj[from].end(), [to](const Edge& e) { return e.to == to; }),
                        adj[from].end());
    }

    // 检查边
    bool has_edge(int from, int to) const {
        for (const Edge& e : adj[from]) {
            if (e.to == to) return true;
        }
        return false;
    }

    // 获取边权重
    T get_weight(int from, int to) const {
        for (const Edge& e : adj[from]) {
            if (e.to == to) return e.weight;
        }
        return T{};  // 默认值
    }

    // 拓扑排序（有向无环图）
    vector<int> topological_sort() {
        vector<int> in_degree(n, 0);
        for (int u = 0; u < n; u++) {
            for (const Edge& e : adj[u]) { in_degree[e.to]++; }
        }

        queue<int> q;
        for (int i = 0; i < n; i++) {
            if (in_degree[i] == 0) { q.push(i); }
        }

        vector<int> result;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            result.push_back(u);

            for (const Edge& e : adj[u]) {
                in_degree[e.to]--;
                if (in_degree[e.to] == 0) { q.push(e.to); }
            }
        }

        return result.size() == n ? result : vector<int>();  // 如果有环返回空
    }

    // 检查有向图是否有环
    bool has_cycle_directed() {
        vector<int> color(n, 0);  // 0: 白色, 1: 灰色, 2: 黑色

        function<bool(int)> dfs_cycle = [&](int u) -> bool {
            color[u] = 1;  // 标记为灰色
            for (const Edge& e : adj[u]) {
                if (color[e.to] == 1 || (color[e.to] == 0 && dfs_cycle(e.to))) { return true; }
            }
            color[u] = 2;  // 标记为黑色
            return false;
        };

        for (int i = 0; i < n; i++) {
            if (color[i] == 0 && dfs_cycle(i)) { return true; }
        }
        return false;
    }
};
]=]),

-- 02_Graph_Theory\Representation\ChainBuild.h
ps("02_graph_theory_representation_chainbuild_h", [=[

// 链式前向星 - 一种高效的图存储结构
// 适用于静态图，空间和时间效率都很高
struct ChainBuild {
    int n, m, cnt = 1;  // n: 顶点数, m: 边数上限, cnt: 当前边编号
    vector<int> head;   // 每个顶点的第一条出边
    vector<int> nxt;    // 下一条边的编号
    vector<int> to;     // 边的终点
    vector<int> val;    // 边的权值

    ChainBuild() {}
    ChainBuild(int n_) : n(n_), m(2 * n_) { init(); }
    ChainBuild(int n_, int m_) : n(n_), m(2 * m_) { init(); }

    void init() {
        head.resize(n + 1, 0);
        nxt.resize(m + 1);
        to.resize(m + 1);
        val.resize(m + 1);
    }

    void clear() {
        cnt = 1;
        fill(head.begin(), head.end(), 0);
    }

    // 添加有向边 u -> v，权重为 w
    void addEdge(int u, int v, int w = 0) {
        nxt[cnt] = head[u];
        to[cnt] = v;
        val[cnt] = w;
        head[u] = cnt++;
    }

    // 添加无向边
    void addUndirectedEdge(int u, int v, int w = 0) {
        addEdge(u, v, w);
        addEdge(v, u, w);
    }

    // 遍历顶点 u 的所有出边
    // 使用方法: for (int i = head[u]; i; i = nxt[i]) { int v = to[i], w = val[i]; }
};
]=]),

-- 02_Graph_Theory\Representation\EdgeList.h
ps("02_graph_theory_representation_edgelist_h", [=[

// 边表法 - 存储所有边的列表形式
// 空间复杂度: O(E)
template <typename T = int>
struct EdgeList {
    struct Edge {
        int from, to;
        T weight;
        int id;

        Edge() : from(0), to(0), weight(0), id(0) {}
        Edge(int f, int t, T w = 0, int i = 0) : from(f), to(t), weight(w), id(i) {}

        bool operator<(const Edge& other) const { return weight < other.weight; }
    };

    vector<Edge> edges;
    int n, edge_count;

    EdgeList(int size) : n(size), edge_count(0) {}

    void add_edge(int from, int to, T weight = 0, bool directed = true) {
        edges.emplace_back(from, to, weight, edge_count++);
        if (!directed) { edges.emplace_back(to, from, weight, edge_count++); }
    }

    Edge& operator[](int idx) { return edges[idx]; }

    const Edge& operator[](int idx) const { return edges[idx]; }

    int size() const { return n; }
    int edge_size() const { return edges.size(); }

    // 按权重排序
    void sort_by_weight() { sort(edges.begin(), edges.end()); }

    // Kruskal算法求最小生成树
    T kruskal_mst(vector<Edge>& mst_edges) {
        sort_by_weight();

        // 并查集
        vector<int> parent(n);
        iota(parent.begin(), parent.end(), 0);

        function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };

        T total_weight = 0;
        mst_edges.clear();

        for (const Edge& e : edges) {
            int u = find(e.from);
            int v = find(e.to);
            if (u != v) {
                parent[u] = v;
                mst_edges.push_back(e);
                total_weight += e.weight;
                if (mst_edges.size() == n - 1) break;
            }
        }

        return mst_edges.size() == n - 1 ? total_weight : T(-1);
    }

    // 只返回最小生成树权重
    T kruskal_weight() {
        vector<Edge> mst_edges;
        return kruskal_mst(mst_edges);
    }

    // 获取所有边
    const vector<Edge>& get_edges() const { return edges; }

    // 按起点排序
    void sort_by_from() {
        sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.from < b.from || (a.from == b.from && a.to < b.to);
        });
    }

    // 按终点排序
    void sort_by_to() {
        sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.to < b.to || (a.to == b.to && a.from < b.from);
        });
    }

    // 按权重范围过滤边
    vector<Edge> filter_by_weight(T min_weight, T max_weight) {
        vector<Edge> filtered;
        for (const Edge& e : edges) {
            if (e.weight >= min_weight && e.weight <= max_weight) { filtered.push_back(e); }
        }
        return filtered;
    }

    // 获取关联某点的边
    vector<Edge> get_incident_edges(int vertex) {
        vector<Edge> incident;
        for (const Edge& e : edges) {
            if (e.from == vertex || e.to == vertex) { incident.push_back(e); }
        }
        return incident;
    }

    // 删除边
    void remove_edge(int idx) {
        if (idx >= 0 && idx < edges.size()) { edges.erase(edges.begin() + idx); }
    }

    // 删除指定权重的边
    void remove_edges_by_weight(T weight) {
        edges.erase(remove_if(edges.begin(), edges.end(), [weight](const Edge& e) { return e.weight == weight; }),
                    edges.end());
    }

    // 清空边表
    void clear() {
        edges.clear();
        edge_count = 0;
    }

    // 检查是否有重边
    bool has_multiple_edges() {
        set<pair<int, int>> edge_set;
        for (const Edge& e : edges) {
            pair<int, int> edge_pair = {min(e.from, e.to), max(e.from, e.to)};
            if (edge_set.count(edge_pair)) { return true; }
            edge_set.insert(edge_pair);
        }
        return false;
    }

    // 去除重边（保留权重最小的）
    void remove_duplicate_edges() {
        map<pair<int, int>, Edge> edge_map;
        for (const Edge& e : edges) {
            pair<int, int> key = {min(e.from, e.to), max(e.from, e.to)};
            if (edge_map.find(key) == edge_map.end() || e.weight < edge_map[key].weight) { edge_map[key] = e; }
        }

        edges.clear();
        for (const auto& [key, edge] : edge_map) { edges.push_back(edge); }
    }
};
]=]),

-- 02_Graph_Theory\Shortest_Path\Dijkstra.h
ps("02_graph_theory_shortest_path_dijkstra_h", [=[
// Dijkstra最短路径算法
// 适用于非负权图，时间复杂度O((V+E)logV)
const long long INF = numeric_limits<long long>::max();

// 基础Dijkstra算法
// 使用优先队列优化，适合稀疏图
struct DijkstraBasic {
    vector<vector<pair<int, long long>>> g;
    vector<long long> dist;
    vector<int> pre;
    int n;

    DijkstraBasic(int n) : n(n), g(n), dist(n), pre(n) {}

    void addEdge(int u, int v, long long w) { g[u].emplace_back(v, w); }

    vector<long long> dijkstra(int s) {
        fill(dist.begin(), dist.end(), INF);
        fill(pre.begin(), pre.end(), -1);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;

        dist[s] = 0;
        pq.emplace(0, s);

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[u]) continue;  // 跳过过期状态

            for (auto [v, w] : g[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pre[v] = u;
                    pq.emplace(dist[v], v);
                }
            }
        }
        return dist;
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
};

// K短路径算法
// 求从源点到目标点的前K条最短路径
struct KShortestPath {
    struct Edge {
        int to;
        long long w;
    };

    struct State {
        long long dist;
        int u, cnt;
        bool operator>(const State& other) const { return dist > other.dist; }
    };

    vector<vector<Edge>> g;
    int n;

    KShortestPath(int n) : n(n), g(n) {}

    void addEdge(int u, int v, long long w) { g[u].push_back({v, w}); }

    vector<long long> kShortest(int s, int t, int k) {
        vector<long long> result;
        priority_queue<State, vector<State>, greater<State>> pq;
        vector<int> cnt(n, 0);

        pq.push({0, s, 0});

        while (!pq.empty() && result.size() < k) {
            auto [d, u, _] = pq.top();
            pq.pop();

            cnt[u]++;
            if (u == t) {
                result.push_back(d);
            }
            if (cnt[u] > k) continue;

            for (auto& e : g[u]) {
                pq.push({d + e.w, e.to, 0});
            }
        }

        return result;
    }
};

// 带限制的最短路径
// 允许使用不超过k条特殊边
struct ConstrainedDijkstra {
    struct Edge {
        int to;
        long long w;
        int type;  // 边的类型，0为普通边，1为特殊边
    };

    vector<vector<Edge>> g;
    int n;

    ConstrainedDijkstra(int n) : n(n), g(n) {}

    void addEdge(int u, int v, long long w, int type = 0) { g[u].push_back({v, w, type}); }

    // 最多使用k条特殊边的最短路径
    vector<vector<long long>> dijkstraWithLimit(int s, int maxSpecial) {
        vector<vector<long long>> dist(n, vector<long long>(maxSpecial + 1, INF));
        priority_queue<tuple<long long, int, int>, vector<tuple<long long, int, int>>, greater<>> pq;

        dist[s][0] = 0;
        pq.emplace(0, s, 0);

        while (!pq.empty()) {
            auto [d, u, used] = pq.top();
            pq.pop();

            if (d > dist[u][used]) continue;

            for (auto& e : g[u]) {
                int newUsed = used + (e.type == 1 ? 1 : 0);
                if (newUsed > maxSpecial) continue;

                if (dist[u][used] + e.w < dist[e.to][newUsed]) {
                    dist[e.to][newUsed] = dist[u][used] + e.w;
                    pq.emplace(dist[e.to][newUsed], e.to, newUsed);
                }
            }
        }

        return dist;
    }
};

// 差分约束系统求解
// 将形如 x[v] - x[u] <= w 的约束转化为最短路径问题
struct DifferenceConstraints {
    vector<vector<pair<int, long long>>> g;
    vector<long long> dist;
    int n;

    DifferenceConstraints(int n) : n(n + 1), g(n + 1), dist(n + 1) {}

    // 添加约束 x[v] - x[u] <= w
    void addConstraint(int u, int v, long long w) { g[u].emplace_back(v, w); }

    // 求解差分约束系统
    bool solve() {
        // 添加超级源点，连接到所有点
        for (int i = 0; i < n - 1; i++) {
            g[n - 1].emplace_back(i, 0);
        }

        fill(dist.begin(), dist.end(), INF);
        dist[n - 1] = 0;

        // 使用Bellman-Ford算法检测负环
        for (int i = 0; i < n; i++) {
            bool updated = false;
            for (int u = 0; u < n; u++) {
                if (dist[u] == INF) continue;
                for (auto [v, w] : g[u]) {
                    if (dist[u] + w < dist[v]) {
                        dist[v] = dist[u] + w;
                        updated = true;
                    }
                }
            }
            if (!updated) break;
            if (i == n - 1) return false;  // 存在负环，无解
        }
        return true;
    }

    // 获取一组可行解
    vector<long long> getSolution() { return vector<long long>(dist.begin(), dist.end() - 1); }
};
]=]),

-- 02_Graph_Theory\Shortest_Path\Floyd.h
ps("02_graph_theory_shortest_path_floyd_h", [=[
// Floyd-Warshall全源最短路径算法
// 用于求解任意两点间的最短距离，时间复杂度O(V^3)
const long long INF = numeric_limits<long long>::max() / 2;

// 基础Floyd-Warshall算法
// 可以处理负权边，检测负环
struct FloydWarshall {
    vector<vector<long long>> dist, path;
    int n;

    FloydWarshall(int n) : n(n), dist(n, vector<long long>(n, INF)), path(n, vector<long long>(n, -1)) {
        for (int i = 0; i < n; i++) {
            dist[i][i] = 0;
        }
    }

    void addEdge(int u, int v, long long w) {
        if (w < dist[u][v]) {
            dist[u][v] = w;
            path[u][v] = v;
        }
    }

    void floyd() {
        // 初始化路径
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && dist[i][j] != INF) {
                    path[i][j] = j;
                }
            }
        }

        // Floyd核心算法
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] != INF && dist[k][j] != INF) {
                        if (dist[i][k] + dist[k][j] < dist[i][j]) {
                            dist[i][j] = dist[i][k] + dist[k][j];
                            path[i][j] = path[i][k];
                        }
                    }
                }
            }
        }
    }

    // 获取从i到j的路径
    vector<int> getPath(int i, int j) {
        if (dist[i][j] == INF) return {};

        vector<int> result;
        int cur = i;
        result.push_back(cur);

        while (cur != j) {
            cur = path[cur][j];
            result.push_back(cur);
        }
        return result;
    }

    // 检查负环
    bool hasNegativeCycle() {
        for (int i = 0; i < n; i++) {
            if (dist[i][i] < 0) {
                return true;
            }
        }
        return false;
    }

    // 获取图的直径（最长最短路径）
    long long getDiameter() {
        long long maxDist = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][j] != INF) {
                    maxDist = max(maxDist, dist[i][j]);
                }
            }
        }
        return maxDist;
    }

    // 获取图的半径（到其他所有点最大距离的最小值）
    long long getRadius() {
        long long radius = INF;
        for (int i = 0; i < n; i++) {
            long long maxFromI = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && dist[i][j] != INF) {
                    maxFromI = max(maxFromI, dist[i][j]);
                }
            }
            radius = min(radius, maxFromI);
        }
        return radius;
    }
};

// 传递闭包
// 用于判断任意两点间的可达性
struct TransitiveClosure {
    vector<vector<bool>> reach;
    int n;

    TransitiveClosure(int n) : n(n), reach(n, vector<bool>(n, false)) {
        for (int i = 0; i < n; i++) {
            reach[i][i] = true;
        }
    }

    void addEdge(int u, int v) { reach[u][v] = true; }

    void buildClosure() {
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
                }
            }
        }
    }

    bool isReachable(int u, int v) { return reach[u][v]; }
};

// 最小环算法
// 使用Floyd变种找图中最小权重环
struct MinimumCycle {
    vector<vector<long long>> dist, original;
    int n;

    MinimumCycle(int n) : n(n), dist(n, vector<long long>(n, INF)), original(n, vector<long long>(n, INF)) {}

    void addEdge(int u, int v, long long w) {
        dist[u][v] = min(dist[u][v], w);
        original[u][v] = min(original[u][v], w);
    }

    long long findMinimumCycle() {
        long long minCycle = INF;

        for (int k = 0; k < n; k++) {
            // 在加入点k之前，检查通过k形成的环
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    if (dist[i][j] != INF && original[j][k] != INF && original[k][i] != INF) {
                        minCycle = min(minCycle, dist[i][j] + original[j][k] + original[k][i]);
                    }
                }
            }

            // Floyd更新
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] != INF && dist[k][j] != INF) {
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }

        return minCycle == INF ? -1 : minCycle;
    }
};

// 最短路径计数
// 统计任意两点间最短路径的数量
struct ShortestPathCount {
    vector<vector<long long>> dist, cnt;
    int n;
    const long long MOD = 1e9 + 7;

    ShortestPathCount(int n) : n(n), dist(n, vector<long long>(n, INF)), cnt(n, vector<long long>(n, 0)) {
        for (int i = 0; i < n; i++) {
            dist[i][i] = 0;
            cnt[i][i] = 1;
        }
    }

    void addEdge(int u, int v, long long w) {
        if (w < dist[u][v]) {
            dist[u][v] = w;
            cnt[u][v] = 1;
        } else if (w == dist[u][v]) {
            cnt[u][v] = (cnt[u][v] + 1) % MOD;
        }
    }

    void floyd() {
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] != INF && dist[k][j] != INF) {
                        long long newDist = dist[i][k] + dist[k][j];
                        if (newDist < dist[i][j]) {
                            dist[i][j] = newDist;
                            cnt[i][j] = (cnt[i][k] * cnt[k][j]) % MOD;
                        } else if (newDist == dist[i][j]) {
                            cnt[i][j] = (cnt[i][j] + cnt[i][k] * cnt[k][j]) % MOD;
                        }
                    }
                }
            }
        }
    }

    long long getPathCount(int u, int v) { return cnt[u][v]; }
};
]=]),

-- 02_Graph_Theory\Shortest_Path\Johnson.h
ps("02_graph_theory_shortest_path_johnson_h", [=[
// Johnson算法全源最短路径
// 结合Bellman-Ford和Dijkstra的优点，适用于稀疏图
// 时间复杂度：O(V^2 log V + VE)
const long long INF = numeric_limits<long long>::max() / 2;

struct Johnson {
    struct Edge {
        int to;
        long long w;
    };

    vector<vector<Edge>> g;
    vector<vector<long long>> dist;
    vector<long long> h;  // 势能函数
    int n;

    Johnson(int n) : n(n), g(n + 1), dist(n, vector<long long>(n)), h(n + 1) {}

    void addEdge(int u, int v, long long w) { g[u].push_back({v, w}); }

    bool johnson() {
        // 添加超级源点
        for (int i = 0; i < n; i++) {
            g[n].push_back({i, 0});
        }

        // 使用Bellman-Ford计算势能函数
        fill(h.begin(), h.end(), INF);
        h[n] = 0;

        for (int i = 0; i <= n; i++) {
            bool updated = false;
            for (int u = 0; u <= n; u++) {
                if (h[u] == INF) continue;
                for (auto& e : g[u]) {
                    if (h[u] + e.w < h[e.to]) {
                        h[e.to] = h[u] + e.w;
                        updated = true;
                    }
                }
            }
            if (!updated) break;
            if (i == n) return false;  // 有负环
        }

        // 重新标记边权
        for (int u = 0; u < n; u++) {
            for (auto& e : g[u]) {
                e.w += h[u] - h[e.to];
            }
        }

        // 对每个点运行Dijkstra
        for (int s = 0; s < n; s++) {
            dijkstraFromSource(s);
        }

        // 恢复原始距离
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][j] != INF) {
                    dist[i][j] += h[j] - h[i];
                }
            }
        }

        return true;
    }

    // 获取从u到v的最短距离
    long long getDistance(int u, int v) { return dist[u][v]; }

    // 获取所有点对最短路径矩阵
    vector<vector<long long>> getAllDistances() { return dist; }

    // 检查是否存在负环
    bool hasNegativeCycle() {
        // 在johnson()中已经检查过了
        // 如果johnson()返回false，说明有负环
        return false;
    }

   private:
    void dijkstraFromSource(int s) {
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;

        fill(dist[s].begin(), dist[s].end(), INF);
        dist[s][s] = 0;
        pq.emplace(0, s);

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[s][u]) continue;

            for (auto& e : g[u]) {
                if (dist[s][u] + e.w < dist[s][e.to]) {
                    dist[s][e.to] = dist[s][u] + e.w;
                    pq.emplace(dist[s][e.to], e.to);
                }
            }
        }
    }
};
]=]),

}
