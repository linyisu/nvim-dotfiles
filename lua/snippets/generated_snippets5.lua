-- Auto-generated LuaSnip snippets
local ls = require("luasnip")
local ps = ls.parser.parse_snippet

return {

-- 03_Dynamic_Programming\Classical\LIS.h
ps("03_dynamic_programming_classical_lis_h", [=[

/**
 * 最长递增子序列(LIS)算法模板
 * 功能：基础LIS、快速LIS、最长递减子序列
 * 时间复杂度：O(n^2)或O(nlogn)
 */

// 基础LIS - O(n^2)算法
struct LIS {
    vector<int> arr;
    int n;

    LIS(const vector<int>& a) : arr(a), n(a.size()) {}

    int solve() {
        vector<int> dp(n, 1);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i]) { dp[i] = max(dp[i], dp[j] + 1); }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }

    vector<int> getLIS() {
        vector<int> dp(n, 1), pre(n, -1);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i] && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                    pre[i] = j;
                }
            }
        }

        int pos = max_element(dp.begin(), dp.end()) - dp.begin();
        vector<int> result;
        while (pos != -1) {
            result.push_back(arr[pos]);
            pos = pre[pos];
        }
        reverse(result.begin(), result.end());
        return result;
    }
};

// 优化LIS - O(nlogn)算法
struct LISFast {
    vector<int> arr;
    int n;

    LISFast(const vector<int>& a) : arr(a), n(a.size()) {}

    int solve() {
        vector<int> tail;
        for (int i = 0; i < n; i++) {
            auto it = lower_bound(tail.begin(), tail.end(), arr[i]);
            if (it == tail.end()) {
                tail.push_back(arr[i]);
            } else {
                *it = arr[i];
            }
        }
        return tail.size();
    }
};

// 最长递减子序列
struct LDS {
    vector<int> arr;
    int n;

    LDS(const vector<int>& a) : arr(a), n(a.size()) {}

    int solve() {
        vector<int> tail;
        for (int i = 0; i < n; i++) {
            auto it = lower_bound(tail.begin(), tail.end(), arr[i], greater<int>());
            if (it == tail.end()) {
                tail.push_back(arr[i]);
            } else {
                *it = arr[i];
            }
        }
        return tail.size();
    }
};
]=]),

-- 03_Dynamic_Programming\Classical\MatrixChain.h
ps("03_dynamic_programming_classical_matrixchain_h", [=[

using ll = long long;
const ll INF = 1e18;

// 基础矩阵链乘法 - 最小代价计算
struct MatrixChain {
    ll solve(const vector<int>& dims) {
        int n = dims.size() - 1;
        if (n <= 0) return 0;

        vector<vector<ll>> dp(n, vector<ll>(n, 0));
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                dp[i][j] = INF;
                for (int k = i; k < j; k++) {
                    ll cost = dp[i][k] + dp[k + 1][j] + (ll)dims[i] * dims[k + 1] * dims[j + 1];
                    dp[i][j] = min(dp[i][j], cost);
                }
            }
        }
        return dp[0][n - 1];
    }
};

// 带分割点记录的矩阵链乘法
struct MatrixChainWithSplit {
    vector<vector<int>> split;

    ll solve(const vector<int>& dims) {
        int n = dims.size() - 1;
        if (n <= 0) return 0;

        vector<vector<ll>> dp(n, vector<ll>(n, 0));
        split.assign(n, vector<int>(n, 0));

        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                dp[i][j] = INF;
                for (int k = i; k < j; k++) {
                    ll cost = dp[i][k] + dp[k + 1][j] + (ll)dims[i] * dims[k + 1] * dims[j + 1];
                    if (cost < dp[i][j]) {
                        dp[i][j] = cost;
                        split[i][j] = k;
                    }
                }
            }
        }
        return dp[0][n - 1];
    }

    void printOrder(int i, int j) {
        if (i == j)
            cout << "A" << i;
        else {
            cout << "(";
            printOrder(i, split[i][j]);
            cout << " x ";
            printOrder(split[i][j] + 1, j);
            cout << ")";
        }
    }
};

// Knuth优化版本 - O(n^3)降为实际更快的运行时间
struct MatrixChainKnuth {
    ll solve(const vector<int>& dims) {
        int n = dims.size() - 1;
        if (n <= 0) return 0;

        vector<vector<ll>> dp(n, vector<ll>(n, 0));
        vector<vector<int>> opt(n, vector<int>(n, 0));

        for (int i = 0; i < n; i++) opt[i][i] = i;

        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                dp[i][j] = INF;

                int left = (i + 1 <= j - 1) ? opt[i][j - 1] : i;
                int right = (i + 1 <= j - 1) ? opt[i + 1][j] : j - 1;

                for (int k = left; k <= right; k++) {
                    ll cost = dp[i][k] + dp[k + 1][j] + (ll)dims[i] * dims[k + 1] * dims[j + 1];
                    if (cost < dp[i][j]) {
                        dp[i][j] = cost;
                        opt[i][j] = k;
                    }
                }
            }
        }
        return dp[0][n - 1];
    }
};
]=]),

-- 03_Dynamic_Programming\Digit_DP\DigitDP_Template.h
ps("03_dynamic_programming_digit_dp_digitdp_template_h", [=[

using ll = long long;

// 基础数位DP模板
struct DigitDP {
    string num;
    vector<vector<vector<ll>>> dp;
    int len;

    // dp[pos][tight][state] 表示当前位置pos，是否受限制tight，当前状态state的方案数
    virtual ll dfs(int pos, bool tight, int state) {
        if (pos == len) return isValid(state);

        if (!tight && dp[pos][0][state] != -1) return dp[pos][0][state];

        int limit = tight ? (num[pos] - '0') : 9;
        ll res = 0;

        for (int digit = 0; digit <= limit; digit++) {
            int newState = getNextState(state, digit, pos);
            bool newTight = tight && (digit == limit);
            res += dfs(pos + 1, newTight, newState);
        }

        if (!tight) dp[pos][0][state] = res;
        return res;
    }

    virtual bool isValid(int state) = 0;
    virtual int getNextState(int state, int digit, int pos) = 0;
    virtual int getInitState() = 0;
    virtual int getMaxState() = 0;

    ll solve(ll n) {
        if (n < 0) return 0;
        num = to_string(n);
        len = num.size();
        dp.assign(len, vector<vector<ll>>(2, vector<ll>(getMaxState() + 1, -1)));
        return dfs(0, true, getInitState());
    }

    ll solve(ll l, ll r) { return solve(r) - solve(l - 1); }
    virtual ~DigitDP() = default;
};

// 数字和能被K整除
struct DivisibleByK : DigitDP {
    int k;
    DivisibleByK(int k_) : k(k_) {}
    bool isValid(int state) override { return state == 0; }
    int getNextState(int state, int digit, int pos) override { return (state + digit) % k; }
    int getInitState() override { return 0; }
    int getMaxState() override { return k - 1; }
};

// 不含指定数字
struct WithoutDigit : DigitDP {
    int forbidden;
    WithoutDigit(int d) : forbidden(d) {}
    bool isValid(int state) override { return state == 0; }
    int getNextState(int state, int digit, int pos) override { return (digit == forbidden) ? 1 : state; }
    int getInitState() override { return 0; }
    int getMaxState() override { return 1; }
};

// 数字和等于目标值
struct DigitSum : DigitDP {
    int target;
    DigitSum(int sum) : target(sum) {}
    bool isValid(int state) override { return state == target; }
    int getNextState(int state, int digit, int pos) override { return state + digit; }
    int getInitState() override { return 0; }
    int getMaxState() override { return target; }
};

// 带前导零处理的数位DP
struct DigitDPWithLeadingZero {
    string num;
    vector<vector<vector<vector<ll>>>> dp;  // dp[pos][tight][state][hasNum]
    int len, target;

    DigitDPWithLeadingZero(int t) : target(t) {}

    ll dfs(int pos, bool tight, int state, bool hasNum) {
        if (pos == len) return hasNum && (state == target);

        if (!tight && dp[pos][0][state][hasNum] != -1) return dp[pos][0][state][hasNum];

        ll res = 0;

        // 可以继续不选数字(前导零)
        if (!hasNum) res += dfs(pos + 1, tight && (num[pos] == '0'), state, false);

        // 选择数字
        int start = hasNum ? 0 : 1;
        int limit = tight ? (num[pos] - '0') : 9;

        for (int digit = start; digit <= limit; digit++) {
            int newState = state + digit;
            bool newTight = tight && (digit == limit);
            res += dfs(pos + 1, newTight, newState, true);
        }

        if (!tight) dp[pos][0][state][hasNum] = res;
        return res;
    }

    ll solve(ll n) {
        if (n < 0) return 0;
        num = to_string(n);
        len = num.size();
        dp.assign(len, vector<vector<vector<ll>>>(2, vector<vector<ll>>(target + 1, vector<ll>(2, -1))));
        return dfs(0, true, 0, false);
    }

    ll solve(ll l, ll r) { return solve(r) - solve(l - 1); }
};
]=]),

-- 03_Dynamic_Programming\Optimization\ConvexHullTrick.h
ps("03_dynamic_programming_optimization_convexhulltrick_h", [=[

using ll = long long;

// 凸包优化 - 斜率递减，查询点递增
struct ConvexHullTrick {
    struct Line {
        ll k, b;
        Line(ll k = 0, ll b = 0) : k(k), b(b) {}
        ll eval(ll x) const { return k * x + b; }
    };

    deque<Line> lines;

    bool bad(const Line& l1, const Line& l2, const Line& l3) {
        return (l3.b - l1.b) * (l1.k - l2.k) <= (l2.b - l1.b) * (l1.k - l3.k);
    }

    void add(ll k, ll b) {
        Line line(k, b);
        while (lines.size() >= 2 && bad(lines[lines.size() - 2], lines.back(), line)) { lines.pop_back(); }
        lines.push_back(line);
    }

    ll query(ll x) {
        while (lines.size() >= 2 && lines[0].eval(x) >= lines[1].eval(x)) { lines.pop_front(); }
        return lines.empty() ? 1e18 : lines[0].eval(x);
    }

    void clear() { lines.clear(); }
};

// Li Chao Tree - 动态插入直线
struct LiChaoTree {
    struct Line {
        ll k, b;
        Line(ll k = 0, ll b = 1e18) : k(k), b(b) {}
        ll eval(ll x) const { return k * x + b; }
    };

    vector<Line> tree;
    int n;
    ll xmin, xmax;

    LiChaoTree(ll xmin, ll xmax) : xmin(xmin), xmax(xmax) {
        n = 1;
        while (n < xmax - xmin + 1) n *= 2;
        tree.assign(2 * n, Line());
    }

    void add(ll k, ll b) { add(Line(k, b), xmin, xmax, 1); }

    void add(Line line, ll tl, ll tr, int v) {
        ll tm = (tl + tr) / 2;
        bool left = line.eval(tl) < tree[v].eval(tl);
        bool mid = line.eval(tm) < tree[v].eval(tm);

        if (mid) swap(tree[v], line);
        if (tl == tr) return;

        if (left != mid)
            add(line, tl, tm, 2 * v);
        else
            add(line, tm + 1, tr, 2 * v + 1);
    }

    ll query(ll x) { return query(x, xmin, xmax, 1); }

    ll query(ll x, ll tl, ll tr, int v) {
        if (tl == tr) return tree[v].eval(x);
        ll tm = (tl + tr) / 2;
        ll res = tree[v].eval(x);
        if (x <= tm)
            res = min(res, query(x, tl, tm, 2 * v));
        else
            res = min(res, query(x, tm + 1, tr, 2 * v + 1));
        return res;
    }
};
]=]),

-- 03_Dynamic_Programming\Optimization\DivideConquer.h
ps("03_dynamic_programming_optimization_divideconquer_h", [=[

using ll = long long;

// 分治优化DP - 决策单调性
struct DivideConquerDP {
    vector<vector<ll>> dp;
    function<ll(int, int)> cost;

    DivideConquerDP(function<ll(int, int)> f) : cost(f) {}

    void compute(int layer, int l, int r, int optl, int optr) {
        if (l > r) return;

        int mid = (l + r) / 2;
        int bestk = optl;

        for (int k = optl; k <= min(mid - 1, optr); k++) {
            if (dp[layer - 1][k] != 1e18) {
                ll val = dp[layer - 1][k] + cost(k + 1, mid);
                if (val < dp[layer][mid]) {
                    dp[layer][mid] = val;
                    bestk = k;
                }
            }
        }

        compute(layer, l, mid - 1, optl, bestk);
        compute(layer, mid + 1, r, bestk, optr);
    }

    vector<ll> solve(int n, int k, const vector<ll>& base) {
        dp.assign(k + 1, vector<ll>(n + 1, 1e18));

        for (int i = 1; i <= n; i++) dp[1][i] = base[i - 1];

        for (int layer = 2; layer <= k; layer++) {
            fill(dp[layer].begin(), dp[layer].end(), 1e18);
            compute(layer, layer, n, layer - 1, n - 1);
        }

        return dp[k];
    }
};

// Knuth-Yao优化 - 四边形不等式
struct KnuthYaoDP {
    vector<vector<ll>> dp;
    vector<vector<int>> opt;
    function<ll(int, int)> cost;

    KnuthYaoDP(function<ll(int, int)> f) : cost(f) {}

    vector<vector<ll>> solve(int n) {
        dp.assign(n, vector<ll>(n, 1e18));
        opt.assign(n, vector<int>(n, 0));

        for (int i = 0; i < n; i++) {
            dp[i][i] = 0;
            opt[i][i] = i;
        }

        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                int left = (i + 1 <= j - 1) ? opt[i][j - 1] : i;
                int right = (i + 1 <= j - 1) ? opt[i + 1][j] : j - 1;

                for (int k = left; k <= right; k++) {
                    ll val = dp[i][k] + dp[k + 1][j] + cost(i, j);
                    if (val < dp[i][j]) {
                        dp[i][j] = val;
                        opt[i][j] = k;
                    }
                }
            }
        }

        return dp;
    }
};

// 单调队列优化
struct MonotonicQueue {
    deque<pair<ll, int>> dq;

    void push(ll val, int idx) {
        while (!dq.empty() && dq.back().first >= val) dq.pop_back();
        dq.push_back({val, idx});
    }

    void popWhile(function<bool(int)> shouldRemove) {
        while (!dq.empty() && shouldRemove(dq.front().second)) dq.pop_front();
    }

    ll getMin() { return dq.empty() ? (ll)1e18 : dq.front().first; }
    pair<ll, int> getMinWithIdx() { return dq.empty() ? make_pair((ll)1e18, -1) : dq.front(); }

    void clear() { dq.clear(); }
};
]=]),

-- 03_Dynamic_Programming\Tree_DP\RerootingDP.h
ps("03_dynamic_programming_tree_dp_rerootingdp_h", [=[

using ll = long long;

// 换根DP通用模板
template <typename T>
struct RerootingDP {
    vector<vector<int>> adj;
    vector<T> down, up, ans;
    int n;

    RerootingDP(int size) : n(size) {
        adj.resize(n);
        down.resize(n);
        up.resize(n);
        ans.resize(n);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    virtual T merge(const T& a, const T& b) = 0;
    virtual T addRoot(const T& subtree, int u) = 0;
    virtual T identity() = 0;

    T dfs1(int u, int parent) {
        down[u] = identity();
        for (int v : adj[u]) {
            if (v != parent) { down[u] = merge(down[u], addRoot(dfs1(v, u), v)); }
        }
        return down[u];
    }

    void dfs2(int u, int parent, const T& fromParent) {
        vector<T> prefix(adj[u].size() + 1, identity());
        vector<T> suffix(adj[u].size() + 1, identity());

        for (int i = 0; i < adj[u].size(); i++) {
            int v = adj[u][i];
            if (v != parent) {
                prefix[i + 1] = merge(prefix[i], addRoot(down[v], v));
            } else {
                prefix[i + 1] = merge(prefix[i], fromParent);
            }
        }

        for (int i = adj[u].size() - 1; i >= 0; i--) {
            int v = adj[u][i];
            if (v != parent) {
                suffix[i] = merge(suffix[i + 1], addRoot(down[v], v));
            } else {
                suffix[i] = merge(suffix[i + 1], fromParent);
            }
        }

        ans[u] = prefix[adj[u].size()];

        for (int i = 0; i < adj[u].size(); i++) {
            int v = adj[u][i];
            if (v != parent) {
                T upValue = addRoot(merge(prefix[i], suffix[i + 1]), u);
                dfs2(v, u, upValue);
            }
        }
    }

    vector<T> solve(int root = 0) {
        dfs1(root, -1);
        dfs2(root, -1, identity());
        return ans;
    }

    virtual ~RerootingDP() = default;
};

// 子树大小和
struct SubtreeSize : RerootingDP<ll> {
    SubtreeSize(int n) : RerootingDP<ll>(n) {}
    ll merge(const ll& a, const ll& b) override { return a + b; }
    ll addRoot(const ll& subtree, int u) override { return subtree + 1; }
    ll identity() override { return 0; }
};

// 距离和计算
struct DistanceSum : RerootingDP<pair<ll, ll>> {
    DistanceSum(int n) : RerootingDP<pair<ll, ll>>(n) {}
    pair<ll, ll> merge(const pair<ll, ll>& a, const pair<ll, ll>& b) override {
        return {a.first + b.first, a.second + b.second};
    }
    pair<ll, ll> addRoot(const pair<ll, ll>& subtree, int u) override {
        return {subtree.first + subtree.second, subtree.second + 1};
    }
    pair<ll, ll> identity() override { return {0, 0}; }
};

// 最大深度
struct MaxDepth : RerootingDP<ll> {
    MaxDepth(int n) : RerootingDP<ll>(n) {}
    ll merge(const ll& a, const ll& b) override { return max(a, b); }
    ll addRoot(const ll& subtree, int u) override { return subtree + 1; }
    ll identity() override { return 0; }
};
]=]),

-- 03_Dynamic_Programming\Tree_DP\TreeCutting.h
ps("03_dynamic_programming_tree_dp_treecutting_h", [=[

using ll = long long;

// 基础树DP模板
struct TreeDP {
    vector<vector<int>> adj;
    int n;

    TreeDP(int size) : n(size) { adj.resize(n); }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
};

// 树的直径
struct TreeDiameter : TreeDP {
    ll diameter;

    TreeDiameter(int n) : TreeDP(n), diameter(0) {}

    ll dfs(int u, int parent) {
        vector<ll> child;
        for (int v : adj[u]) {
            if (v != parent) { child.push_back(dfs(v, u) + 1); }
        }

        sort(child.rbegin(), child.rend());
        ll path = 0;
        if (child.size() >= 1) path += child[0];
        if (child.size() >= 2) path += child[1];
        diameter = max(diameter, path);

        return child.empty() ? 0 : child[0];
    }

    ll solve() {
        diameter = 0;
        dfs(0, -1);
        return diameter;
    }
};

// 树的重心
struct TreeCentroid : TreeDP {
    vector<int> size;
    int centroid;

    TreeCentroid(int n) : TreeDP(n), centroid(-1) { size.resize(n); }

    int dfs(int u, int parent) {
        size[u] = 1;
        int maxSub = 0;

        for (int v : adj[u]) {
            if (v != parent) {
                int sub = dfs(v, u);
                size[u] += sub;
                maxSub = max(maxSub, sub);
            }
        }

        maxSub = max(maxSub, n - size[u]);

        if (centroid == -1 || maxSub < size[centroid]) { centroid = u; }

        return size[u];
    }

    int solve() {
        centroid = -1;
        dfs(0, -1);
        return centroid;
    }
};

// 最大独立集
struct MaxIndependentSet : TreeDP {
    vector<ll> dp0, dp1;  // dp0[u]: 不选u, dp1[u]: 选u

    MaxIndependentSet(int n) : TreeDP(n) {
        dp0.resize(n);
        dp1.resize(n);
    }

    void dfs(int u, int parent) {
        dp0[u] = 0;
        dp1[u] = 1;

        for (int v : adj[u]) {
            if (v != parent) {
                dfs(v, u);
                dp0[u] += max(dp0[v], dp1[v]);
                dp1[u] += dp0[v];
            }
        }
    }

    ll solve() {
        dfs(0, -1);
        return max(dp0[0], dp1[0]);
    }
};

// 子树大小
struct SubtreeSize : TreeDP {
    vector<int> size;

    SubtreeSize(int n) : TreeDP(n) { size.resize(n); }

    int dfs(int u, int parent) {
        size[u] = 1;
        for (int v : adj[u]) {
            if (v != parent) { size[u] += dfs(v, u); }
        }
        return size[u];
    }

    vector<int> solve() {
        dfs(0, -1);
        return size;
    }
};
]=]),

-- 03_Dynamic_Programming\Tree_DP\TreeDP_Basic.h
ps("03_dynamic_programming_tree_dp_treedp_basic_h", [=[

using ll = long long;

// 树DP通用模板
template <typename T>
struct TreeDPTemplate {
    vector<vector<int>> adj;
    vector<T> dp;
    int n;

    TreeDPTemplate(int size) : n(size) {
        adj.resize(n);
        dp.resize(n);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    virtual T dfs(int u, int parent) = 0;
    virtual T getIdentity() = 0;

    T solve(int root = 0) {
        fill(dp.begin(), dp.end(), getIdentity());
        return dfs(root, -1);
    }

    virtual ~TreeDPTemplate() = default;
};

// 树上路径数
struct TreePaths : TreeDPTemplate<ll> {
    ll totalPaths;

    TreePaths(int n) : TreeDPTemplate<ll>(n), totalPaths(0) {}

    ll getIdentity() override { return 0; }

    ll dfs(int u, int parent) override {
        dp[u] = 1;
        for (int v : adj[u]) {
            if (v != parent) {
                ll child = dfs(v, u);
                totalPaths += dp[u] * child;
                dp[u] += child;
            }
        }
        return dp[u];
    }

    ll getPaths() {
        totalPaths = 0;
        solve();
        return totalPaths;
    }
};

// 树上最长路径
struct TreeLongestPath : TreeDPTemplate<ll> {
    ll maxPath;

    TreeLongestPath(int n) : TreeDPTemplate<ll>(n), maxPath(0) {}

    ll getIdentity() override { return 0; }

    ll dfs(int u, int parent) override {
        vector<ll> children;
        for (int v : adj[u]) {
            if (v != parent) {
                ll child = dfs(v, u);
                children.push_back(child);
            }
        }

        sort(children.rbegin(), children.rend());
        ll path = 0;
        if (children.size() >= 1) path += children[0];
        if (children.size() >= 2) path += children[1];
        maxPath = max(maxPath, path);

        dp[u] = children.empty() ? 0 : children[0] + 1;
        return dp[u];
    }

    ll getMaxPath() {
        maxPath = 0;
        solve();
        return maxPath;
    }
};

// 简化版树DP
struct SimpleTreeDP {
    vector<vector<int>> adj;
    int n;

    SimpleTreeDP(int size) : n(size) { adj.resize(n); }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // 计算子树大小
    vector<int> getSubtreeSizes() {
        vector<int> size(n);
        function<int(int, int)> dfs = [&](int u, int p) {
            size[u] = 1;
            for (int v : adj[u]) {
                if (v != p) size[u] += dfs(v, u);
            }
            return size[u];
        };
        dfs(0, -1);
        return size;
    }

    // 计算树的直径
    ll getDiameter() {
        ll ans = 0;
        function<ll(int, int)> dfs = [&](int u, int p) -> ll {
            vector<ll> depths;
            for (int v : adj[u]) {
                if (v != p) depths.push_back(dfs(v, u) + 1);
            }
            sort(depths.rbegin(), depths.rend());
            ll path = 0;
            if (depths.size() >= 1) path += depths[0];
            if (depths.size() >= 2) path += depths[1];
            ans = max(ans, path);
            return depths.empty() ? 0 : depths[0];
        };
        dfs(0, -1);
        return ans;
    }

    // 最大独立集
    ll getMaxIndependentSet() {
        vector<ll> dp0(n), dp1(n);  // dp0: 不选, dp1: 选
        function<void(int, int)> dfs = [&](int u, int p) {
            dp0[u] = 0;
            dp1[u] = 1;
            for (int v : adj[u]) {
                if (v != p) {
                    dfs(v, u);
                    dp0[u] += max(dp0[v], dp1[v]);
                    dp1[u] += dp0[v];
                }
            }
        };
        dfs(0, -1);
        return max(dp0[0], dp1[0]);
    }
};
]=]),

-- 04_Math\Combinatorics\Advanced\Bell.h
ps("04_math_combinatorics_advanced_bell_h", [=[

/**
 * 贝尔数模板
 * 功能：贝尔数计算、贝尔三角形
 * 时间复杂度：O(n^2)
 */

using ll = long long;

// 贝尔数（集合分割数）
struct Bell {
    vector<ll> B;
    vector<vector<ll>> triangle;
    int n;
    ll mod;

    Bell(int n, ll mod = 1e9 + 7) : n(n), mod(mod) {
        B.resize(n + 1);
        triangle.resize(n + 1);
        for (int i = 0; i <= n; i++) { triangle[i].resize(i + 2); }
        init();
    }

    void init() {
        // 贝尔三角形计算贝尔数
        triangle[0][0] = B[0] = 1;

        for (int i = 1; i <= n; i++) {
            triangle[i][0] = triangle[i - 1][i - 1];

            for (int j = 1; j <= i; j++) { triangle[i][j] = (triangle[i - 1][j - 1] + triangle[i][j - 1]) % mod; }

            B[i] = triangle[i][0];
        }
    }

    ll get(int n) {
        if (n < 0 || n > this->n) return 0;
        return B[n];
    }
};

// 使用第二类斯特林数计算贝尔数
struct BellByStirling {
    vector<ll> B;
    int n;
    ll mod;

    BellByStirling(int n, ll mod = 1e9 + 7) : n(n), mod(mod), B(n + 1, 0) { init(); }

    void init() {
        // B(n) = sum_{k=0}^{n} S(n,k)
        vector<vector<ll>> S(n + 1, vector<ll>(n + 1, 0));
        S[0][0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) { S[i][j] = (S[i - 1][j - 1] + (ll)j * S[i - 1][j]) % mod; }
        }

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= i; j++) { B[i] = (B[i] + S[i][j]) % mod; }
        }
    }

    ll get(int n) {
        if (n < 0 || n > this->n) return 0;
        return B[n];
    }
};

// 扩展贝尔数模板类
template <typename T = long long>
class BellNumbers {
   private:
    static const int maxn = 1000;
    static const T MOD = 1e9 + 7;

    vector<T> bell;
    vector<vector<T>> triangle;

    // 快速幂
    T power(T base, T exp) {
        T result = 1;
        while (exp > 0) {
            if (exp & 1) result = (result * base) % MOD;
            base = (base * base) % MOD;
            exp >>= 1;
        }
        return result;
    }

   public:
    BellNumbers() {
        bell.resize(maxn + 1);
        triangle.resize(maxn + 1);
        for (int i = 0; i <= maxn; i++) { triangle[i].resize(i + 2); }
        init();
    }

    void init() {
        // 初始化贝尔三角形
        triangle[0][0] = bell[0] = 1;

        for (int i = 1; i <= maxn; i++) {
            triangle[i][0] = triangle[i - 1][i - 1];

            for (int j = 1; j <= i; j++) { triangle[i][j] = (triangle[i - 1][j - 1] + triangle[i][j - 1]) % MOD; }

            bell[i] = triangle[i][0];
        }
    }

    // 获取第n个贝尔数
    T get_bell(int n) {
        if (n <= maxn) return bell[n];
        return calculate_bell_large(n);
    }

    // 使用第二类斯特林数计算贝尔数
    T bell_from_stirling2(int n) {
        T res = 0;
        vector<T> stirling2(n + 1, 0);
        stirling2[0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = min(i, n); j >= 1; j--) { stirling2[j] = (stirling2[j - 1] + (T)j * stirling2[j]) % MOD; }
        }

        for (int k = 0; k <= n; k++) { res = (res + stirling2[k]) % MOD; }

        return res;
    }

    // 使用指数生成函数计算大贝尔数
    T calculate_bell_large(int n) { return bell_from_stirling2(n); }

    // 使用Dobinski公式计算贝尔数（仅理论用途）
    double dobinski_formula(int n) {
        const double E = 2.718281828459045;
        double sum = 0.0;
        double factorial = 1.0;

        for (int k = 0; k <= 100; k++) {
            if (k > 0) factorial *= k;
            double term = pow(k, n) / factorial;
            sum += term;
            if (term < 1e-15) break;
        }

        return sum / E;
    }

    // 贝尔数的递推关系
    T bell_recurrence(int n) {
        if (n == 0) return 1;

        vector<T> dp(n + 1);
        dp[0] = 1;

        for (int i = 1; i <= n; i++) {
            dp[i] = 0;
            T binom = 1;

            for (int k = 0; k < i; k++) {
                dp[i] = (dp[i] + binom * dp[k]) % MOD;

                if (k + 1 < i) {
                    binom = binom * (i - 1 - k) % MOD;
                    binom = binom * power(k + 1, MOD - 2) % MOD;
                }
            }
        }

        return dp[n];
    }

    // 计算贝尔数的前n项和
    T bell_sum(int n) {
        T sum = 0;
        for (int i = 0; i <= n; i++) { sum = (sum + get_bell(i)) % MOD; }
        return sum;
    }

    // 获取贝尔三角形的某一行
    vector<T> get_bell_triangle_row(int row) {
        if (row <= maxn) { return vector<T>(triangle[row].begin(), triangle[row].begin() + row + 1); }

        vector<T> prev_row = get_bell_triangle_row(row - 1);
        vector<T> curr_row(row + 1);

        curr_row[0] = prev_row.back();
        for (int j = 1; j <= row; j++) { curr_row[j] = (prev_row[j - 1] + curr_row[j - 1]) % MOD; }

        return curr_row;
    }
};

// 使用示例
/*
BellNumbers<long long> bell;
cout << bell.get_bell(5) << endl; // 输出52
cout << bell.bell_sum(5) << endl; // 输出前6个贝尔数的和

vector<long long> row = bell.get_bell_triangle_row(4);
for (auto x : row) {
    cout << x << " ";
}
*/
]=]),

-- 04_Math\Combinatorics\Advanced\Catalan.h
ps("04_math_combinatorics_advanced_catalan_h", [=[

/**
 * 卡特兰数模板
 * 功能：卡特兰数计算、应用实例
 * 时间复杂度：O(n)预处理，O(1)查询
 */

using ll = long long;

template <typename T = ll>
struct CatalanNumbers {
    static const int maxn = 1000005;
    static const T MOD = 1e9 + 7;

    vector<T> catalan;
    vector<T> fact, inv_fact;
    int n;
    T mod;

    CatalanNumbers(int n = maxn, T mod = MOD) : n(n), mod(mod) {
        catalan.resize(n + 1);
        fact.resize(2 * n + 1);
        inv_fact.resize(2 * n + 1);
        init();
    }

    T power(T a, T b) {
        T res = 1;
        a %= mod;
        while (b) {
            if (b & 1) res = res * a % mod;
            a = a * a % mod;
            b >>= 1;
        }
        return res;
    }

    void init() {
        // 预处理阶乘
        fact[0] = 1;
        for (int i = 1; i <= 2 * n; i++) { fact[i] = fact[i - 1] * i % mod; }
        inv_fact[2 * n] = power(fact[2 * n], mod - 2);
        for (int i = 2 * n - 1; i >= 0; i--) { inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod; }

        calculate_catalan();
    }

    void calculate_catalan() {
        // 方法1: 递推公式 C(n) = sum_{i=0}^{n-1} C(i) * C(n-1-i)
        catalan[0] = 1;
        for (int i = 1; i <= n; i++) {
            catalan[i] = 0;
            for (int j = 0; j < i; j++) { catalan[i] = (catalan[i] + catalan[j] * catalan[i - 1 - j]) % mod; }
        }
    }

    // 使用组合数公式计算: C(n) = C(2n, n) / (n + 1)
    T catalan_formula(int n) {
        if (n == 0) return 1;
        return fact[2 * n] * inv_fact[n] % mod * inv_fact[n + 1] % mod;
    }

    // 另一个公式: C(n) = C(2n, n) - C(2n, n+1)
    T catalan_formula2(int n) {
        T c1 = fact[2 * n] * inv_fact[n] % mod * inv_fact[n] % mod;
        T c2 = fact[2 * n] * inv_fact[n + 1] % mod * inv_fact[n - 1] % mod;
        return (c1 - c2 + mod) % mod;
    }

    // 获取第n个卡特兰数
    T get_catalan(int n) {
        if (n <= this->n) return catalan[n];
        return catalan_formula(n);
    }

    // 卡特兰数的各种应用

    // 1. n对括号的合法匹配数
    T valid_parentheses(int n) { return get_catalan(n); }

    // 2. n+1个数字构成的二叉搜索树数量
    T binary_search_trees(int n) { return get_catalan(n); }

    // 3. 从(0,0)到(n,n)不越过对角线的路径数
    T lattice_paths(int n) { return get_catalan(n); }

    // 4. n+2边形的三角剖分数
    T polygon_triangulations(int n) { return get_catalan(n); }

    // 5. 长度为2n的Dyck路径数
    T dyck_paths(int n) { return get_catalan(n); }

    // 6. n个节点的满二叉树数量
    T full_binary_trees(int n) {
        if (n % 2 == 0) return 0;  // 满二叉树节点数必须为奇数
        return get_catalan((n - 1) / 2);
    }

    // 7. 山脉数组的数量（n个上升，n个下降）
    T mountain_ranges(int n) { return get_catalan(n); }

    // 扩展：带权重的卡特兰数
    T weighted_catalan(int n, vector<T>& weights) {
        vector<T> dp(n + 1, 0);
        dp[0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) { dp[i] = (dp[i] + dp[j] * dp[i - 1 - j] % mod * weights[j]) % mod; }
        }

        return dp[n];
    }

    // 超级卡特兰数：S(m,n) = ((2m)!(2n)!) / ((m+n)!m!n!)
    T super_catalan(int m, int n) {
        T res = fact[2 * m] * fact[2 * n] % mod;
        res = res * inv_fact[m + n] % mod;
        res = res * inv_fact[m] % mod;
        res = res * inv_fact[n] % mod;
        return res;
    }

    // 生成函数方法计算卡特兰数
    vector<T> catalan_generating_function(int n) {
        vector<T> c(n + 1, 0);
        c[0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) { c[i] = (c[i] + c[j] * c[i - 1 - j]) % mod; }
        }

        return c;
    }
};

// 使用示例
/*
CatalanNumbers<long long> cat(100);
cout << cat.get_catalan(5) << endl; // 输出42
cout << cat.valid_parentheses(3) << endl; // 3对括号的合法匹配数：5
cout << cat.binary_search_trees(4) << endl; // 4个节点的BST数量：14
*/
]=]),

-- 04_Math\Combinatorics\Advanced\Stirling.h
ps("04_math_combinatorics_advanced_stirling_h", [=[

/**
 * 斯特林数模板
 * 功能：第一类斯特林数、第二类斯特林数
 * 时间复杂度：O(n^2)
 */

using ll = long long;

// 第一类斯特林数
struct Stirling1 {
    vector<vector<ll>> s;
    int n;
    ll mod;

    Stirling1(int n, ll mod = 1e9 + 7) : n(n), mod(mod) {
        s.assign(n + 1, vector<ll>(n + 1, 0));
        init();
    }

    void init() {
        s[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) { s[i][j] = (s[i - 1][j - 1] + (ll)(i - 1) * s[i - 1][j]) % mod; }
        }
    }

    ll get(int n, int k) {
        if (n < 0 || k < 0 || k > n) return 0;
        return s[n][k];
    }
};

// 第二类斯特林数
struct Stirling2 {
    vector<vector<ll>> S;
    int n;
    ll mod;

    Stirling2(int n, ll mod = 1e9 + 7) : n(n), mod(mod) {
        S.assign(n + 1, vector<ll>(n + 1, 0));
        init();
    }

    void init() {
        S[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) { S[i][j] = (S[i - 1][j - 1] + (ll)j * S[i - 1][j]) % mod; }
        }
    }

    ll get(int n, int k) {
        if (n < 0 || k < 0 || k > n) return 0;
        return S[n][k];
    }
};

// 贝尔数（第二类斯特林数的行和）
struct Bell {
    vector<ll> B;
    int n;
    ll mod;

    Bell(int n, ll mod = 1e9 + 7) : n(n), mod(mod), B(n + 1, 0) { init(); }

    void init() {
        Stirling2 s2(n, mod);
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= i; j++) { B[i] = (B[i] + s2.get(i, j)) % mod; }
        }
    }

    ll get(int n) {
        if (n < 0 || n > this->n) return 0;
        return B[n];
    }
};

// 斯特林数工具类
template <typename T = long long>
class StirlingNumbers {
   private:
    static const T MOD = 1e9 + 7;
    static const int maxn = 1000;
    vector<vector<T>> stirling1, stirling2;
    int n;

   public:
    StirlingNumbers(int size) : n(size) {
        stirling1.assign(size + 1, vector<T>(size + 1, 0));
        stirling2.assign(size + 1, vector<T>(size + 1, 0));
        init();
    }

    void init() {
        // 初始化第一类斯特林数
        stirling1[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                stirling1[i][j] = (stirling1[i - 1][j - 1] + (T)(i - 1) * stirling1[i - 1][j]) % MOD;
            }
        }

        // 初始化第二类斯特林数
        stirling2[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                stirling2[i][j] = (stirling2[i - 1][j - 1] + (T)j * stirling2[i - 1][j]) % MOD;
            }
        }
    }

    // 获取第一类斯特林数
    T get_stirling1(int n, int k) {
        if (n > this->n || k > this->n || n < 0 || k < 0) return 0;
        return stirling1[n][k];
    }

    // 获取第二类斯特林数
    T get_stirling2(int n, int k) {
        if (n > this->n || k > this->n || n < 0 || k < 0) return 0;
        return stirling2[n][k];
    }

    // 计算贝尔数 B(n) = sum_{k=0}^n S(n,k)
    T bell_number(int n) {
        T res = 0;
        for (int k = 0; k <= n; k++) { res = (res + stirling2[n][k]) % MOD; }
        return res;
    }

    // 快速幂
    T power(T a, T b) {
        T res = 1;
        while (b > 0) {
            if (b & 1) res = res * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return res;
    }

    // 组合数计算
    T combination(T n, T k) {
        if (k > n || k < 0) return 0;
        if (k == 0 || k == n) return 1;

        T res = 1;
        for (T i = 0; i < k; i++) {
            res = res * ((n - i) % MOD) % MOD;
            res = res * power(i + 1, MOD - 2) % MOD;
        }
        return res;
    }

    // 使用容斥原理计算第二类斯特林数的大数版本
    T stirling2_large(T n, T k) {
        if (k > n || k == 0) return 0;
        if (k == 1 || k == n) return 1;

        T res = 0;
        T fact_k = 1;
        for (T i = 1; i <= k; i++) { fact_k = fact_k * i % MOD; }

        T inv_fact_k = power(fact_k, MOD - 2);

        for (T i = 0; i <= k; i++) {
            T term = combination(k, i) * power(k - i, n) % MOD;
            if (i & 1) {
                res = (res - term + MOD) % MOD;
            } else {
                res = (res + term) % MOD;
            }
        }

        return res * inv_fact_k % MOD;
    }

    // 第二类斯特林数的生成函数方法
    vector<T> stirling2_row(int n) {
        vector<T> dp(n + 1, 0);
        dp[0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = min(i, n); j >= 1; j--) { dp[j] = (dp[j - 1] + (T)j * dp[j]) % MOD; }
        }

        return dp;
    }

    // 第一类斯特林数某一行
    vector<T> stirling1_row(int n) {
        vector<T> dp(n + 1, 0);
        dp[0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = min(i, n); j >= 1; j--) { dp[j] = (dp[j - 1] + (T)(i - 1) * dp[j]) % MOD; }
        }

        return dp;
    }
};

// 使用示例
/*
StirlingNumbers<long long> stirling(100);
cout << stirling.get_stirling2(5, 3) << endl; // 输出25
cout << stirling.bell_number(5) << endl; // 输出52
*/
]=]),

-- 04_Math\Combinatorics\Basic\BasicMath.h
ps("04_math_combinatorics_basic_basicmath_h", [=[

/**
 * 基础数学函数模板
 * 功能：整数除法、开方、对数、数学工具函数
 * 时间复杂度：O(1)或O(log n)
 */

using ll = long long;

// 向上取整除法
template <typename T>
constexpr T ceil_div(T n, T m) {
    return (n + m - 1) / m;
}

// 向下取整除法
template <typename T>
constexpr T floor_div(T n, T m) {
    return n / m;
}

// 更新最大值
template <typename T>
void chmax(T &a, T b) {
    if (a < b) a = b;
}

// 更新最小值
template <typename T>
void chmin(T &a, T b) {
    if (a > b) a = b;
}

// 整数开方
ll isqrt(ll n) {
    ll s = sqrtl(n);
    while (s * s > n) s--;
    while ((s + 1) * (s + 1) <= n) s++;
    return s;
}

// 找到最小的u使得1+2+...+u >= n
ll triangular_root(ll n) {
    ll u = isqrt(2 * n);
    while (u * (u + 1) / 2 < n) u++;
    while (u * (u - 1) / 2 >= n) u--;
    return u;
}

// 整数对数：返回log_a(b)的上界
int ilog(int a, ll b) {
    int t = 0;
    ll v = 1;
    while (v < b) {
        v *= a;
        t++;
    }
    return t;
}

// 判断是否为2的幂
bool is_power_of_2(ll x) { return x > 0 && (x & (x - 1)) == 0; }

// 返回小于等于x的最大2的幂
ll max_power_of_2(ll x) { return x <= 0 ? 0 : 1LL << (63 - __builtin_clzll(x)); }

// 预处理log2数组
struct Log2Table {
    vector<int> lg;

    Log2Table(int n = 100000) : lg(n + 1) {
        lg[1] = 0;
        for (int i = 2; i <= n; i++) { lg[i] = lg[i / 2] + 1; }
    }

    int operator[](int x) const { return lg[x]; }
};
]=]),

-- 04_Math\Combinatorics\Basic\Combination.h
ps("04_math_combinatorics_basic_combination_h", [=[

/**
 * 组合数计算模板
 * 功能：杨辉三角、直接计算、Lucas定理
 * 时间复杂度：O(n^2)预处理，O(1)查询或O(log p)
 */

using ll = long long;

// 快速幂
ll power(ll a, ll b, ll mod) {
    ll res = 1;
    a %= mod;
    while (b) {
        if (b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}

// 杨辉三角预处理组合数
struct PascalTriangle {
    vector<vector<ll>> C;
    int n;
    ll mod;

    PascalTriangle(int n, ll mod = 1e9 + 7) : n(n), mod(mod) {
        C.assign(n + 1, vector<ll>(n + 1, 0));
        init();
    }

    void init() {
        for (int i = 0; i <= n; i++) {
            C[i][0] = C[i][i] = 1;
            for (int j = 1; j < i; j++) { C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % mod; }
        }
    }

    ll get(int n, int k) {
        if (n < 0 || k < 0 || k > n) return 0;
        return C[n][k];
    }
};

// 直接计算组合数
ll combination(ll n, ll k, ll mod = 1e9 + 7) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;

    k = min(k, n - k);  // 优化：利用对称性

    ll num = 1, den = 1;
    for (int i = 0; i < k; i++) {
        num = num * ((n - i) % mod) % mod;
        den = den * (i + 1) % mod;
    }

    return num * power(den, mod - 2, mod) % mod;
}

// Lucas 定理计算大数组合数
ll lucas(ll n, ll k, ll p) {
    if (k == 0) return 1;
    return combination(n % p, k % p, p) * lucas(n / p, k / p, p) % p;
}

// 多项式系数
ll multinomial(const vector<int>& k, ll mod = 1e9 + 7) {
    int n = 0;
    for (int x : k) n += x;

    ll result = 1;
    for (int x : k) {
        result = result * combination(n, x, mod) % mod;
        n -= x;
    }
    return result;
}
]=]),

-- 04_Math\Combinatorics\Basic\ExLucas.h
ps("04_math_combinatorics_basic_exlucas_h", [=[

/**
 * 扩展Lucas定理模板
 * 功能：计算C(n,m) mod p，p可以不是质数
 * 时间复杂度：O(p^k * log n)
 */

using ll = long long;

// 扩展Lucas定理
struct ExLucas {
    ll power(ll a, ll b, ll mod) {
        ll res = 1;
        a %= mod;
        while (b) {
            if (b & 1) res = res * a % mod;
            a = a * a % mod;
            b >>= 1;
        }
        return res;
    }

    ll exgcd(ll a, ll b, ll &x, ll &y) {
        if (!b) {
            x = 1, y = 0;
            return a;
        }
        ll d = exgcd(b, a % b, y, x);
        y -= a / b * x;
        return d;
    }

    ll inv(ll a, ll mod) {
        ll x, y;
        ll d = exgcd(a, mod, x, y);
        return d == 1 ? (x % mod + mod) % mod : -1;
    }

    ll calc_fac(ll n, ll p, ll pk) {
        if (!n) return 1;
        ll res = 1;
        for (ll i = 1; i <= pk; i++) {
            if (i % p) res = res * i % pk;
        }
        res = power(res, n / pk, pk);
        for (ll i = 1; i <= n % pk; i++) {
            if (i % p) res = res * i % pk;
        }
        return res * calc_fac(n / p, p, pk) % pk;
    }

    ll calc_power(ll n, ll p) {
        ll cnt = 0;
        while (n) {
            n /= p;
            cnt += n;
        }
        return cnt;
    }

    ll solve(ll n, ll m, ll p, ll pk) {
        ll fac_n = calc_fac(n, p, pk);
        ll fac_m = calc_fac(m, p, pk);
        ll fac_nm = calc_fac(n - m, p, pk);
        ll cnt = calc_power(n, p) - calc_power(m, p) - calc_power(n - m, p);
        ll res = fac_n * inv(fac_m, pk) % pk * inv(fac_nm, pk) % pk;
        return res * power(p, cnt, pk) % pk;
    }

    ll C(ll n, ll m, ll mod) {
        if (n < m || m < 0) return 0;

        vector<pair<ll, ll>> factors;
        ll temp = mod;
        for (ll i = 2; i * i <= temp; i++) {
            if (temp % i == 0) {
                ll pk = 1;
                while (temp % i == 0) {
                    temp /= i;
                    pk *= i;
                }
                factors.push_back({i, pk});
            }
        }
        if (temp > 1) factors.push_back({temp, temp});

        vector<ll> a, m_vec;
        for (auto [p, pk] : factors) {
            a.push_back(solve(n, m, p, pk));
            m_vec.push_back(pk);
        }

        // 中国剩余定理
        ll res = 0, M = 1;
        for (ll mi : m_vec) M *= mi;

        for (int i = 0; i < a.size(); i++) {
            ll Mi = M / m_vec[i];
            res = (res + a[i] * Mi % M * inv(Mi, m_vec[i]) % M) % M;
        }
        return (res + M) % M;
    }
};
T res = 0;
while (n) {
    n /= p;
    res += n;
}
return res;
}

// 计算n! / p^k mod p^alpha
T calc_factorial(T n, T p, T alpha) {
    if (n == 0) return 1;
    T pk = power(p, alpha, LLONG_MAX);
    T res = 1;

    // 计算不包含p的部分
    for (T i = 1; i <= pk; i++) {
        if (i % p != 0) { res = res * i % pk; }
    }
    res = power(res, n / pk, pk);

    // 计算剩余部分
    for (T i = 1; i <= n % pk; i++) {
        if (i % p != 0) { res = res * i % pk; }
    }

    return res * calc_factorial(n / p, p, alpha) % pk;
}

// 计算C(n, m) mod p^alpha
T calc_comb(T n, T m, T p, T alpha) {
    if (n < m || n < 0 || m < 0) return 0;

    T pk = power(p, alpha, LLONG_MAX);
    T cnt = calc_power(n, p) - calc_power(m, p) - calc_power(n - m, p);

    if (cnt >= alpha) return 0;

    T res = calc_factorial(n, p, alpha);
    res = res * inv(calc_factorial(m, p, alpha), pk) % pk;
    res = res * inv(calc_factorial(n - m, p, alpha), pk) % pk;
    res = res * power(p, cnt, pk) % pk;

    return res;
}

// 中国剩余定理合并
T crt(vector<T> &a, vector<T> &m) {
    T res = 0, M = 1;
    for (auto mod : m) M *= mod;

    for (int i = 0; i < a.size(); i++) {
        T Mi = M / m[i];
        T ti = inv(Mi, m[i]);
        res = (res + a[i] * Mi * ti) % M;
    }
    return (res + M) % M;
}

// 扩展Lucas定理主函数
T exlucas(T n, T m, T mod) {
    vector<T> primes, alphas;
    T temp = mod;

    // 分解模数
    for (T i = 2; i * i <= temp; i++) {
        if (temp % i == 0) {
            primes.push_back(i);
            T alpha = 0;
            while (temp % i == 0) {
                temp /= i;
                alpha++;
            }
            alphas.push_back(alpha);
        }
    }
    if (temp > 1) {
        primes.push_back(temp);
        alphas.push_back(1);
    }

    // 使用CRT合并结果
    vector<T> a, moduli;
    for (int i = 0; i < primes.size(); i++) {
        T pk = power(primes[i], alphas[i], LLONG_MAX);
        a.push_back(calc_comb(n, m, primes[i], alphas[i]));
        moduli.push_back(pk);
    }

    return crt(a, moduli);
}
}
;

// 使用示例
/*
ExLucas<long long> exlucas;
cout << exlucas.exlucas(1000000, 500000, 999999937) << endl;
*/
]=]),

-- 04_Math\Combinatorics\Basic\Factorial.h
ps("04_math_combinatorics_basic_factorial_h", [=[

/**
 * 阶乘预处理模板
 * 功能：阶乘、阶乘逆元、组合数、排列数
 * 时间复杂度：预处理O(n)，查询O(1)
 */

using ll = long long;

// 阶乘预处理模板
template <int MAXN = 200005, ll MOD = 1e9 + 7>
struct Factorial {
    ll fac[MAXN], inv_fac[MAXN];

    Factorial() { init(); }

    ll power(ll a, ll b) {
        ll res = 1;
        a %= MOD;
        while (b) {
            if (b & 1) res = res * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return res;
    }

    void init() {
        fac[0] = 1;
        for (int i = 1; i < MAXN; i++) { fac[i] = fac[i - 1] * i % MOD; }
        inv_fac[MAXN - 1] = power(fac[MAXN - 1], MOD - 2);
        for (int i = MAXN - 2; i >= 0; i--) { inv_fac[i] = inv_fac[i + 1] * (i + 1) % MOD; }
    }

    ll factorial(int n) { return n >= MAXN ? 0 : fac[n]; }

    ll inv_factorial(int n) { return n >= MAXN ? 0 : inv_fac[n]; }

    ll C(int n, int m) {
        if (n < m || m < 0 || n >= MAXN) return 0;
        return fac[n] * inv_fac[m] % MOD * inv_fac[n - m] % MOD;
    }

    ll A(int n, int m) {
        if (n < m || m < 0 || n >= MAXN) return 0;
        return fac[n] * inv_fac[n - m] % MOD;
    }

    ll inv(int n) { return n >= MAXN ? 0 : fac[n - 1] * inv_fac[n] % MOD; }
};

// 威尔逊定理相关
struct Wilson {
    // 威尔逊定理：(p-1)! ≡ -1 (mod p) 当且仅当 p 是质数
    bool is_prime_wilson(ll p) {
        if (p <= 1) return false;
        if (p == 2) return true;

        ll fact = 1;
        for (ll i = 2; i < p; i++) { fact = fact * i % p; }
        return fact == p - 1;
    }
};

// 排列数 P(n, k) = A(n, k)
ll P(int n, int k) {
    if (k < 0 || k > n) return 0;
    return fact[n] * inv_fact[n - k] % MOD;
}

// 多重组合数 H(n, k) = C(n+k-1, k)
ll H(int n, int k) { return C(n + k - 1, k); }

// 卡特兰数 Cat(n) = C(2n, n) / (n+1)
ll catalan(int n) {
    if (n == 0) return 1;
    return C(2 * n, n) * modinv(n + 1) % MOD;
}

// 斯特林数第一类 s(n, k) - 符号的
// n个不同元素构成k个循环的方案数
vector<vector<ll>> stirling1(int n) {
    vector<vector<ll>> s(n + 1, vector<ll>(n + 1, 0));
    s[0][0] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) { s[i][j] = ((ll)(i - 1) * s[i - 1][j] % MOD + s[i - 1][j - 1]) % MOD; }
    }
    return s;
}

// 斯特林数第二类 S(n, k)
// n个不同元素分成k个非空子集的方案数
vector<vector<ll>> stirling2(int n) {
    vector<vector<ll>> S(n + 1, vector<ll>(n + 1, 0));
    S[0][0] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) { S[i][j] = ((ll)j * S[i - 1][j] % MOD + S[i - 1][j - 1]) % MOD; }
    }
    return S;
}

// 贝尔数 B(n) - n个元素的所有划分数
ll bell_number(int n) {
    vector<vector<ll>> S = stirling2(n);
    ll result = 0;
    for (int k = 0; k <= n; k++) { result = (result + S[n][k]) % MOD; }
    return result;
}

// 错排数 D(n)
ll derangement(int n) {
    if (n == 0) return 1;
    if (n == 1) return 0;

    vector<ll> d(n + 1);
    d[0] = 1;
    d[1] = 0;

    for (int i = 2; i <= n; i++) { d[i] = ((ll)(i - 1) * (d[i - 1] + d[i - 2])) % MOD; }
    return d[n];
}

// 分拆数 - 将n分拆成若干正整数之和的方案数
vector<ll> partition_numbers(int n) {
    vector<ll> p(n + 1, 0);
    p[0] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = i; j <= n; j++) { p[j] = (p[j] + p[j - i]) % MOD; }
    }
    return p;
}

// 欧拉函数值的计算（单个）
ll euler_phi(ll n) {
    ll result = n;
    for (ll i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0) n /= i;
            result -= result / i;
        }
    }
    if (n > 1) result -= result / n;
    return result;
}

// 二项式定理展开系数
vector<ll> binomial_expansion(int n) {
    vector<ll> coeffs(n + 1);
    for (int k = 0; k <= n; k++) { coeffs[k] = C(n, k); }
    return coeffs;
}
]=]),

}
