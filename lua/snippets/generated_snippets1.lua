-- Auto-generated LuaSnip snippets
local ls = require("luasnip")
local ps = ls.parser.parse_snippet

return {

-- 00_Common\CompileSettings.h
ps("00_common_compilesettings_h", [=[

// 编译器优化设置
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

// 类型定义
using ll = long long;
using ull = unsigned long long;
using i128 = __int128;
using ld = long double;
using pii = pair<int, int>;
using pll = pair<ll, ll>;

// 常用宏定义
#define sz(x) ((int)(x).size())
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define pb push_back
#define eb emplace_back
#define fi first
#define se second

// 调试宏定义
#ifndef ONLINE_JUDGE
#define dbg(x) cerr << #x << " = " << x << '\n'
#define dbg2(x, y) cerr << #x << " = " << x << ", " << #y << " = " << y << '\n'
#define dbgv(v)                        \
    cerr << #v << ": ";                \
    for (auto x : v) cerr << x << ' '; \
    cerr << '\n'
#else
#define dbg(x)
#define dbg2(x, y)
#define dbgv(v)
#endif

// 常用常量
constexpr int INF = 1e9 + 7;
constexpr ll LINF = 1e18 + 7;
constexpr int MOD = 1e9 + 7;
constexpr int MOD2 = 998244353;
constexpr ld EPS = 1e-9;
const ld PI = acos(-1);

// 方向数组 (上下左右)
constexpr int dx[] = {-1, 1, 0, 0};
constexpr int dy[] = {0, 0, -1, 1};

// 八方向数组
constexpr int dx8[] = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr int dy8[] = {-1, 0, 1, -1, 1, -1, 0, 1};

// 快速幂模板
template <typename T>
T qpow(T a, T b, T mod = MOD) {
    T res = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}

// 模运算工具函数
template <typename T>
T add(T a, T b, T mod = MOD) {
    return (a + b) % mod;
}

template <typename T>
T sub(T a, T b, T mod = MOD) {
    return (a - b + mod) % mod;
}

template <typename T>
T mul(T a, T b, T mod = MOD) {
    return a * b % mod;
}

template <typename T>
T inv(T a, T mod = MOD) {
    return qpow(a, mod - 2, mod);
}

template <typename T>
T div_mod(T a, T b, T mod = MOD) {
    return mul(a, inv(b, mod), mod);
}
]=]),

-- 00_Common\Coordinate.h
ps("00_common_coordinate_h", [=[

// 坐标离散化工具类
template <typename T>
struct Coordinate {
    vector<T> vals;
    bool sorted = false;

    Coordinate() {}
    Coordinate(const vector<T>& v) : vals(v) {}

    // 添加单个值
    void add(const T& val) {
        vals.push_back(val);
        sorted = false;
    }

    // 添加多个值
    void add(const vector<T>& v) {
        for (const auto& val : v) vals.push_back(val);
        sorted = false;
    }

    // 构建离散化数组(去重排序)
    void build() {
        sort(vals.begin(), vals.end());
        vals.erase(unique(vals.begin(), vals.end()), vals.end());
        sorted = true;
    }

    // 获取值在离散化数组中的位置
    int get(const T& val) {
        if (!sorted) build();
        return lower_bound(vals.begin(), vals.end(), val) - vals.begin();
    }

    // 根据位置获取值
    T operator[](int idx) {
        if (!sorted) build();
        return vals[idx];
    }

    // 获取离散化后的数组大小
    int size() {
        if (!sorted) build();
        return vals.size();
    }

    // 查找值在离散化数组中的位置，不存在返回-1
    int find(const T& val) {
        if (!sorted) build();
        auto it = lower_bound(vals.begin(), vals.end(), val);
        if (it != vals.end() && *it == val) { return it - vals.begin(); }
        return -1;
    }

    // 获取第一个大于等于val的位置
    int lower(const T& val) {
        if (!sorted) build();
        return lower_bound(vals.begin(), vals.end(), val) - vals.begin();
    }

    // 获取第一个大于val的位置
    int upper(const T& val) {
        if (!sorted) build();
        return upper_bound(vals.begin(), vals.end(), val) - vals.begin();
    }
};

// 二维坐标离散化
template <typename T>
struct Coordinate2D {
    Coordinate<T> x_coord, y_coord;

    // 添加单个点
    void add(const T& x, const T& y) {
        x_coord.add(x);
        y_coord.add(y);
    }

    // 添加多个点
    void add(const vector<pair<T, T>>& points) {
        for (const auto& [x, y] : points) { add(x, y); }
    }

    // 构建离散化数组
    void build() {
        x_coord.build();
        y_coord.build();
    }

    // 获取点的离散化坐标
    pair<int, int> get(const T& x, const T& y) { return {x_coord.get(x), y_coord.get(y)}; }

    // 根据离散化坐标获取原坐标
    pair<T, T> operator[](const pair<int, int>& idx) { return {x_coord[idx.first], y_coord[idx.second]}; }

    // 获取离散化后的二维大小
    pair<int, int> size() { return {x_coord.size(), y_coord.size()}; }
};
]=]),

-- 00_Common\Debug.h
ps("00_common_debug_h", [=[
#pragma once
#include <bitset>
#include <iostream>
#include <map>
#include <print>
#include <queue>
#include <ranges>
#include <set>
#include <stack>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T>
concept StringLike = std::same_as<std::decay_t<T>, std::string> || std::same_as<std::decay_t<T>, std::string_view>;

template <typename T>
concept Pair = requires(T a) {
    { a.first };
    { a.second };
} && !StringLike<T>;

template <typename T>
concept Tuple = requires { typename std::tuple_size<std::remove_cvref_t<T>>::type; } && std::tuple_size_v<std::remove_cvref_t<T>> >= 2 && !Pair<T> && !StringLike<T>;

template <typename T>
concept Queue = requires(T t) {
    typename T::value_type;
    typename T::container_type;
    t.front();
    t.back();
    t.size();
    t.empty();
    t.push(std::declval<typename T::value_type>());
    t.pop();
} && std::same_as<std::decay_t<T>, std::queue<typename T::value_type, typename T::container_type>>;

template <typename T>
concept Stack = requires(T t) {
    typename T::value_type;
    typename T::container_type;
    t.top();
    t.size();
    t.empty();
    t.push(std::declval<typename T::value_type>());
    t.pop();
} && std::same_as<std::decay_t<T>, std::stack<typename T::value_type, typename T::container_type>>;

template <typename T>
concept PriorityQueue = requires(T t) {
    typename T::value_type;
    typename T::container_type;
    t.top();
    t.size();
    t.empty();
    t.push(std::declval<typename T::value_type>());
    t.pop();
} && std::same_as<std::decay_t<T>, std::priority_queue<typename T::value_type, typename T::container_type>>;

template <typename T>
concept AssociativeContainer =
    requires(T t) {
        typename T::key_type;
        typename T::value_type;
        t.begin();
        t.end();
        t.size();
        t.empty();
    } && (std::same_as<std::decay_t<T>, std::set<typename T::key_type>> || std::same_as<std::decay_t<T>, std::multiset<typename T::key_type>> ||
          std::same_as<std::decay_t<T>, std::map<typename T::key_type, typename T::mapped_type>> ||
          std::same_as<std::decay_t<T>, std::multimap<typename T::key_type, typename T::mapped_type>> || std::same_as<std::decay_t<T>, std::unordered_set<typename T::key_type>> ||
          std::same_as<std::decay_t<T>, std::unordered_multiset<typename T::key_type>> ||
          std::same_as<std::decay_t<T>, std::unordered_map<typename T::key_type, typename T::mapped_type>> ||
          std::same_as<std::decay_t<T>, std::unordered_multimap<typename T::key_type, typename T::mapped_type>>);

template <typename T>
concept VectorBitset = requires(T t) {
    typename T::value_type;
    requires std::same_as<std::decay_t<T>, std::vector<typename T::value_type>>;
    requires requires(typename T::value_type v) {
        { v.size() } -> std::same_as<std::size_t>;
        { v.to_string() } -> std::same_as<std::string>;
        { v[std::size_t{}] } -> std::same_as<typename std::decay_t<typename T::value_type>::reference>;
    };
};

template <typename T>
concept Range = requires(T a) {
    std::ranges::begin(a);
    std::ranges::end(a);
} && !StringLike<T> && !Pair<T> && !Tuple<T> && !Queue<T> && !Stack<T> && !PriorityQueue<T> && !AssociativeContainer<T> && !VectorBitset<T>;

template <typename T>
concept Range2D = Range<T> && requires(T t) { typename T::value_type; } && Range<typename T::value_type>;

void _print_one(const auto& x, int indent = 0)
    requires(!Queue<std::remove_cvref_t<decltype(x)>> && !Stack<std::remove_cvref_t<decltype(x)>> && !PriorityQueue<std::remove_cvref_t<decltype(x)>> &&
             !AssociativeContainer<std::remove_cvref_t<decltype(x)>> && !Pair<std::remove_cvref_t<decltype(x)>> && !Tuple<std::remove_cvref_t<decltype(x)>> &&
             !Range<std::remove_cvref_t<decltype(x)>> && !VectorBitset<std::remove_cvref_t<decltype(x)>>)
{
    if constexpr (StringLike<decltype(x)>) {
        std::print(std::cerr, "\"{}\"", x);
    } else {
        std::print(std::cerr, "{}", x);
    }
}

template <Pair P>
void _print_one(const P& p, int indent) {
    std::print(std::cerr, "(");
    _print_one(p.first, indent);
    std::print(std::cerr, ", ");
    _print_one(p.second, indent);
    std::print(std::cerr, ")");
}

template <Tuple T>
void _print_one(const T& t, int indent) {
    std::print(std::cerr, "(");
    bool first = true;
    std::apply(
        [&](auto&&... elems) {
            ((std::print(std::cerr, "{}{}", first ? "" : ", ", elems), first = false, _print_one(elems, indent)), ...);
        },
        t);
    std::print(std::cerr, ")");
}

template <Range2D R>
void _print_one(const R& r, int indent) {
    std::println(std::cerr, "{{");
    int idx = 0;
    for (auto&& row : r) {
        std::print(std::cerr, "{: <{}}[{}]: {{", "", indent + 2, idx++);
        bool first = true;
        for (auto&& e : row) {
            std::print(std::cerr, "{}{}", first ? "" : ", ", "");
            first = false;
            _print_one(e, indent + 2);
        }
        std::println(std::cerr, "}}");
    }
    // 动态缩进替代非constexpr格式化
    std::print(std::cerr, "{}{}", std::string(indent, ' '), "}");
}

template <VectorBitset VB>
void _print_one(const VB& vb, int indent) {
    if (vb.empty()) {
        std::print(std::cerr, "{{}}");
        return;
    }

    std::size_t max_size = 0;
    for (const auto& bitset : vb) {
        max_size = std::max(max_size, bitset.size());
    }

    std::println(std::cerr, "{{");
    int idx = 0;
    for (const auto& bitset : vb) {
        std::print(std::cerr, "{: <{}}[{}]: ", "", indent + 2, idx++);
        std::string bitstr = bitset.to_string();
        std::size_t leading_zeros = max_size - bitset.size();
        for (std::size_t i = 0; i < leading_zeros; ++i) {
            std::print(std::cerr, "0");
        }
        std::println(std::cerr, "{}", bitstr);
    }
    // 动态缩进替代非constexpr格式化
    std::print(std::cerr, "{}{}", std::string(indent, ' '), "}");
}

template <Range R>
void _print_one(const R& r, int indent)
    requires(!Range2D<R>)
{
    std::print(std::cerr, "{{");
    bool first = true;
    for (auto&& e : r) {
        std::print(std::cerr, "{}{}", first ? "" : ", ", "");
        first = false;
        _print_one(e, indent);
    }
    std::print(std::cerr, "}}");
}

template <Queue Q>
void _print_one(Q q, int indent = 0) {
    std::print(std::cerr, "queue{{");
    bool first = true;
    while (!q.empty()) {
        std::print(std::cerr, "{}{}", first ? "" : ", ", "");
        first = false;
        _print_one(q.front(), indent);
        q.pop();
    }
    std::print(std::cerr, "}}");
}

template <Stack S>
void _print_one(S s, int indent = 0) {
    std::print(std::cerr, "stack{{");
    std::vector<typename S::value_type> elements;
    while (!s.empty()) {
        elements.push_back(s.top());
        s.pop();
    }

    bool first = true;
    for (auto it = elements.rbegin(); it != elements.rend(); ++it) {
        std::print(std::cerr, "{}{}", first ? "" : ", ", "");
        first = false;
        _print_one(*it, indent);
    }
    std::print(std::cerr, "}}");
}

template <PriorityQueue PQ>
void _print_one(PQ pq, int indent = 0) {
    std::print(std::cerr, "priority_queue{{");
    std::vector<typename PQ::value_type> elements;
    while (!pq.empty()) {
        elements.push_back(pq.top());
        pq.pop();
    }

    bool first = true;
    for (const auto& elem : elements) {
        std::print(std::cerr, "{}{}", first ? "" : ", ", "");
        first = false;
        _print_one(elem, indent);
    }
    std::print(std::cerr, "}}");
}

template <AssociativeContainer AC>
void _print_one(const AC& ac, int indent = 0) {
    if constexpr (requires { typename AC::mapped_type; }) {
        std::print(std::cerr, "map{{");
    } else {
        std::print(std::cerr, "set{{");
    }

    bool first = true;
    for (const auto& elem : ac) {
        std::print(std::cerr, "{}{}", first ? "" : ", ", "");
        first = false;
        _print_one(elem, indent);
    }
    std::print(std::cerr, "}}");
}

template <typename T>
void _debug_print_args(const char* names, const T& arg) {
    std::print(std::cerr, "{} = ", names);
    _print_one(arg, 0);
    std::println(std::cerr, "");
}

template <typename T, typename... Args>
void _debug_print_args(const char* names, const T& arg, const Args&... args) {
    const char* comma = names;
    while (*comma && *comma != ',') comma++;

    std::print(std::cerr, "{: <{}} = ", std::string(names, comma - names), 0);
    _print_one(arg, 0);
    std::println(std::cerr, "");

    if (*comma == ',') {
        comma++;
        while (*comma == ' ') comma++;
        _debug_print_args(comma, args...);
    }
}

#define debug(...)                                                    \
    do {                                                              \
        std::println(std::cerr, "=== DEBUG [Line {}] ===", __LINE__); \
        _debug_print_args(#__VA_ARGS__, __VA_ARGS__);                 \
        std::println(std::cerr, "=============");                     \
    } while (0)
]=]),

-- 00_Common\FastIO.h
ps("00_common_fastio_h", [=[
void wr() {}
bool rd() { return true; }
void _wr(char c) { putchar(c); }
void _wr(const string &s) { fputs(s.c_str(), stdout); }
bool rd(char &c) {
    int ch = getchar();
    while (isspace(ch)) ch = getchar();
    if (ch == EOF) return false;
    c = ch;
    return true;
}
bool rd(string &s) {
    s.clear();
    int ch = getchar();
    while (isspace(ch)) ch = getchar();
    if (ch == EOF) return false;
    while (!isspace(ch) && ch != EOF) {
        s += (char)ch;
        ch = getchar();
    }
    return true;
}
template <typename T>
void _wr(T x) {
    if (x < 0) {
        putchar('-');
        x = -x;
    }
    if (x == 0) {
        putchar('0');
        return;
    }
    char buf[21];
    int i = 0;
    while (x > 0) {
        buf[i++] = x % 10 + '0';
        x /= 10;
    }
    while (i-- > 0) putchar(buf[i]);
}
template <typename T, typename... Args>
void wr(T x, Args... args) {
    _wr(x);
    if (sizeof...(args) > 0) putchar(' ');
    wr(args...);
}
template <typename... Args>
void wrln(Args... args) {
    wr(args...);
    putchar('\n');
}
template <typename T, typename... Args>
bool rd(T &x, Args &...args) {
    x = 0;
    int f = 1;
    int ch = getchar();
    while (true) {
        if (ch >= '0' && ch <= '9') break;
        if (ch == EOF) return false;
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    x *= f;
    return rd(args...);
}
]=]),

-- 00_Common\ModularArithmetic.h
ps("00_common_modulararithmetic_h", [=[
#pragma once
#include <bits/stdc++.h>

constexpr long long exgcd(long long a, long long b, long long &x, long long &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    long long d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

template <class T>
constexpr T power(T a, long long b) {
    T res = 1;
    for (; b; b /= 2, a *= a) {
        if (b % 2) {
            res *= a;
        }
    }
    return res;
}

constexpr long long safe_mul(long long a, long long b, long long p) {
    long long res = a * b - static_cast<long long>(static_cast<long double>(a) * b / p) * p;
    res %= p;
    if (res < 0) {
        res += p;
    }
    return res;
}

template <long long P>
struct MInt {
    long long x;

    constexpr MInt() : x{} {}
    constexpr MInt(long long val) : x{norm(val % getMod())} {}

    static long long Mod;
    constexpr static long long getMod() { return P > 0 ? P : Mod; }

    static void setMod(long long Mod_) {
        if constexpr (P == 0) {
            Mod = Mod_;
        }
    }

    constexpr long long norm(long long val) const {
        if (val < 0) val += getMod();
        if (val >= getMod()) val -= getMod();
        return val;
    }

    constexpr long long val() const { return x; }

    explicit constexpr operator long long() const { return x; }

    constexpr MInt operator-() const {
        MInt res;
        res.x = norm(getMod() - x);
        return res;
    }

    constexpr MInt inv() const {
        assert(x != 0);
        return power(*this, getMod() - 2);
    }

    constexpr MInt inv_exgcd() const {
        assert(x != 0);
        long long inv_x, y;
        long long d = exgcd(x, getMod(), inv_x, y);
        assert(d == 1);
        return MInt(inv_x);
    }

    constexpr MInt &operator*=(MInt rhs) & {
        if (getMod() > 2000000000) {
            x = safe_mul(x, rhs.x, getMod());
        } else {
            x = x * rhs.x % getMod();
        }
        return *this;
    }

    constexpr MInt &operator+=(MInt rhs) & {
        x = norm(x + rhs.x);
        return *this;
    }
    constexpr MInt &operator-=(MInt rhs) & {
        x = norm(x - rhs.x);
        return *this;
    }
    constexpr MInt &operator/=(MInt rhs) & { return *this *= rhs.inv(); }

    friend constexpr MInt operator*(MInt lhs, MInt rhs) {
        MInt res = lhs;
        res *= rhs;
        return res;
    }
    friend constexpr MInt operator+(MInt lhs, MInt rhs) {
        MInt res = lhs;
        res += rhs;
        return res;
    }
    friend constexpr MInt operator-(MInt lhs, MInt rhs) {
        MInt res = lhs;
        res -= rhs;
        return res;
    }
    friend constexpr MInt operator/(MInt lhs, MInt rhs) {
        MInt res = lhs;
        res /= rhs;
        return res;
    }

    friend constexpr bool operator==(MInt lhs, MInt rhs) { return lhs.val() == rhs.val(); }
    friend constexpr bool operator!=(MInt lhs, MInt rhs) { return lhs.val() != rhs.val(); }

    friend std::istream &operator>>(std::istream &is, MInt &a) {
        long long v;
        is >> v;
        a = MInt(v);
        return is;
    }
    friend std::ostream &operator<<(std::ostream &os, const MInt &a) { return os << a.val(); }
};

template <long long P>
long long MInt<P>::Mod = 1000000007;

using Z = MInt<998244353>;
]=]),

-- 00_Common\Random.h
ps("00_common_random_h", [=[

// 随机数生成器
struct Random {
    mt19937 rng;

    Random() : rng(chrono::steady_clock::now().time_since_epoch().count()) {}
    Random(uint32_t seed) : rng(seed) {}

    // 生成 [l, r] 范围内的整数
    int randint(int l, int r) { return uniform_int_distribution<int>(l, r)(rng); }

    // 生成 [l, r] 范围内的长整数
    long long randll(long long l, long long r) { return uniform_int_distribution<long long>(l, r)(rng); }

    // 生成 [0, 1) 范围内的实数
    double randdouble() { return uniform_real_distribution<double>(0.0, 1.0)(rng); }

    // 生成 [l, r) 范围内的实数
    double randdouble(double l, double r) { return uniform_real_distribution<double>(l, r)(rng); }

    // 以概率p返回true，否则返回false
    bool randbool(double p = 0.5) { return bernoulli_distribution(p)(rng); }

    // 随机打乱数组
    template <typename T>
    void shuffle(vector<T>& v) {
        shuffle(v.begin(), v.end(), rng);
    }

    // 从数组中随机选择k个元素
    template <typename T>
    vector<T> sample(const vector<T>& v, int k) {
        vector<int> indices(v.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices);

        vector<T> result;
        for (int i = 0; i < min(k, (int)v.size()); i++) { result.push_back(v[indices[i]]); }
        return result;
    }

    // 生成随机字符串
    string randstring(int len, const string& chars = "abcdefghijklmnopqrstuvwxyz") {
        string result;
        for (int i = 0; i < len; i++) { result += chars[randint(0, chars.size() - 1)]; }
        return result;
    }

    // 生成随机排列
    vector<int> randperm(int n) {
        vector<int> perm(n);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm);
        return perm;
    }

    // 生成随机树
    vector<pair<int, int>> randtree(int n) {
        vector<pair<int, int>> edges;
        vector<int> parent(n);
        parent[0] = -1;

        for (int i = 1; i < n; i++) {
            parent[i] = randint(0, i - 1);
            edges.push_back({parent[i], i});
        }

        return edges;
    }

    // 生成随机图
    vector<pair<int, int>> randgraph(int n, int m) {
        set<pair<int, int>> edge_set;
        vector<pair<int, int>> edges;

        while (edges.size() < m) {
            int u = randint(0, n - 1);
            int v = randint(0, n - 1);
            if (u != v && edge_set.find({min(u, v), max(u, v)}) == edge_set.end()) {
                edge_set.insert({min(u, v), max(u, v)});
                edges.push_back({u, v});
            }
        }

        return edges;
    }
};

// 全局随机数生成器
Random rnd;
]=]),

-- 00_Common\Timing.h
ps("00_common_timing_h", [=[

// 计时器类
struct Timer {
    chrono::high_resolution_clock::time_point start_time;

    Timer() { start(); }

    void start() { start_time = chrono::high_resolution_clock::now(); }

    // 获取经过的时间(秒)
    double elapsed() const {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration<double>(end_time - start_time).count();
    }

    // 获取经过的时间(毫秒)
    long long elapsed_ms() const {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    }

    // 获取经过的时间(微秒)
    long long elapsed_us() const {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    }

    // 获取经过的时间(纳秒)
    long long elapsed_ns() const {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
    }

    // 打印经过的时间(秒)
    void print_elapsed(const string& msg = "") const {
        if (!msg.empty()) cout << msg << ": ";
        cout << elapsed() << "s" << endl;
    }

    // 打印经过的时间(毫秒)
    void print_elapsed_ms(const string& msg = "") const {
        if (!msg.empty()) cout << msg << ": ";
        cout << elapsed_ms() << "ms" << endl;
    }
};

// 计时函数执行时间
template <typename Func>
double time_func(Func&& func) {
    Timer timer;
    func();
    return timer.elapsed();
}

template <typename Func>
pair<double, decltype(Func())> time_func_with_result(Func&& func) {
    Timer timer;
    auto result = func();
    return {timer.elapsed(), result};
}

// 自动计时RAII类
struct AutoTimer {
    Timer timer;
    string name;

    AutoTimer(const string& n = "") : name(n) {}

    ~AutoTimer() {
        if (name.empty()) {
            cout << "Elapsed: " << timer.elapsed() << "s" << endl;
        } else {
            cout << name << ": " << timer.elapsed() << "s" << endl;
        }
    }
};

// 性能测试
template <typename Func>
void benchmark(Func&& func, int iterations = 1000, const string& name = "") {
    vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        Timer timer;
        func();
        times.push_back(timer.elapsed());
    }

    sort(times.begin(), times.end());

    double sum = accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / iterations;
    double median = times[iterations / 2];
    double min_time = times[0];
    double max_time = times[iterations - 1];

    if (!name.empty()) cout << name << " ";
    cout << "Benchmark Results:" << endl;
    cout << "  Iterations: " << iterations << endl;
    cout << "  Average: " << avg << "s" << endl;
    cout << "  Median: " << median << "s" << endl;
    cout << "  Min: " << min_time << "s" << endl;
    cout << "  Max: " << max_time << "s" << endl;
}

// 代码块计时宏
#define TIME_BLOCK(name) AutoTimer _timer(name)
]=]),

-- 00_Common\i128.h
ps("00_common_i128_h", [=[

using i128 = __int128;

// 128位整数输出
ostream& operator<<(ostream& os, i128 n) {
    if (n == 0) return os << 0;
    if (n < 0) {
        os << '-';
        n = -n;
    }
    string s;
    while (n > 0) {
        s += char('0' + n % 10);
        n /= 10;
    }
    reverse(s.begin(), s.end());
    return os << s;
}

// 128位整数输入
istream& operator>>(istream& is, i128& n) {
    string s;
    is >> s;
    n = 0;
    bool neg = false;
    int start = 0;
    if (s[0] == '-') {
        neg = true;
        start = 1;
    }
    for (int i = start; i < s.size(); i++) { n = n * 10 + (s[i] - '0'); }
    if (neg) n = -n;
    return is;
}

// 字符串转128位整数
i128 toi128(const string& s) {
    i128 n = 0;
    bool neg = false;
    int start = 0;
    if (s[0] == '-') {
        neg = true;
        start = 1;
    }
    for (int i = start; i < s.size(); i++) { n = n * 10 + (s[i] - '0'); }
    if (neg) n = -n;
    return n;
}

// 128位整数开方
i128 sqrti128(i128 n) {
    if (n < 0) return -1;
    i128 lo = 0, hi = 2e18;
    while (lo < hi) {
        i128 mid = (lo + hi + 1) / 2;
        if (mid <= n / mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// 128位整数最大公约数
i128 gcd(i128 a, i128 b) { return b ? gcd(b, a % b) : a; }

// 128位整数最小公倍数
i128 lcm(i128 a, i128 b) { return a / gcd(a, b) * b; }

// 128位整数快速幂
i128 qpow(i128 a, i128 b) {
    i128 res = 1;
    while (b > 0) {
        if (b & 1) res *= a;
        a *= a;
        b >>= 1;
    }
    return res;
}

// 128位整数模快速幂
i128 qpow(i128 a, i128 b, i128 mod) {
    i128 res = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}
]=]),

-- 01_Data_Structures\Advanced\KDTree.h
ps("01_data_structures_advanced_kdtree_h", [=[

// K-D树模板，支持K维空间的最近邻查询和范围查询
template <int K>
struct KDTree {
    // K维点结构
    struct Point {
        int dim[K];  // K维坐标
        int id;      // 点的标识符

        Point() : id(-1) { memset(dim, 0, sizeof(dim)); }

        Point(const vector<int>& coords, int _id = -1) : id(_id) {
            for (int i = 0; i < K; i++) { dim[i] = coords[i]; }
        }
    };

    // K-D树节点结构
    struct Node {
        Point pt;                        // 节点存储的点
        int left, right;                 // 左右子树索引
        int min_coord[K], max_coord[K];  // 子树边界范围

        Node() : left(-1), right(-1) {}
    };

    vector<Node> tree;   // 树节点数组
    int root, node_cnt;  // 根节点索引和节点计数

    KDTree() : root(-1), node_cnt(0) {}

    // 构建K-D树
    int build(vector<Point>& points, int l, int r, int depth = 0) {
        if (l > r) return -1;

        int cur = node_cnt++;
        tree.resize(node_cnt);

        // 选择当前层的分割维度
        int dim = depth % K;

        // 按当前维度排序并选择中位数
        nth_element(points.begin() + l,
                    points.begin() + (l + r) / 2,
                    points.begin() + r + 1,
                    [dim](const Point& a, const Point& b) { return a.dim[dim] < b.dim[dim]; });

        int mid = (l + r) / 2;
        tree[cur].pt = points[mid];

        // 初始化当前节点的边界范围
        for (int i = 0; i < K; i++) { tree[cur].min_coord[i] = tree[cur].max_coord[i] = points[mid].dim[i]; }

        // 递归构建左右子树
        tree[cur].left = build(points, l, mid - 1, depth + 1);
        tree[cur].right = build(points, mid + 1, r, depth + 1);

        // 更新边界范围
        if (tree[cur].left != -1) {
            for (int i = 0; i < K; i++) {
                tree[cur].min_coord[i] = min(tree[cur].min_coord[i], tree[tree[cur].left].min_coord[i]);
                tree[cur].max_coord[i] = max(tree[cur].max_coord[i], tree[tree[cur].left].max_coord[i]);
            }
        }
        if (tree[cur].right != -1) {
            for (int i = 0; i < K; i++) {
                tree[cur].min_coord[i] = min(tree[cur].min_coord[i], tree[tree[cur].right].min_coord[i]);
                tree[cur].max_coord[i] = max(tree[cur].max_coord[i], tree[tree[cur].right].max_coord[i]);
            }
        }

        return cur;
    }

    // 计算两点间欧几里得距离
    double distance(const Point& a, const Point& b) {
        double sum = 0;
        for (int i = 0; i < K; i++) { sum += (a.dim[i] - b.dim[i]) * (a.dim[i] - b.dim[i]); }
        return sqrt(sum);
    }

    // 计算点到矩形区域的最小距离
    double min_distance_to_rect(const Point& pt, int node) {
        double sum = 0;
        for (int i = 0; i < K; i++) {
            if (pt.dim[i] < tree[node].min_coord[i]) {
                sum += (tree[node].min_coord[i] - pt.dim[i]) * (tree[node].min_coord[i] - pt.dim[i]);
            } else if (pt.dim[i] > tree[node].max_coord[i]) {
                sum += (pt.dim[i] - tree[node].max_coord[i]) * (pt.dim[i] - tree[node].max_coord[i]);
            }
        }
        return sqrt(sum);
    }

    // 最近邻查询的递归函数
    void nearest_neighbor(int node, const Point& target, Point& best, double& best_dist) {
        if (node == -1) return;

        // 检查当前节点是否更近
        double curr_dist = distance(tree[node].pt, target);
        if (curr_dist < best_dist) {
            best_dist = curr_dist;
            best = tree[node].pt;
        }

        // 计算当前节点的深度(分割维度)
        int depth = 0;
        int cur = root;
        while (cur != node) {
            if (target.dim[depth % K] < tree[cur].pt.dim[depth % K]) {
                cur = tree[cur].left;
            } else {
                cur = tree[cur].right;
            }
            depth++;
        }

        int split_dim = depth % K;
        int first, second;

        // 根据分割维度确定搜索顺序
        if (target.dim[split_dim] < tree[node].pt.dim[split_dim]) {
            first = tree[node].left;
            second = tree[node].right;
        } else {
            first = tree[node].right;
            second = tree[node].left;
        }

        // 优先搜索可能包含最近点的子树
        nearest_neighbor(first, target, best, best_dist);

        // 如果另一个子树可能包含更近的点，则继续搜索
        if (second != -1 && min_distance_to_rect(target, second) < best_dist) {
            nearest_neighbor(second, target, best, best_dist);
        }
    }

    // 查找最近邻点
    Point find_nearest(const Point& target) {
        Point best;
        double best_dist = 1e18;
        nearest_neighbor(root, target, best, best_dist);
        return best;
    }

    // 范围查询的递归函数
    void range_query(int node, const vector<int>& min_range, const vector<int>& max_range, vector<Point>& result) {
        if (node == -1) return;

        // 检查当前节点是否在查询范围内
        bool in_range = true;
        for (int i = 0; i < K; i++) {
            if (tree[node].pt.dim[i] < min_range[i] || tree[node].pt.dim[i] > max_range[i]) {
                in_range = false;
                break;
            }
        }
        if (in_range) { result.push_back(tree[node].pt); }

        // 检查左右子树是否与查询范围相交
        bool left_possible = true, right_possible = true;
        for (int i = 0; i < K; i++) {
            if (tree[node].left != -1) {
                if (tree[tree[node].left].max_coord[i] < min_range[i] ||
                    tree[tree[node].left].min_coord[i] > max_range[i]) {
                    left_possible = false;
                }
            }
            if (tree[node].right != -1) {
                if (tree[tree[node].right].max_coord[i] < min_range[i] ||
                    tree[tree[node].right].min_coord[i] > max_range[i]) {
                    right_possible = false;
                }
            }
        }

        // 递归搜索可能相交的子树
        if (left_possible) range_query(tree[node].left, min_range, max_range, result);
        if (right_possible) range_query(tree[node].right, min_range, max_range, result);
    }

    // 范围查询
    vector<Point> range_search(const vector<int>& min_range, const vector<int>& max_range) {
        vector<Point> result;
        range_query(root, min_range, max_range, result);
        return result;
    }

    // 构建K-D树的公共接口
    void build(vector<Point> points) {
        tree.clear();
        node_cnt = 0;
        if (!points.empty()) { root = build(points, 0, points.size() - 1); }
    }
};
]=]),

-- 01_Data_Structures\Advanced\ODT.h
ps("01_data_structures_advanced_odt_h", [=[

// 颜色段树(ODT/珂朵莉树)，用于处理区间赋值、区间加、区间第k小等操作
struct ODT {
    // 节点结构，表示一个颜色段[l, r]，值为val
    struct Node {
        int l, r;               // 区间左右端点
        mutable long long val;  // 区间的值(mutable允许在set中修改)

        Node(int l, int r, long long val) : l(l), r(r), val(val) {}

        // 按左端点排序
        bool operator<(const Node& o) const { return l < o.l; }
    };

    set<Node> tree;  // 用set维护所有颜色段

    ODT() {}

    // 分裂操作：在位置pos处分裂区间，返回[pos, r]区间的迭代器
    set<Node>::iterator split(int pos) {
        auto it = tree.lower_bound(Node(pos, -1, -1));
        if (it != tree.end() && it->l == pos) { return it; }

        --it;  // 找到包含pos的区间
        int l = it->l, r = it->r;
        long long val = it->val;
        tree.erase(it);

        // 分裂成[l, pos-1]和[pos, r]两个区间
        tree.insert(Node(l, pos - 1, val));
        return tree.insert(Node(pos, r, val)).first;
    }

    // 区间赋值：将[l, r]区间的值都设为val
    void assign(int l, int r, long long val) {
        auto itr = split(r + 1), itl = split(l);
        tree.erase(itl, itr);          // 删除[l, r]范围内的所有区间
        tree.insert(Node(l, r, val));  // 插入新的统一区间
    }

    // 区间加法：将[l, r]区间的值都加上val
    void add(int l, int r, long long val) {
        auto itr = split(r + 1), itl = split(l);
        for (; itl != itr; ++itl) { itl->val += val; }
    }

    // 查询[l, r]区间第k小的值
    long long kth(int l, int r, int k) {
        auto itr = split(r + 1), itl = split(l);
        vector<pair<long long, int>> vp;  // {值, 长度}

        for (; itl != itr; ++itl) { vp.push_back({itl->val, itl->r - itl->l + 1}); }

        sort(vp.begin(), vp.end());
        for (auto& p : vp) {
            k -= p.second;
            if (k <= 0) return p.first;
        }
        return -1;  // 不存在第k小
    }

    // 计算[l, r]区间内sum(val^x) mod y
    long long sum_pow(int l, int r, long long x, long long y) {
        auto itr = split(r + 1), itl = split(l);
        long long res = 0;
        for (; itl != itr; ++itl) { res = (res + (long long)(itl->r - itl->l + 1) * pow_mod(itl->val, x, y)) % y; }
        return res;
    }

    // 快速幂取模
    long long pow_mod(long long a, long long b, long long mod) {
        long long res = 1;
        a %= mod;
        while (b > 0) {
            if (b & 1) res = res * a % mod;
            a = a * a % mod;
            b >>= 1;
        }
        return res;
    }

    // 插入新区间(初始化时使用)
    void insert(int l, int r, long long val) { tree.insert(Node(l, r, val)); }

    // 调试输出所有区间
    void print() {
        for (auto& node : tree) { cout << "[" << node.l << ", " << node.r << "] = " << node.val << "\n"; }
    }
};
// odt.add(l, r, val);     // 区间加法
// long long kth_val = odt.kth(l, r, k);  // 区间第k小
]=]),

-- 01_Data_Structures\Hashing\StringHash.h
ps("01_data_structures_hashing_stringhash_h", [=[

using ll = long long;
using ull = unsigned long long;

/**
 * 字符串哈希模板
 * 功能特性:
 * - 双哈希防止冲突
 * - 正向和反向哈希计算
 * - 回文检测和分析
 * - 周期计算
 * 时间复杂度: O(n) 预处理, O(1) 查询
 */

namespace StringHashing {

namespace Detail {
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

// 生成随机哈希基数
ull generate_base() {
    uniform_int_distribution<ull> distrib(257, ULLONG_MAX);
    ull base = distrib(rng);
    return base | 1;  // 确保为奇数
}
}  // namespace Detail

struct Hasher {
    int n;               // 字符串长度
    string str;          // 原始字符串
    ull P1, P2;          // 两个哈希基数
    vector<ull> p1, p2;  // 基数的幂次
    vector<ull> f1, f2;  // 正向哈希值
    vector<ull> b1, b2;  // 反向哈希值

    Hasher(const string& s) : n(s.size()), str(s) {
        // 生成两个不同的随机基数
        P1 = Detail::generate_base();
        P2 = Detail::generate_base();
        while (P2 == P1) P2 = Detail::generate_base();

        // 初始化向量
        p1.resize(n + 1);
        p2.resize(n + 1);
        f1.resize(n + 1, 0);
        f2.resize(n + 1, 0);
        b1.resize(n + 1, 0);
        b2.resize(n + 1, 0);

        // 预计算基数的幂次
        p1[0] = p2[0] = 1;
        for (int i = 0; i < n; ++i) {
            p1[i + 1] = p1[i] * P1;
            p2[i + 1] = p2[i] * P2;
        }

        // 计算正向哈希值
        for (int i = 0; i < n; ++i) {
            ull val = (unsigned char)s[i];
            f1[i + 1] = f1[i] * P1 + val;
            f2[i + 1] = f2[i] * P2 + val;
        }

        // 计算反向哈希值
        for (int i = 0; i < n; ++i) {
            ull val = (unsigned char)s[n - 1 - i];
            b1[i + 1] = b1[i] * P1 + val;
            b2[i + 1] = b2[i] * P2 + val;
        }
    }

    // 获取子串 s[l...r] 的正向哈希值 (0索引, 包含边界)
    pair<ull, ull> get_forward_hash(int l, int r) const {
        if (l > r || l < 0 || r >= n) return {0, 0};
        ull val1 = f1[r + 1] - f1[l] * p1[r - l + 1];
        ull val2 = f2[r + 1] - f2[l] * p2[r - l + 1];
        return {val1, val2};
    }

    // 获取子串 s[l...r] 的反向哈希值 (0索引, 包含边界)
    pair<ull, ull> get_backward_hash(int l, int r) const {
        if (l > r || l < 0 || r >= n) return {0, 0};
        int rev_l = n - 1 - r;
        int rev_r = n - 1 - l;
        ull val1 = b1[rev_r + 1] - b1[rev_l] * p1[r - l + 1];
        ull val2 = b2[rev_r + 1] - b2[rev_l] * p2[r - l + 1];
        return {val1, val2};
    }

    // 检查子串 s[l...r] 是否为回文
    bool is_palindrome(int l, int r) const {
        if (l > r) return true;
        if (l < 0 || r >= n) return false;
        return get_forward_hash(l, r) == get_backward_hash(l, r);
    }

    // 找到以 center_idx 为中心的最长奇数长度回文的半径
    int get_odd_palindrome_radius(int center_idx) const {
        int low = 0, high = min(center_idx, n - 1 - center_idx);
        int ans = 0;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (is_palindrome(center_idx - mid, center_idx + mid)) {
                ans = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return ans;
    }

    // 找到以 left_center_idx 和 left_center_idx+1 之间为中心的最长偶数长度回文的半径
    int get_even_palindrome_radius(int left_center_idx) const {
        if (left_center_idx + 1 >= n) return 0;
        int low = 1, high = min(left_center_idx + 1, n - (left_center_idx + 1));
        int ans = 0;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (is_palindrome(left_center_idx - mid + 1, left_center_idx + mid)) {
                ans = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return ans;
    }

    // 找到最长回文子串，返回 {长度, 起始索引}
    pair<int, int> get_longest_palindrome() const {
        if (n == 0) return {0, 0};
        int max_len = 1, start_pos = 0;

        // 检查奇数长度回文
        for (int i = 0; i < n; ++i) {
            int radius = get_odd_palindrome_radius(i);
            int len = 2 * radius + 1;
            if (len > max_len) {
                max_len = len;
                start_pos = i - radius;
            }
        }

        // 检查偶数长度回文
        for (int i = 0; i < n - 1; ++i) {
            int radius = get_even_palindrome_radius(i);
            int len = 2 * radius;
            if (len > max_len) {
                max_len = len;
                start_pos = i - radius + 1;
            }
        }

        return {max_len, start_pos};
    }
};

}  // namespace StringHashing

// 计算回文子串的总数
ll count_palindromic_substrings(const string& s) {
    if (s.empty()) return 0;

    StringHashing::Hasher hasher(s);
    ll count = 0;

    // 计算奇数长度回文
    for (int i = 0; i < hasher.n; ++i) {
        int radius = hasher.get_odd_palindrome_radius(i);
        count += (radius + 1);
    }

    // 计算偶数长度回文
    for (int i = 0; i < hasher.n - 1; ++i) {
        int radius = hasher.get_even_palindrome_radius(i);
        count += radius;
    }

    return count;
}

// 找到字符串的最小周期
int minimal_period(const string& s) {
    if (s.empty()) return 0;

    StringHashing::Hasher hasher(s);
    int n = hasher.n;

    for (int period = 1; period < n; ++period) {
        if (hasher.get_forward_hash(0, n - 1 - period) == hasher.get_forward_hash(period, n - 1)) { return period; }
    }

    return n;  // 字符串没有小于自身的周期
}
]=]),

-- 01_Data_Structures\Linear\Deque.h
ps("01_data_structures_linear_deque_h", [=[

/**
 * 双端队列数据结构模板
 * 功能：
 * - 基础双端队列操作封装
 * - 滑动窗口最小值/最大值
 * - 单调队列应用
 * 时间复杂度：所有操作均摊O(1)
 */

// 基础双端队列应用封装
template <typename T>
struct DequeApp {
    deque<T> dq;

    // 基础操作
    void push_front(const T& val) { dq.push_front(val); }
    void push_back(const T& val) { dq.push_back(val); }
    void pop_front() {
        if (!dq.empty()) dq.pop_front();
    }
    void pop_back() {
        if (!dq.empty()) dq.pop_back();
    }

    T front() const { return dq.front(); }
    T back() const { return dq.back(); }
    T operator[](int idx) const { return dq[idx]; }

    int size() const { return dq.size(); }
    bool empty() const { return dq.empty(); }
    void clear() { dq.clear(); }

    // 维护固定大小的滑动窗口
    void maintain_window(int window_size) {
        while (dq.size() > window_size) { dq.pop_front(); }
    }

    // 获取极值（假设队列维护了有序性）
    T get_min() const { return dq.front(); }
    T get_max() const { return dq.back(); }
};

// 使用单调队列的滑动窗口最小值
template <typename T>
struct SlidingWindowMin {
    deque<pair<T, int>> dq;  // (值, 索引)
    int window_size;

    SlidingWindowMin(int ws) : window_size(ws) {}

    void push(const T& val, int idx) {
        // 移除窗口外的元素
        while (!dq.empty() && dq.front().second <= idx - window_size) { dq.pop_front(); }

        // 维护单调性（递增序列）
        while (!dq.empty() && dq.back().first >= val) { dq.pop_back(); }

        dq.push_back({val, idx});
    }

    T get_min() const { return dq.empty() ? T{} : dq.front().first; }

    bool empty() const { return dq.empty(); }
    void clear() { dq.clear(); }
};

// 使用单调队列的滑动窗口最大值
template <typename T>
struct SlidingWindowMax {
    deque<pair<T, int>> dq;  // (值, 索引)
    int window_size;

    SlidingWindowMax(int ws) : window_size(ws) {}

    void push(const T& val, int idx) {
        // 移除窗口外的元素
        while (!dq.empty() && dq.front().second <= idx - window_size) { dq.pop_front(); }

        // 维护单调性（递减序列）
        while (!dq.empty() && dq.back().first <= val) { dq.pop_back(); }

        dq.push_back({val, idx});
    }

    T get_max() const { return dq.empty() ? T{} : dq.front().first; }

    bool empty() const { return dq.empty(); }
    void clear() { dq.clear(); }
};
]=]),

-- 01_Data_Structures\Linear\MonotonicQueue.h
ps("01_data_structures_linear_monotonicqueue_h", [=[

/**
 * 单调队列模板
 * 功能特性:
 * - 维护单调性质（递增/递减）
 * - 滑动窗口最小值/最大值查询
 * - 高效的窗口操作
 * 时间复杂度: 所有操作均摊 O(1)
 */

template <typename T>
struct MonotonicQueue {
    deque<pair<T, int>> dq;  // (值, 索引)

    // 维护滑动窗口最小值的单调队列
    void push_min(const T& val, int idx) {
        // 移除 >= 当前值的元素以维护递增顺序
        while (!dq.empty() && dq.back().first >= val) { dq.pop_back(); }
        dq.push_back({val, idx});
    }

    // 维护滑动窗口最大值的单调队列
    void push_max(const T& val, int idx) {
        // 移除 <= 当前值的元素以维护递减顺序
        while (!dq.empty() && dq.back().first <= val) { dq.pop_back(); }
        dq.push_back({val, idx});
    }

    // 移除窗口外的过期元素
    void pop_expired(int left_bound) {
        while (!dq.empty() && dq.front().second < left_bound) { dq.pop_front(); }
    }

    // 获取当前窗口的极值
    T front_value() const { return dq.empty() ? T{} : dq.front().first; }

    // 获取极值对应的索引
    int front_index() const { return dq.empty() ? -1 : dq.front().second; }

    // 检查队列是否为空
    bool empty() const { return dq.empty(); }

    // 清空队列
    void clear() { dq.clear(); }
};

// 解决滑动窗口最小值问题
template <typename T>
vector<T> sliding_window_minimum(const vector<T>& arr, int k) {
    vector<T> result;
    MonotonicQueue<T> mq;

    for (int i = 0; i < arr.size(); i++) {
        mq.push_min(arr[i], i);     // 添加当前元素到单调队列
        mq.pop_expired(i - k + 1);  // 移除窗口外的元素

        if (i >= k - 1) {  // 当窗口大小达到k时开始记录结果
            result.push_back(mq.front_value());
        }
    }
    return result;
}

// 解决滑动窗口最大值问题
template <typename T>
vector<T> sliding_window_maximum(const vector<T>& arr, int k) {
    vector<T> result;
    MonotonicQueue<T> mq;

    for (int i = 0; i < arr.size(); i++) {
        mq.push_max(arr[i], i);     // 添加当前元素到单调队列
        mq.pop_expired(i - k + 1);  // 移除窗口外的元素

        if (i >= k - 1) {  // 当窗口大小达到k时开始记录结果
            result.push_back(mq.front_value());
        }
    }
    return result;
}

// 使用单调栈求直方图中的最大矩形面积
long long largest_rectangle_histogram(const vector<int>& heights) {
    vector<int> h = heights;
    h.push_back(0);  // 添加哨兵元素，确保所有柱子都被处理

    deque<int> stk;  // 单调栈（存储递增高度的索引）
    long long max_area = 0;

    for (int i = 0; i < h.size(); i++) {
        // 当前高度小于栈顶高度时，计算以栈顶为高的矩形面积
        while (!stk.empty() && h[stk.back()] > h[i]) {
            int height = h[stk.back()];  // 矩形的高度
            stk.pop_back();
            // 计算矩形的宽度：右边界到左边界的距离
            int width = stk.empty() ? i : i - stk.back() - 1;
            max_area = max(max_area, (long long)height * width);
        }
        stk.push_back(i);  // 将当前索引入栈
    }

    return max_area;
}
]=]),

-- 01_Data_Structures\Linear\SparseTable.h
ps("01_data_structures_linear_sparsetable_h", [=[

/**
 * 稀疏表模板
 * 功能特性:
 * - 区间最小值/最大值查询 (RMQ)
 * - 区间最大公约数/最小公倍数查询
 * - 支持幂等操作
 * 时间复杂度: O(n log n) 预处理, O(1) 查询
 */

template <typename T>
struct SparseTable {
    vector<vector<T>> st;  // 稀疏表，st[k][i] 表示从位置 i 开始长度为 2^k 的区间的值
    vector<int> lg;        // 预计算的对数值，lg[i] = floor(log2(i))
    int n;                 // 数组长度
    function<T(T, T)> op;  // 操作函数（必须是幂等且结合的）

    SparseTable() {}

    // op 应该是幂等结合操作（如 min, max, gcd, lcm）
    SparseTable(const vector<T>& arr, function<T(T, T)> operation) : n(arr.size()), op(operation) { build(arr); }

    void build(const vector<T>& arr) {
        // 预计算对数值
        lg.resize(n + 1);
        lg[1] = 0;
        for (int i = 2; i <= n; i++) { lg[i] = lg[i / 2] + 1; }

        int max_log = lg[n] + 1;
        st.assign(max_log, vector<T>(n));

        // 初始化第一层（长度为1的区间）
        for (int i = 0; i < n; i++) { st[0][i] = arr[i]; }

        // 构建稀疏表
        for (int k = 1; k < max_log; k++) {
            for (int i = 0; i + (1 << k) <= n; i++) {
                // st[k][i] = op(前半段, 后半段)
                st[k][i] = op(st[k - 1][i], st[k - 1][i + (1 << (k - 1))]);
            }
        }
    }

    // 查询区间 [l, r] (0索引，包含边界)
    T query(int l, int r) {
        if (l > r) return T{};
        int k = lg[r - l + 1];  // 找到最大的 k 使得 2^k <= 区间长度
        // 利用幂等性质：区间可以用两个重叠的子区间覆盖
        return op(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

// 区间最小值查询的特化版本
template <typename T>
struct RMQ {
    SparseTable<T> st;

    RMQ(const vector<T>& arr) {
        st = SparseTable<T>(arr, [](T a, T b) { return min(a, b); });
    }

    T query_min(int l, int r) { return st.query(l, r); }
};

// 区间最大值查询的特化版本
template <typename T>
struct RMaxQ {
    SparseTable<T> st;

    RMaxQ(const vector<T>& arr) {
        st = SparseTable<T>(arr, [](T a, T b) { return max(a, b); });
    }

    T query_max(int l, int r) { return st.query(l, r); }
};

// 区间最大公约数查询的特化版本
struct RGCDQ {
    SparseTable<long long> st;

    RGCDQ(const vector<long long>& arr) {
        st = SparseTable<long long>(arr, [](long long a, long long b) { return gcd(a, b); });
    }

    long long query_gcd(int l, int r) { return st.query(l, r); }
};

// 区间最小公倍数查询的特化版本（注意溢出问题）
struct RLCMQ {
    SparseTable<long long> st;

    RLCMQ(const vector<long long>& arr) {
        st = SparseTable<long long>(arr, [](long long a, long long b) {
            return (a / gcd(a, b)) * b;  // 先除后乘避免溢出
        });
    }

    long long query_lcm(int l, int r) { return st.query(l, r); }
};
]=]),

-- 01_Data_Structures\Linear\Stack.h
ps("01_data_structures_linear_stack_h", [=[

/**
 * 基于栈的算法模板
 * 功能特性:
 * - 单调栈应用
 * - 寻找下一个更大/更小元素
 * - 直方图中的最大矩形
 * - 有效括号问题
 * 时间复杂度: 大多数操作 O(n)
 */

template <typename T>
struct MonotonicStack {
    stack<pair<T, int>> st;  // (值, 索引)

    // 寻找每个位置的下一个更大元素
    vector<int> next_greater(const vector<T>& arr) {
        vector<int> result(arr.size(), -1);
        stack<int> st;

        for (int i = 0; i < arr.size(); i++) {
            while (!st.empty() && arr[st.top()] < arr[i]) {
                result[st.top()] = i;
                st.pop();
            }
            st.push(i);
        }

        return result;
    }

    // 寻找每个位置的下一个更小元素
    vector<int> next_smaller(const vector<T>& arr) {
        vector<int> result(arr.size(), -1);
        stack<int> st;

        for (int i = 0; i < arr.size(); i++) {
            while (!st.empty() && arr[st.top()] > arr[i]) {
                result[st.top()] = i;
                st.pop();
            }
            st.push(i);
        }

        return result;
    }

    // 寻找每个位置的前一个更大元素
    vector<int> prev_greater(const vector<T>& arr) {
        vector<int> result(arr.size(), -1);
        stack<int> st;

        for (int i = 0; i < arr.size(); i++) {
            while (!st.empty() && arr[st.top()] <= arr[i]) { st.pop(); }
            if (!st.empty()) { result[i] = st.top(); }
            st.push(i);
        }

        return result;
    }

    // 寻找每个位置的前一个更小元素
    vector<int> prev_smaller(const vector<T>& arr) {
        vector<int> result(arr.size(), -1);
        stack<int> st;

        for (int i = 0; i < arr.size(); i++) {
            while (!st.empty() && arr[st.top()] >= arr[i]) { st.pop(); }
            if (!st.empty()) { result[i] = st.top(); }
            st.push(i);
        }

        return result;
    }
};

// 计算直方图中的最大矩形面积
long long largest_rectangle_histogram(const vector<int>& heights) {
    vector<int> h = heights;
    h.push_back(0);  // 添加哨兵元素

    stack<int> st;  // 单调递增栈
    long long max_area = 0;

    for (int i = 0; i < h.size(); i++) {
        while (!st.empty() && h[st.top()] > h[i]) {
            int height = h[st.top()];  // 当前矩形的高度
            st.pop();
            // 计算宽度：左右边界之间的距离
            int width = st.empty() ? i : i - st.top() - 1;
            max_area = max(max_area, (long long)height * width);
        }
        st.push(i);
    }

    return max_area;
}

// 检查括号字符串是否有效
bool is_valid_parentheses(const string& s) {
    stack<char> st;

    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);  // 左括号入栈
        } else if (c == ')' || c == ']' || c == '}') {
            if (st.empty()) return false;  // 没有匹配的左括号

            char top = st.top();
            st.pop();

            // 检查括号是否匹配
            if ((c == ')' && top != '(') || (c == ']' && top != '[') || (c == '}' && top != '{')) { return false; }
        }
    }

    return st.empty();  // 所有括号都已匹配
}

// 寻找二进制矩阵中矩形的最大面积
long long max_rectangle_in_matrix(const vector<vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return 0;

    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<int> heights(cols, 0);  // 每列的连续1的高度
    long long max_area = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 更新每列的高度
            heights[j] = matrix[i][j] == 0 ? 0 : heights[j] + 1;
        }
        // 计算当前行为底边的最大矩形面积
        max_area = max(max_area, largest_rectangle_histogram(heights));
    }

    return max_area;
}

// 计算所有子数组最小值的和
long long sum_subarray_minimums(const vector<int>& arr) {
    const int MOD = 1e9 + 7;
    int n = arr.size();

    MonotonicStack<int> ms;
    vector<int> prev_smaller = ms.prev_smaller(arr);  // 前一个更小元素的位置
    vector<int> next_smaller = ms.next_smaller(arr);  // 下一个更小元素的位置

    long long result = 0;

    for (int i = 0; i < n; i++) {
        // 计算以arr[i]为最小值的子数组数量
        int left = prev_smaller[i] == -1 ? i + 1 : i - prev_smaller[i];   // 左边界数量
        int right = next_smaller[i] == -1 ? n - i : next_smaller[i] - i;  // 右边界数量

        // 贡献 = 元素值 × 包含该元素的子数组数量
        result = (result + (long long)arr[i] * left * right) % MOD;
    }

    return result;
}
]=]),

}
