-- Auto-generated LuaSnip snippets
local ls = require("luasnip")
local ps = ls.parser.parse_snippet

return {

-- 04_Math\Combinatorics\Basic\FactorialCombination.h
ps("04_math_combinatorics_basic_factorialcombination_h", [=[

/**
 * 阶乘与组合数模板
 * 功能：快速幂、阶乘预处理、组合数计算
 * 时间复杂度：预处理O(n)，查询O(1)
 */

using ll = long long;

// 快速幂模板
struct FastPower {
    ll pow(ll a, ll b, ll mod) {
        ll res = 1;
        a %= mod;
        while (b) {
            if (b & 1) res = res * a % mod;
            a = a * a % mod;
            b >>= 1;
        }
        return res;
    }
};

// 阶乘与组合数预处理
template <int MAXN = 200005>
struct FactorialCombination {
    static constexpr ll MOD = 1e9 + 7;
    ll fac[MAXN], inv_fac[MAXN];
    FastPower fp;

    FactorialCombination() { init(); }

    void init() {
        fac[0] = 1;
        for (int i = 1; i < MAXN; i++) { fac[i] = fac[i - 1] * i % MOD; }
        inv_fac[MAXN - 1] = fp.pow(fac[MAXN - 1], MOD - 2, MOD);
        for (int i = MAXN - 2; i >= 0; i--) { inv_fac[i] = inv_fac[i + 1] * (i + 1) % MOD; }
    }

    ll C(int n, int m) {
        if (n < m || m < 0) return 0;
        return fac[n] * inv_fac[m] % MOD * inv_fac[n - m] % MOD;
    }

    ll A(int n, int m) {
        if (n < m || m < 0) return 0;
        return fac[n] * inv_fac[n - m] % MOD;
    }

    ll factorial(int n) { return fac[n]; }
    ll inv_factorial(int n) { return inv_fac[n]; }
};

// 小范围组合数（杨辉三角）
struct SmallCombination {
    vector<vector<ll>> C;
    int n;
    ll mod;

    SmallCombination(int n, ll mod = 1e9 + 7) : n(n), mod(mod) {
        C.assign(n + 1, vector<ll>(n + 1, 0));
        init();
    }

    void init() {
        for (int i = 0; i <= n; i++) {
            C[i][0] = C[i][i] = 1;
            for (int j = 1; j < i; j++) { C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % mod; }
        }
    }

    ll get(int n, int m) {
        if (n < 0 || m < 0 || n < m) return 0;
        return C[n][m];
    }
};
]=]),

-- 04_Math\Combinatorics\Basic\LucasTheorem.h
ps("04_math_combinatorics_basic_lucastheorem_h", [=[

/**
 * Lucas定理模板
 * 功能：计算C(n,m) mod p，p为质数
 * 时间复杂度：O(p + log_p(n))
 */

using ll = long long;

// Lucas定理
struct Lucas {
    ll p;
    vector<ll> fac, inv_fac;

    Lucas(ll p) : p(p) { init(); }

    ll power(ll a, ll b) {
        ll res = 1;
        a %= p;
        while (b) {
            if (b & 1) res = res * a % p;
            a = a * a % p;
            b >>= 1;
        }
        return res;
    }

    void init() {
        fac.resize(p);
        inv_fac.resize(p);
        fac[0] = 1;
        for (int i = 1; i < p; i++) { fac[i] = fac[i - 1] * i % p; }
        inv_fac[p - 1] = power(fac[p - 1], p - 2);
        for (int i = p - 2; i >= 0; i--) { inv_fac[i] = inv_fac[i + 1] * (i + 1) % p; }
    }

    ll C(ll n, ll m) {
        if (n < m || m < 0) return 0;
        if (n < p && m < p) { return fac[n] * inv_fac[m] % p * inv_fac[n - m] % p; }
        return C(n / p, m / p) * C(n % p, m % p) % p;
    }

    // 计算阶乘的最高次幂
    ll factorial_power(ll n, ll prime) {
        ll res = 0;
        while (n) {
            n /= prime;
            res += n;
        }
        return res;
    }
};

// 使用示例
/*
Lucas lucas(1000000007);
cout << lucas.C(1000000, 500000) << endl;
cout << lucas.factorial_power(10, 2) << endl; // 计算10!中2的幂次
*/
]=]),

-- 04_Math\Combinatorics\Generating_Functions\EGF.h
ps("04_math_combinatorics_generating_functions_egf_h", [=[

// 指数生成函数基础结构
struct EGF {
    static const int MOD = 1e9 + 7;
    static const int MAXN = 100005;
    vector<long long> fact, inv_fact;

    EGF() { precompute_factorials(); }

    void precompute_factorials() {
        fact.resize(MAXN);
        inv_fact.resize(MAXN);
        fact[0] = 1;
        for (int i = 1; i < MAXN; i++) { fact[i] = fact[i - 1] * i % MOD; }
        inv_fact[MAXN - 1] = quick_pow(fact[MAXN - 1], MOD - 2);
        for (int i = MAXN - 2; i >= 0; i--) { inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD; }
    }

    long long quick_pow(long long a, long long b) {
        long long res = 1;
        while (b > 0) {
            if (b & 1) res = res * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return res;
    }

    // EGF基本运算
    vector<long long> add(const vector<long long>& f, const vector<long long>& g) {
        int n = max(f.size(), g.size());
        vector<long long> result(n);
        for (int i = 0; i < n; i++) {
            long long fi = (i < f.size()) ? f[i] : 0;
            long long gi = (i < g.size()) ? g[i] : 0;
            result[i] = (fi + gi) % MOD;
        }
        return result;
    }

    vector<long long> multiply(const vector<long long>& f, const vector<long long>& g) {
        int n = f.size() + g.size() - 1;
        vector<long long> result(n, 0);
        for (int i = 0; i < f.size(); i++) {
            for (int j = 0; j < g.size(); j++) {
                if (i + j < n) {
                    long long term = f[i] * g[j] % MOD;
                    term = term * fact[i + j] % MOD;
                    term = term * inv_fact[i] % MOD;
                    term = term * inv_fact[j] % MOD;
                    result[i + j] = (result[i + j] + term) % MOD;
                }
            }
        }
        return result;
    }
};  // 常用指数生成函数
struct EGFGenerator {
    EGF egf;

    // 指数函数 e^x
    vector<long long> exponential(int n) {
        vector<long long> result(n);
        for (int i = 0; i < n; i++) result[i] = egf.inv_fact[i];
        return result;
    }

    // 三角函数 sin(x)
    vector<long long> sine(int n) {
        vector<long long> result(n, 0);
        long long sign = 1;
        for (int i = 1; i < n; i += 2) {
            result[i] = sign * egf.inv_fact[i] % EGF::MOD;
            if (result[i] < 0) result[i] += EGF::MOD;
            sign = -sign;
        }
        return result;
    }

    // 三角函数 cos(x)
    vector<long long> cosine(int n) {
        vector<long long> result(n, 0);
        long long sign = 1;
        for (int i = 0; i < n; i += 2) {
            result[i] = sign * egf.inv_fact[i] % EGF::MOD;
            if (result[i] < 0) result[i] += EGF::MOD;
            sign = -sign;
        }
        return result;
    }

    // 二项式 (1+x)^r
    vector<long long> binomial(long long r, int n) {
        vector<long long> result(n);
        result[0] = 1;
        long long coeff = 1;
        for (int i = 1; i < n; i++) {
            coeff = coeff * (r - i + 1) % EGF::MOD;
            coeff = coeff * egf.inv_fact[1] % EGF::MOD;
            result[i] = coeff * egf.inv_fact[i] % EGF::MOD;
        }
        return result;
    }

    // 从EGF系数获取原始数列
    vector<long long> get_sequence(const vector<long long>& egf_coeffs) {
        vector<long long> result(egf_coeffs.size());
        for (int i = 0; i < egf_coeffs.size(); i++) { result[i] = egf_coeffs[i] * egf.fact[i] % EGF::MOD; }
        return result;
    }

    // 从数列生成EGF系数
    vector<long long> from_sequence(const vector<long long>& seq) {
        vector<long long> result(seq.size());
        for (int i = 0; i < seq.size(); i++) { result[i] = seq[i] * egf.inv_fact[i] % EGF::MOD; }
        return result;
    }
};
]=]),

-- 04_Math\Combinatorics\Generating_Functions\OGF.h
ps("04_math_combinatorics_generating_functions_ogf_h", [=[

// 普通生成函数基础结构
struct OGF {
    vector<long long> coeffs;
    static const int MOD = 1e9 + 7;

    OGF(int size = 0) : coeffs(size, 0) {}
    OGF(const vector<long long>& c) : coeffs(c) {}

    long long operator[](int n) const { return n < coeffs.size() ? coeffs[n] : 0; }
    long long& operator[](int n) {
        if (n >= coeffs.size()) coeffs.resize(n + 1, 0);
        return coeffs[n];
    }

    OGF operator+(const OGF& other) const {
        int max_size = max(coeffs.size(), other.coeffs.size());
        OGF result(max_size);
        for (int i = 0; i < max_size; i++) { result.coeffs[i] = ((*this)[i] + other[i]) % MOD; }
        return result;
    }

    OGF operator*(const OGF& other) const {
        if (coeffs.empty() || other.coeffs.empty()) return OGF();
        int result_size = coeffs.size() + other.coeffs.size() - 1;
        OGF result(result_size);
        for (int i = 0; i < coeffs.size(); i++) {
            for (int j = 0; j < other.coeffs.size(); j++) {
                result.coeffs[i + j] = (result.coeffs[i + j] + coeffs[i] * other.coeffs[j]) % MOD;
            }
        }
        return result;
    }

    static long long quick_pow(long long base, long long exp) {
        long long result = 1;
        while (exp > 0) {
            if (exp & 1) result = (result * base) % MOD;
            base = (base * base) % MOD;
            exp >>= 1;
        }
        return result;
    }
};  // 常用生成函数生成器
struct OGFGenerator {
    static const int MOD = 1e9 + 7;

    // 几何级数: 1/(1-x) = 1 + x + x^2 + ...
    static OGF geometric_series(int max_degree) {
        OGF result(max_degree + 1);
        for (int i = 0; i <= max_degree; i++) result[i] = 1;
        return result;
    }

    // 斐波那契数列生成函数
    static OGF fibonacci(int max_degree) {
        vector<long long> fib(max_degree + 1, 0);
        if (max_degree >= 0) fib[0] = 0;
        if (max_degree >= 1) fib[1] = 1;
        for (int i = 2; i <= max_degree; i++) { fib[i] = (fib[i - 1] + fib[i - 2]) % MOD; }
        return OGF(fib);
    }

    // 卡特兰数生成函数
    static OGF catalan(int max_degree) {
        vector<long long> cat(max_degree + 1, 0);
        if (max_degree >= 0) cat[0] = 1;
        for (int n = 1; n <= max_degree; n++) {
            for (int i = 0; i < n; i++) { cat[n] = (cat[n] + cat[i] * cat[n - 1 - i]) % MOD; }
        }
        return OGF(cat);
    }

    // 二项式 (1+x)^n
    static OGF binomial(int n, int max_degree) {
        OGF result(min(n + 1, max_degree + 1));
        long long c = 1;
        for (int k = 0; k <= min(n, max_degree); k++) {
            result[k] = c;
            if (k < n) {
                c = c * (n - k) % MOD;
                c = c * OGF::quick_pow(k + 1, MOD - 2) % MOD;
            }
        }
        return result;
    }
};
]=]),

-- 04_Math\Combinatorics\Inclusion_Exclusion\InclusionExclusion.h
ps("04_math_combinatorics_inclusion_exclusion_inclusionexclusion_h", [=[

// 容斥原理基础结构
struct InclusionExclusion {
    static const int MOD = 1e9 + 7;

    // 错位排列：容斥原理经典应用
    long long derangement(int n) {
        if (n == 0) return 1;
        if (n == 1) return 0;

        vector<long long> fact(n + 1);
        fact[0] = 1;
        for (int i = 1; i <= n; i++) { fact[i] = fact[i - 1] * i % MOD; }

        long long result = 0;
        long long sign = 1;
        for (int i = 0; i <= n; i++) {
            result = (result + sign * fact[n - i]) % MOD;
            sign = -sign;
            if (result < 0) result += MOD;
        }
        return result;
    }

    // Möbius函数容斥：计算与给定数互质的数
    long long coprime_count(long long n, const vector<int>& primes) {
        long long result = 0;
        int k = primes.size();

        for (int mask = 0; mask < (1 << k); mask++) {
            long long product = 1;
            int bit_count = 0;

            for (int i = 0; i < k; i++) {
                if (mask & (1 << i)) {
                    product *= primes[i];
                    bit_count++;
                }
            }

            long long contribution = n / product;
            if (bit_count & 1) {
                result -= contribution;
            } else {
                result += contribution;
            }
        }
        return result;
    }

    static long long quick_pow(long long a, long long b) {
        long long res = 1;
        while (b > 0) {
            if (b & 1) res = res * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return res;
    }
};  // Min-Max容斥结构
struct MinMaxInclusionExclusion {
    // max(S) = Σ(-1)^(|T|-1) * min(T), T⊆S, T≠∅
    template <typename T>
    static T compute_max(const vector<T>& values) {
        int n = values.size();
        T result = 0;

        for (int mask = 1; mask < (1 << n); mask++) {
            T min_val = numeric_limits<T>::max();
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) { min_val = min(min_val, values[i]); }
            }

            int bit_count = __builtin_popcount(mask);
            if ((bit_count - 1) & 1) {
                result -= min_val;
            } else {
                result += min_val;
            }
        }
        return result;
    }

    // min(S) = Σ(-1)^(|T|-1) * max(T), T⊆S, T≠∅
    template <typename T>
    static T compute_min(const vector<T>& values) {
        int n = values.size();
        T result = 0;

        for (int mask = 1; mask < (1 << n); mask++) {
            T max_val = numeric_limits<T>::min();
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) { max_val = max(max_val, values[i]); }
            }

            int bit_count = __builtin_popcount(mask);
            if ((bit_count - 1) & 1) {
                result -= max_val;
            } else {
                result += max_val;
            }
        }
        return result;
    }
};

// 容斥原理应用示例
struct InclusionExclusionApplications {
    static const int MOD = 1e9 + 7;

    // 计算满足条件的排列数
    static long long valid_permutations(int n, const vector<pair<int, int>>& forbidden) {
        long long total = 1;
        for (int i = 1; i <= n; i++) total = total * i % MOD;

        long long forbidden_count = 0;
        int k = forbidden.size();

        for (int mask = 1; mask < (1 << k); mask++) {
            long long remaining = n;
            for (int i = 0; i < k; i++) {
                if (mask & (1 << i)) remaining--;
            }

            long long ways = 1;
            for (int i = 1; i <= remaining; i++) { ways = ways * i % MOD; }

            int bit_count = __builtin_popcount(mask);
            if (bit_count & 1) {
                forbidden_count = (forbidden_count + ways) % MOD;
            } else {
                forbidden_count = (forbidden_count - ways + MOD) % MOD;
            }
        }

        return (total - forbidden_count + MOD) % MOD;
    }
};
]=]),

-- 04_Math\Game_Theory\MiniMax.h
ps("04_math_game_theory_minimax_h", [=[

// 基础极小极大算法
template <typename GameState, typename Move>
struct MiniMax {
    static const int INF = 1e9;

    struct GameResult {
        int score;
        Move best_move;
        bool has_move;

        GameResult(int s = 0, Move m = Move(), bool h = false) : score(s), best_move(m), has_move(h) {}
    };

    // 标准MiniMax算法
    static GameResult minimax(const GameState& state,
                              int depth,
                              bool maximizing,
                              function<bool(const GameState&)> is_terminal,
                              function<int(const GameState&)> evaluate,
                              function<vector<pair<Move, GameState>>(const GameState&)> get_moves) {
        if (depth == 0 || is_terminal(state)) { return GameResult(evaluate(state)); }

        vector<pair<Move, GameState>> moves = get_moves(state);
        if (moves.empty()) return GameResult(evaluate(state));

        GameResult result;
        result.has_move = true;
        result.best_move = moves[0].first;

        if (maximizing) {
            result.score = -INF;
            for (const auto& [move, next_state] : moves) {
                GameResult child = minimax(next_state, depth - 1, false, is_terminal, evaluate, get_moves);
                if (child.score > result.score) {
                    result.score = child.score;
                    result.best_move = move;
                }
            }
        } else {
            result.score = INF;
            for (const auto& [move, next_state] : moves) {
                GameResult child = minimax(next_state, depth - 1, true, is_terminal, evaluate, get_moves);
                if (child.score < result.score) {
                    result.score = child.score;
                    result.best_move = move;
                }
            }
        }
        return result;
    }
};  // Alpha-Beta剪枝算法
template <typename GameState, typename Move>
struct AlphaBeta {
    static const int INF = 1e9;

    struct GameResult {
        int score;
        Move best_move;
        bool has_move;

        GameResult(int s = 0, Move m = Move(), bool h = false) : score(s), best_move(m), has_move(h) {}
    };

    // Alpha-Beta剪枝
    static GameResult alpha_beta(const GameState& state,
                                 int depth,
                                 int alpha,
                                 int beta,
                                 bool maximizing,
                                 function<bool(const GameState&)> is_terminal,
                                 function<int(const GameState&)> evaluate,
                                 function<vector<pair<Move, GameState>>(const GameState&)> get_moves) {
        if (depth == 0 || is_terminal(state)) { return GameResult(evaluate(state)); }

        vector<pair<Move, GameState>> moves = get_moves(state);
        if (moves.empty()) return GameResult(evaluate(state));

        GameResult result;
        result.has_move = true;
        result.best_move = moves[0].first;

        if (maximizing) {
            result.score = -INF;
            for (const auto& [move, next_state] : moves) {
                GameResult child =
                    alpha_beta(next_state, depth - 1, alpha, beta, false, is_terminal, evaluate, get_moves);
                if (child.score > result.score) {
                    result.score = child.score;
                    result.best_move = move;
                }
                alpha = max(alpha, child.score);
                if (beta <= alpha) break;  // Beta剪枝
            }
        } else {
            result.score = INF;
            for (const auto& [move, next_state] : moves) {
                GameResult child =
                    alpha_beta(next_state, depth - 1, alpha, beta, true, is_terminal, evaluate, get_moves);
                if (child.score < result.score) {
                    result.score = child.score;
                    result.best_move = move;
                }
                beta = min(beta, child.score);
                if (beta <= alpha) break;  // Alpha剪枝
            }
        }
        return result;
    }

    // 迭代加深搜索
    static GameResult iterative_deepening(const GameState& state,
                                          int max_depth,
                                          bool maximizing,
                                          function<bool(const GameState&)> is_terminal,
                                          function<int(const GameState&)> evaluate,
                                          function<vector<pair<Move, GameState>>(const GameState&)> get_moves) {
        GameResult best_result;
        for (int depth = 1; depth <= max_depth; depth++) {
            GameResult current = alpha_beta(state, depth, -INF, INF, maximizing, is_terminal, evaluate, get_moves);
            if (current.has_move) best_result = current;
            if (abs(current.score) >= INF / 2) break;  // 找到必胜/必败
        }
        return best_result;
    }
};

// 井字棋示例应用
struct TicTacToe {
    struct State {
        vector<vector<char>> board;
        char player;
        State() : board(3, vector<char>(3, ' ')), player('X') {}
    };

    struct Move {
        int row, col;
        Move(int r = -1, int c = -1) : row(r), col(c) {}
        bool operator==(const Move& other) const { return row == other.row && col == other.col; }
    };

    static bool is_terminal(const State& state) { return get_winner(state) != ' ' || is_full(state); }

    static int evaluate(const State& state) {
        char winner = get_winner(state);
        if (winner == 'X') return 1;
        if (winner == 'O') return -1;
        return 0;
    }

    static vector<pair<Move, State>> get_moves(const State& state) {
        vector<pair<Move, State>> moves;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state.board[i][j] == ' ') {
                    State next = state;
                    next.board[i][j] = state.player;
                    next.player = (state.player == 'X') ? 'O' : 'X';
                    moves.push_back({Move(i, j), next});
                }
            }
        }
        return moves;
    }

    static Move get_best_move(const State& state) {
        bool maximizing = (state.player == 'X');
        auto result = AlphaBeta<State, Move>::alpha_beta(state,
                                                         9,
                                                         -AlphaBeta<State, Move>::INF,
                                                         AlphaBeta<State, Move>::INF,
                                                         maximizing,
                                                         is_terminal,
                                                         evaluate,
                                                         get_moves);
        return result.best_move;
    }

   private:
    static char get_winner(const State& state) {
        const auto& board = state.board;
        // 检查行、列、对角线
        for (int i = 0; i < 3; i++) {
            if (board[i][0] != ' ' && board[i][0] == board[i][1] && board[i][1] == board[i][2]) return board[i][0];
            if (board[0][i] != ' ' && board[0][i] == board[1][i] && board[1][i] == board[2][i]) return board[0][i];
        }
        if (board[0][0] != ' ' && board[0][0] == board[1][1] && board[1][1] == board[2][2]) return board[0][0];
        if (board[0][2] != ' ' && board[0][2] == board[1][1] && board[1][1] == board[2][0]) return board[0][2];
        return ' ';
    }

    static bool is_full(const State& state) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state.board[i][j] == ' ') return false;
            }
        }
        return true;
    }
};
]=]),

-- 04_Math\Game_Theory\Nim.h
ps("04_math_game_theory_nim_h", [=[

// 基础Nim游戏
struct Nim {
    // 基础Nim游戏判断
    static bool is_winning(const vector<int>& piles) {
        int nim_sum = 0;
        for (int pile : piles) nim_sum ^= pile;
        return nim_sum != 0;
    }
    
    // 计算Nim和
    static int nim_sum(const vector<int>& piles) {
        int result = 0;
        for (int pile : piles) result ^= pile;
        return result;
    }
    
    // 找到获胜的移动
    static pair<int, int> find_winning_move(const vector<int>& piles) {
        int current_nim = nim_sum(piles);
        if (current_nim == 0) return {-1, -1}; // 已经是必败态
        
        for (int i = 0; i < piles.size(); i++) {
            int target = piles[i] ^ current_nim;
            if (target < piles[i]) {
                return {i, piles[i] - target}; // 从第i堆取走指定数量
            }
        }
        return {-1, -1};
    }
};

// Nim游戏变种
struct NimVariants {
    // 反Nim游戏（最后取完者败）
    static bool is_winning_reverse_nim(const vector<int>& piles) {
        int nim_sum_val = Nim::nim_sum(piles);
        bool all_one = true;
        for (int pile : piles) {
            if (pile > 1) {
                all_one = false;
                break;
            }
        }
        if (all_one) return piles.size() % 2 == 0;
        return nim_sum_val != 0;
    }
    
    // k-Nim游戏（每次最多取k个）
    static bool is_winning_k_nim(const vector<int>& piles, int k) {
        int result = 0;
        for (int pile : piles) result ^= (pile % (k + 1));
        return result != 0;
    }
    
    // Wythoff游戏
    static bool is_winning_wythoff(int a, int b) {
        if (a > b) swap(a, b);
        int diff = b - a;
        double golden_ratio = (1.0 + sqrt(5.0)) / 2.0;
        return (int)(diff * golden_ratio) != a;
    }
    
    // 阶梯游戏
    static int staircase_nim_sum(const vector<int>& steps) {
        int result = 0;
        for (int i = 0; i < steps.size(); i += 2) {
            result ^= steps[i];
        }
        return result;
    }
    
    // Fibonacci Nim
    static bool is_winning_fibonacci_nim(int n) {
        if (n <= 2) return n != 0;
        
        vector<int> fib = {1, 2};
        while (fib.back() <= n) {
            fib.push_back(fib[fib.size() - 1] + fib[fib.size() - 2]);
        }
        
        vector<int> sg(n + 1, 0);
        for (int i = 1; i <= n; i++) {
            set<int> reachable;
            for (int f : fib) {
                if (f > i) break;
                reachable.insert(sg[i - f]);
            }
            
            int mex = 0;
            while (reachable.count(mex)) mex++;
            sg[i] = mex;
        }
        return sg[n] != 0;
    }
};

// 多种取法的Nim游戏
struct MultiNim {
    // 计算单堆的Grundy数
    static int calculate_grundy_single_pile(int n, const vector<int>& moves) {
        vector<int> grundy(n + 1, 0);
        
        for (int i = 1; i <= n; i++) {
            set<int> reachable;
            for (int move : moves) {
                if (i >= move) reachable.insert(grundy[i - move]);
            }
            
            int mex = 0;
            while (reachable.count(mex)) mex++;
            grundy[i] = mex;
        }
        return grundy[n];
    }
    
    // Multi-Nim（有多种取法的Nim）
    static bool is_winning_multi_nim(const vector<int>& piles, const vector<int>& allowed_moves) {
        int result = 0;
        for (int pile : piles) {
            result ^= calculate_grundy_single_pile(pile, allowed_moves);
        }
        return result != 0;
    }
    
    // 获取获胜移动
    static tuple<int, int, int> find_winning_move(const vector<int>& piles, const vector<int>& allowed_moves) {
        vector<int> grundy_values;
        for (int pile : piles) {
            grundy_values.push_back(calculate_grundy_single_pile(pile, allowed_moves));
        }
        
        int xor_sum = 0;
        for (int g : grundy_values) xor_sum ^= g;
        
        if (xor_sum == 0) return {-1, -1, -1}; // 必败态
        
        for (int i = 0; i < piles.size(); i++) {
            int target_grundy = grundy_values[i] ^ xor_sum;
            
            // 找到能使该堆达到目标Grundy值的移动
            for (int move : allowed_moves) {
                if (piles[i] >= move) {
                    int new_pile = piles[i] - move;
                    if (calculate_grundy_single_pile(new_pile, allowed_moves) == target_grundy) {
                        return {i, move, new_pile};
                    }
                }
            }
        }
        return {-1, -1, -1};
    }
};
]=]),

-- 04_Math\Game_Theory\SG.h
ps("04_math_game_theory_sg_h", [=[

// 基础SG函数
struct SG {
    // 计算MEX（最小排斥数）
    static int calculate_mex(const set<int>& s) {
        int mex = 0;
        while (s.count(mex)) mex++;
        return mex;
    }
    
    // 动态规划计算SG函数值
    static vector<int> calculate_sg_dp(int max_state, const function<vector<int>(int)>& get_next_states) {
        vector<int> sg(max_state + 1, 0);
        
        for (int state = 0; state <= max_state; state++) {
            set<int> reachable;
            vector<int> next_states = get_next_states(state);
            
            for (int next : next_states) {
                if (next >= 0 && next <= max_state) reachable.insert(sg[next]);
            }
            sg[state] = calculate_mex(reachable);
        }
        return sg;
    }
    
    // 记忆化搜索计算SG函数值
    static int calculate_sg_memo(int state, const function<vector<int>(int)>& get_next_states, 
                               unordered_map<int, int>& memo) {
        if (memo.count(state)) return memo[state];
        
        set<int> reachable;
        vector<int> next_states = get_next_states(state);
        
        for (int next : next_states) {
            reachable.insert(calculate_sg_memo(next, get_next_states, memo));
        }
        return memo[state] = calculate_mex(reachable);
    }
    
    // 组合游戏的SG值（多个独立子游戏）
    static int combined_sg(const vector<int>& sg_values) {
        int result = 0;
        for (int sg : sg_values) result ^= sg;
        return result;
    }
    
    // 判断当前位置是否为胜利位置
    static bool is_winning_position(const vector<int>& sg_values) {
        return combined_sg(sg_values) != 0;
    }
};

// 经典SG游戏
struct SGGames {
    // Subtraction游戏的SG函数
    static vector<int> subtraction_game_sg(const vector<int>& subtraction_set, int max_n) {
        vector<int> sg(max_n + 1, 0);
        
        for (int i = 1; i <= max_n; i++) {
            set<int> reachable;
            for (int sub : subtraction_set) {
                if (i >= sub) reachable.insert(sg[i - sub]);
            }
            sg[i] = SG::calculate_mex(reachable);
        }
        return sg;
    }
    
    // Dawson's Chess游戏
    static vector<int> dawson_chess_sg(int max_n) {
        vector<int> sg(max_n + 1, 0);
        
        for (int i = 2; i <= max_n; i++) {
            set<int> reachable;
            
            // 可以在任意位置放置一个棋子，将区间分割
            for (int j = 0; j < i; j++) {
                int left = j;
                int right = i - j - 1;
                reachable.insert(sg[left] ^ sg[right]);
            }
            sg[i] = SG::calculate_mex(reachable);
        }
        return sg;
    }
    
    // Kayles游戏的SG函数
    static vector<int> kayles_sg(int max_n) {
        vector<int> sg(max_n + 1, 0);
        
        for (int i = 1; i <= max_n; i++) {
            set<int> reachable;
            
            // 移除一个pin
            for (int j = 0; j < i; j++) {
                int left = j;
                int right = i - j - 1;
                reachable.insert(sg[left] ^ sg[right]);
            }
            
            // 移除相邻的两个pins
            for (int j = 0; j < i - 1; j++) {
                int left = j;
                int right = i - j - 2;
                reachable.insert(sg[left] ^ sg[right]);
            }
            sg[i] = SG::calculate_mex(reachable);
        }
        return sg;
    }
    
    // Green Hackenbush的SG函数（树上博弈）
    static int green_hackenbush_sg(const vector<vector<int>>& tree, int root) {
        function<int(int, int)> dfs = [&](int u, int parent) -> int {
            int sg = 0;
            for (int v : tree[u]) {
                if (v != parent) sg ^= (dfs(v, u) + 1);
            }
            return sg;
        };
        return dfs(root, -1);
    }
    
    // 翻硬币游戏
    static int coin_flipping_sg(const vector<bool>& coins) {
        int n = coins.size();
        int sg = 0;
        
        // 找出所有正面朝上的硬币位置
        vector<int> heads;
        for (int i = 0; i < n; i++) {
            if (coins[i]) heads.push_back(i);
        }
        
        // 计算相邻正面硬币之间的距离
        for (int i = 0; i < heads.size(); i += 2) {
            if (i + 1 < heads.size()) {
                int distance = heads[i + 1] - heads[i] - 1;
                sg ^= distance;
            }
        }
        return sg;
    }
};

// 两堆石子游戏
struct TwoPileGame {
    map<pair<int, int>, int> memo;
    function<vector<pair<int, int>>(int, int)> get_moves;
    
    TwoPileGame(const function<vector<pair<int, int>>(int, int)>& moves) : get_moves(moves) {}
    
    int calculate_sg(int a, int b) {
        if (a > b) swap(a, b);
        pair<int, int> state = {a, b};
        
        if (memo.count(state)) return memo[state];
        
        set<int> reachable;
        vector<pair<int, int>> next_states = get_moves(a, b);
        
        for (auto [na, nb] : next_states) {
            if (na >= 0 && nb >= 0) reachable.insert(calculate_sg(na, nb));
        }
        return memo[state] = SG::calculate_mex(reachable);
    }
    
    // Wythoff游戏的移动
    static vector<pair<int, int>> wythoff_moves(int a, int b) {
        vector<pair<int, int>> moves;
        
        // 从第一堆取任意个
        for (int i = 1; i <= a; i++) moves.push_back({a - i, b});
        
        // 从第二堆取任意个
        for (int i = 1; i <= b; i++) moves.push_back({a, b - i});
        
        // 从两堆取相同个数
        for (int i = 1; i <= min(a, b); i++) moves.push_back({a - i, b - i});
        
        return moves;
    }
};
]=]),

-- 04_Math\Linear_Algebra\GaussianElimination.h
ps("04_math_linear_algebra_gaussianelimination_h", [=[

using ll = long long;

// 实数高斯消元
struct RealGauss {
    static constexpr double EPS = 1e-9;
    
    // 求解线性方程组 Ax = b，返回值：0-无解，1-唯一解，2-无穷多解
    int solve(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
        int n = A.size(), m = A[0].size();
        vector<vector<double>> aug(n, vector<double>(m + 1));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) aug[i][j] = A[i][j];
            aug[i][m] = b[i];
        }
        
        vector<int> pivot_col(n, -1);
        int rank = 0;
        
        for (int col = 0; col < m && rank < n; col++) {
            int pivot_row = -1;
            for (int row = rank; row < n; row++) {
                if (abs(aug[row][col]) > EPS) {
                    if (pivot_row == -1 || abs(aug[row][col]) > abs(aug[pivot_row][col])) 
                        pivot_row = row;
                }
            }
            
            if (pivot_row == -1) continue;
            if (pivot_row != rank) swap(aug[rank], aug[pivot_row]);
            
            pivot_col[rank] = col;
            for (int row = 0; row < n; row++) {
                if (row != rank && abs(aug[row][col]) > EPS) {
                    double factor = aug[row][col] / aug[rank][col];
                    for (int j = col; j <= m; j++) aug[row][j] -= factor * aug[rank][j];
                }
            }
            rank++;
        }
        
        for (int i = rank; i < n; i++) {
            if (abs(aug[i][m]) > EPS) return 0;
        }
        
        x.assign(m, 0);
        for (int i = rank - 1; i >= 0; i--) {
            if (pivot_col[i] != -1) x[pivot_col[i]] = aug[i][m] / aug[i][pivot_col[i]];
        }
        
        return rank == m ? 1 : 2;
    }
    
    // 计算矩阵的秩
    int rank(vector<vector<double>>& A) {
        int n = A.size(), m = A[0].size();
        vector<vector<double>> mat = A;
        int rank = 0;
        
        for (int col = 0; col < m && rank < n; col++) {
            int pivot_row = -1;
            for (int row = rank; row < n; row++) {
                if (abs(mat[row][col]) > EPS) {
                    if (pivot_row == -1 || abs(mat[row][col]) > abs(mat[pivot_row][col]))
                        pivot_row = row;
                }
            }
            
            if (pivot_row == -1) continue;
            if (pivot_row != rank) swap(mat[rank], mat[pivot_row]);
            
            for (int row = rank + 1; row < n; row++) {
                if (abs(mat[row][col]) > EPS) {
                    double factor = mat[row][col] / mat[rank][col];
                    for (int j = col; j < m; j++) mat[row][j] -= factor * mat[rank][j];
                }
            }
            rank++;
        }
        
        return rank;
    }
};

// 模运算高斯消元
struct ModGauss {
    static constexpr ll MOD = 1e9 + 7;
    
    ll power(ll a, ll b) {
        ll res = 1;
        while (b > 0) {
            if (b & 1) res = res * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return res;
    }
    
    ll inv(ll a) { return power(a, MOD - 2); }
    
    // 求解模运算线性方程组
    int solve(vector<vector<ll>>& A, vector<ll>& b, vector<ll>& x) {
        int n = A.size(), m = A[0].size();
        vector<vector<ll>> aug(n, vector<ll>(m + 1));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) aug[i][j] = ((A[i][j] % MOD) + MOD) % MOD;
            aug[i][m] = ((b[i] % MOD) + MOD) % MOD;
        }
        
        vector<int> pivot_col(n, -1);
        int rank = 0;
        
        for (int col = 0; col < m && rank < n; col++) {
            int pivot_row = -1;
            for (int row = rank; row < n; row++) {
                if (aug[row][col] != 0) {
                    pivot_row = row;
                    break;
                }
            }
            
            if (pivot_row == -1) continue;
            if (pivot_row != rank) swap(aug[rank], aug[pivot_row]);
            
            pivot_col[rank] = col;
            ll inv_pivot = inv(aug[rank][col]);
            for (int row = 0; row < n; row++) {
                if (row != rank && aug[row][col] != 0) {
                    ll factor = aug[row][col] * inv_pivot % MOD;
                    for (int j = col; j <= m; j++) {
                        aug[row][j] = (aug[row][j] - factor * aug[rank][j] % MOD + MOD) % MOD;
                    }
                }
            }
            rank++;
        }
        
        for (int i = rank; i < n; i++) {
            if (aug[i][m] != 0) return 0;
        }
        
        x.assign(m, 0);
        for (int i = rank - 1; i >= 0; i--) {
            if (pivot_col[i] != -1) x[pivot_col[i]] = aug[i][m] * inv(aug[i][pivot_col[i]]) % MOD;
        }
        
        return rank == m ? 1 : 2;
    }
};

// 矩阵求逆
struct MatrixInverse {
    static constexpr double EPS = 1e-9;
    
    // 求矩阵的逆
    bool inverse(vector<vector<double>>& A, vector<vector<double>>& inv_A) {
        int n = A.size();
        if (A[0].size() != n) return false;
        
        vector<vector<double>> aug(n, vector<double>(2 * n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = A[i][j];
                aug[i][j + n] = (i == j) ? 1 : 0;
            }
        }
        
        for (int i = 0; i < n; i++) {
            int pivot = -1;
            for (int j = i; j < n; j++) {
                if (abs(aug[j][i]) > EPS) {
                    if (pivot == -1 || abs(aug[j][i]) > abs(aug[pivot][i])) pivot = j;
                }
            }
            
            if (pivot == -1) return false;
            if (pivot != i) swap(aug[i], aug[pivot]);
            
            double pivot_val = aug[i][i];
            for (int j = 0; j < 2 * n; j++) aug[i][j] /= pivot_val;
            
            for (int j = 0; j < n; j++) {
                if (j != i && abs(aug[j][i]) > EPS) {
                    double factor = aug[j][i];
                    for (int k = 0; k < 2 * n; k++) aug[j][k] -= factor * aug[i][k];
                }
            }
        }
        
        inv_A.assign(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) inv_A[i][j] = aug[i][j + n];
        }
        
        return true;
    }
};

]=]),

-- 04_Math\Linear_Algebra\LinearSystem2D.h
ps("04_math_linear_algebra_linearsystem2d_h", [=[

using ll = long long;

// 二元一次方程组求解器
struct LinearSystem2D {
    // 求解二元一次方程组
    // a1*x + b1*y = c1
    // a2*x + b2*y = c2
    struct Solution {
        bool has_solution;
        bool unique;
        double x, y;  // 唯一解的情况
        // 对于无穷解的情况，可以用参数方程表示
    };

    Solution solve(double a1, double b1, double c1, double a2, double b2, double c2) {
        Solution result;

        // 计算行列式
        double det = a1 * b2 - a2 * b1;

        if (abs(det) < 1e-9) {  // 行列式为0
            // 检查是否有解
            if (abs(a1 * c2 - a2 * c1) < 1e-9 && abs(b1 * c2 - b2 * c1) < 1e-9) {
                result.has_solution = true;
                result.unique = false;  // 无穷多解
            } else {
                result.has_solution = false;  // 无解
            }
        } else {  // 有唯一解
            result.has_solution = true;
            result.unique = true;
            result.x = (c1 * b2 - c2 * b1) / det;
            result.y = (a1 * c2 - a2 * c1) / det;
        }

        return result;
    }

    // 整数版本（使用分数避免精度问题）
    struct Fraction {
        ll num, den;
        Fraction(ll n = 0, ll d = 1) : num(n), den(d) {
            if (den < 0) {
                num = -num;
                den = -den;
            }
            ll g = gcd(abs(num), abs(den));
            num /= g;
            den /= g;
        }

        Fraction operator+(const Fraction& other) const {
            return Fraction(num * other.den + other.num * den, den * other.den);
        }

        Fraction operator-(const Fraction& other) const {
            return Fraction(num * other.den - other.num * den, den * other.den);
        }

        Fraction operator*(const Fraction& other) const { return Fraction(num * other.num, den * other.den); }

        Fraction operator/(const Fraction& other) const { return Fraction(num * other.den, den * other.num); }

        bool is_zero() const { return num == 0; }

        double to_double() const { return (double)num / den; }
    };

    struct IntSolution {
        bool has_solution;
        bool unique;
        Fraction x, y;
    };

    IntSolution solve_int(ll a1, ll b1, ll c1, ll a2, ll b2, ll c2) {
        IntSolution result;

        // 计算行列式
        ll det = a1 * b2 - a2 * b1;

        if (det == 0) {
            // 检查是否有解
            if (a1 * c2 == a2 * c1 && b1 * c2 == b2 * c1) {
                result.has_solution = true;
                result.unique = false;
            } else {
                result.has_solution = false;
            }
        } else {
            result.has_solution = true;
            result.unique = true;
            result.x = Fraction(c1 * b2 - c2 * b1, det);
            result.y = Fraction(a1 * c2 - a2 * c1, det);
        }

        return result;
    }
};

// 二元一次方程组整数解求解器
struct LinearSystem2DInt {
    struct IntSolution {
        bool has_solution;
        bool unique;
        ll x, y;  // 整数解
    };

    IntSolution solve_int_only(ll a1, ll b1, ll c1, ll a2, ll b2, ll c2) {
        IntSolution result;
        result.has_solution = false;
        result.unique = false;

        // 计算行列式
        ll det = a1 * b2 - a2 * b1;

        if (det == 0) {
            // 系数矩阵奇异，需要特殊处理
            // 这里先简单返回无解
            return result;
        }

        // 计算解的分子
        ll x_num = c1 * b2 - c2 * b1;
        ll y_num = a1 * c2 - a2 * c1;

        // 检查是否能整除（即解是否为整数）
        if (x_num % det == 0 && y_num % det == 0) {
            result.has_solution = true;
            result.unique = true;
            result.x = x_num / det;
            result.y = y_num / det;
        }

        return result;
    }
};
]=]),

-- 04_Math\Linear_Algebra\Matrix.h
ps("04_math_linear_algebra_matrix_h", [=[

using ll = long long;

// 基础矩阵运算
struct Matrix {
    static constexpr ll MOD = 1e9 + 7;
    int n, m;
    vector<vector<ll>> a;
    
    Matrix(int r = 0, int c = 0) : n(r), m(c), a(r, vector<ll>(c, 0)) {}
    Matrix(vector<vector<ll>>& mat) : n(mat.size()), m(mat.size() ? mat[0].size() : 0), a(mat) {}
    
    vector<ll>& operator[](int i) { return a[i]; }
    const vector<ll>& operator[](int i) const { return a[i]; }
    
    Matrix operator+(const Matrix& b) const {
        Matrix res(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res[i][j] = (a[i][j] + b[i][j]) % MOD;
        }
        return res;
    }
    
    Matrix operator-(const Matrix& b) const {
        Matrix res(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res[i][j] = (a[i][j] - b[i][j] + MOD) % MOD;
        }
        return res;
    }
    
    Matrix operator*(const Matrix& b) const {
        Matrix res(n, b.m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < b.m; j++) {
                for (int k = 0; k < m; k++) res[i][j] = (res[i][j] + a[i][k] * b[k][j]) % MOD;
            }
        }
        return res;
    }
    
    Matrix power(ll k) const {
        Matrix res = identity(n), base = *this;
        while (k > 0) {
            if (k & 1) res = res * base;
            base = base * base;
            k >>= 1;
        }
        return res;
    }
    
    static Matrix identity(int sz) {
        Matrix res(sz, sz);
        for (int i = 0; i < sz; i++) res[i][i] = 1;
        return res;
    }
    
    Matrix transpose() const {
        Matrix res(m, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res[j][i] = a[i][j];
        }
        return res;
    }
};

// 矩阵行列式和逆
struct MatrixDet {
    static constexpr ll MOD = 1e9 + 7;
    
    ll power(ll a, ll b) {
        ll res = 1;
        while (b > 0) {
            if (b & 1) res = res * a % MOD;
            a = a * a % MOD;
            b >>= 1;
        }
        return res;
    }
    
    ll inv(ll a) { return power(a, MOD - 2); }
    
    // 计算行列式
    ll determinant(vector<vector<ll>> a) {
        int n = a.size();
        ll det = 1;
        
        for (int i = 0; i < n; i++) {
            int pivot = -1;
            for (int j = i; j < n; j++) {
                if (a[j][i] != 0) {
                    pivot = j;
                    break;
                }
            }
            
            if (pivot == -1) return 0;
            
            if (pivot != i) {
                swap(a[i], a[pivot]);
                det = (MOD - det) % MOD;
            }
            
            det = det * a[i][i] % MOD;
            ll inv_pivot = inv(a[i][i]);
            
            for (int j = i + 1; j < n; j++) {
                if (a[j][i] != 0) {
                    ll factor = a[j][i] * inv_pivot % MOD;
                    for (int k = i; k < n; k++) {
                        a[j][k] = (a[j][k] - factor * a[i][k] % MOD + MOD) % MOD;
                    }
                }
            }
        }
        
        return det;
    }
    
    // 计算矩阵的秩
    int rank(vector<vector<ll>> a) {
        int n = a.size(), m = a[0].size();
        int rank = 0;
        
        for (int col = 0; col < m && rank < n; col++) {
            int pivot = -1;
            for (int row = rank; row < n; row++) {
                if (a[row][col] != 0) {
                    pivot = row;
                    break;
                }
            }
            
            if (pivot == -1) continue;
            if (pivot != rank) swap(a[rank], a[pivot]);
            
            ll inv_pivot = inv(a[rank][col]);
            for (int row = rank + 1; row < n; row++) {
                if (a[row][col] != 0) {
                    ll factor = a[row][col] * inv_pivot % MOD;
                    for (int j = col; j < m; j++) {
                        a[row][j] = (a[row][j] - factor * a[rank][j] % MOD + MOD) % MOD;
                    }
                }
            }
            rank++;
        }
        
        return rank;
    }
};

// 矩阵特殊操作
struct MatrixOps {
    // 检查对称矩阵
    bool isSymmetric(const vector<vector<ll>>& a) {
        int n = a.size();
        if (a[0].size() != n) return false;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (a[i][j] != a[j][i]) return false;
            }
        }
        return true;
    }
    
    // 检查上三角矩阵
    bool isUpperTriangular(const vector<vector<ll>>& a) {
        int n = a.size();
        if (a[0].size() != n) return false;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (a[i][j] != 0) return false;
            }
        }
        return true;
    }
    
    // 矩阵的迹
    ll trace(const vector<vector<ll>>& a) {
        ll res = 0;
        int n = a.size();
        for (int i = 0; i < n; i++) res = (res + a[i][i]) % Matrix::MOD;
        return res;
    }
    
    // 生成随机矩阵
    Matrix random(int n, int m, ll maxVal = 100) {
        Matrix res(n, m);
        srand(time(nullptr));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res[i][j] = rand() % maxVal;
        }
        return res;
    }
};


]=]),

-- 04_Math\Number_Theory\Advanced\MobiusFunction.h
ps("04_math_number_theory_advanced_mobiusfunction_h", [=[

// 莫比乌斯函数与相关应用
template <typename T = int>
struct MobiusFunction {
    vector<T> mu, primes;
    vector<bool> is_prime;
    vector<T> phi;  // 欧拉函数
    int n;

    MobiusFunction(int _n) : n(_n) {
        mu.resize(n + 1);
        phi.resize(n + 1);
        is_prime.resize(n + 1, true);
        calculate_mobius();
    }

    // 线性筛计算莫比乌斯函数
    void calculate_mobius() {
        mu[1] = phi[1] = 1;
        is_prime[0] = is_prime[1] = false;

        for (int i = 2; i <= n; i++) {
            if (is_prime[i]) {
                primes.push_back(i);
                mu[i] = -1;
                phi[i] = i - 1;
            }

            for (int j = 0; j < primes.size() && i * primes[j] <= n; j++) {
                int next = i * primes[j];
                is_prime[next] = false;

                if (i % primes[j] == 0) {
                    mu[next] = 0;
                    phi[next] = phi[i] * primes[j];
                    break;
                } else {
                    mu[next] = -mu[i];
                    phi[next] = phi[i] * (primes[j] - 1);
                }
            }
        }
    }

    // 获取莫比乌斯函数值
    T get_mu(int x) { return x <= n ? mu[x] : calculate_single_mu(x); }

    // 计算单个数的莫比乌斯函数值
    T calculate_single_mu(long long x) {
        if (x == 1) return 1;

        int cnt = 0;
        for (long long i = 2; i * i <= x; i++) {
            if (x % i == 0) {
                cnt++;
                x /= i;
                if (x % i == 0) return 0;  // 有平方因子
            }
        }
        if (x > 1) cnt++;

        return (cnt & 1) ? -1 : 1;
    }

    // 莫比乌斯函数前缀和（杜教筛）
    map<long long, long long> mu_sum_cache;

    long long mu_sum(long long x) {
        if (x <= n) {
            long long res = 0;
            for (int i = 1; i <= x; i++) { res += mu[i]; }
            return res;
        }

        if (mu_sum_cache.count(x)) { return mu_sum_cache[x]; }

        long long res = 1;  // mu[1] = 1
        for (long long i = 2, j; i <= x; i = j + 1) {
            j = x / (x / i);
            res -= (j - i + 1) * mu_sum(x / i);
        }

        return mu_sum_cache[x] = res;
    }

    // 欧拉函数前缀和（杜教筛）
    map<long long, long long> phi_sum_cache;

    long long phi_sum(long long x) {
        if (x <= n) {
            long long res = 0;
            for (int i = 1; i <= x; i++) { res += phi[i]; }
            return res;
        }

        if (phi_sum_cache.count(x)) { return phi_sum_cache[x]; }

        long long res = x * (x + 1) / 2;
        for (long long i = 2, j; i <= x; i = j + 1) {
            j = x / (x / i);
            res -= (j - i + 1) * phi_sum(x / i);
        }

        return phi_sum_cache[x] = res;
    }

    // 计算gcd(i,n)=1的i的个数（1<=i<=m）
    long long count_coprime(long long m, long long n) {
        if (m == 0) return 0;
        if (n == 1) return m;

        vector<long long> factors;
        long long temp = n;
        for (long long i = 2; i * i <= temp; i++) {
            if (temp % i == 0) {
                factors.push_back(i);
                while (temp % i == 0) temp /= i;
            }
        }
        if (temp > 1) factors.push_back(temp);

        long long res = 0;
        int sz = factors.size();

        // 容斥原理
        for (int mask = 0; mask < (1 << sz); mask++) {
            long long prod = 1;
            int bits = 0;

            for (int i = 0; i < sz; i++) {
                if (mask & (1 << i)) {
                    prod *= factors[i];
                    bits++;
                }
            }

            if (bits & 1) {
                res -= m / prod;
            } else {
                res += m / prod;
            }
        }

        return res;
    }

    // 莫比乌斯反演求解经典问题
    // 求sum_{i=1}^n sum_{j=1}^m [gcd(i,j)=1]
    long long count_coprime_pairs(long long n, long long m) {
        if (n > m) swap(n, m);
        long long res = 0;

        for (int d = 1; d <= n; d++) { res += mu[d] * (n / d) * (m / d); }

        return res;
    }

    // 求sum_{i=1}^n sum_{j=1}^m gcd(i,j)
    long long sum_gcd(long long n, long long m) {
        if (n > m) swap(n, m);
        long long res = 0;

        for (int d = 1; d <= n; d++) { res += d * phi[d] * (n / d) * (m / d); }

        return res;
    }
};

// 使用示例
/*
MobiusFunction<int> mf(100000);
cout << mf.get_mu(30) << endl; // 输出-1
cout << mf.mu_sum(100) << endl; // 输出莫比乌斯函数前100项和
cout << mf.count_coprime_pairs(100, 100) << endl; // 计算互质对数
*/
]=]),

-- 04_Math\Number_Theory\Advanced\PrimitiveRoot.h
ps("04_math_number_theory_advanced_primitiveroot_h", [=[

using ll = long long;

struct PrimitiveRoot {
    ll power(ll a, ll b, ll m) {
        ll res = 1;
        a %= m;
        while (b > 0) {
            if (b & 1) res = res * a % m;
            a = a * a % m;
            b >>= 1;
        }
        return res;
    }

    ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }

    vector<ll> factorize(ll n) {
        vector<ll> factors;
        for (ll i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                factors.push_back(i);
                while (n % i == 0) n /= i;
            }
        }
        if (n > 1) factors.push_back(n);
        return factors;
    }

    ll euler_phi(ll n) {
        ll res = n;
        for (ll i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                while (n % i == 0) n /= i;
                res -= res / i;
            }
        }
        if (n > 1) res -= res / n;
        return res;
    }

    bool is_primitive_root(ll g, ll m) {
        if (gcd(g, m) != 1) return false;
        ll phi = euler_phi(m);
        vector<ll> factors = factorize(phi);
        for (ll factor : factors) {
            if (power(g, phi / factor, m) == 1) return false;
        }
        return true;
    }

    ll find_primitive_root(ll m) {
        if (m == 1) return 0;
        if (m == 2) return 1;
        if (m == 4) return 3;

        for (ll g = 1; g < m; g++) {
            if (is_primitive_root(g, m)) return g;
        }
        return -1;
    }

    vector<ll> all_primitive_roots(ll m) {
        vector<ll> roots;
        ll g = find_primitive_root(m);
        if (g == -1) return roots;

        ll phi = euler_phi(m);
        for (ll i = 1; i < phi; i++) {
            if (gcd(i, phi) == 1) {
                roots.push_back(power(g, i, m));
            }
        }
        sort(roots.begin(), roots.end());
        return roots;
    }
};
]=]),

-- 04_Math\Number_Theory\Advanced\QuadraticResidue.h
ps("04_math_number_theory_advanced_quadraticresidue_h", [=[

using ll = long long;

struct QuadraticResidue {
    ll power(ll a, ll b, ll p) {
        ll res = 1;
        a %= p;
        while (b > 0) {
            if (b & 1) res = res * a % p;
            a = a * a % p;
            b >>= 1;
        }
        return res;
    }

    ll legendre(ll a, ll p) { return power(a, (p - 1) / 2, p); }

    bool is_residue(ll a, ll p) {
        if (p == 2) return true;
        return legendre(a, p) == 1;
    }

    ll solve(ll n, ll p) {
        if (p == 2) return n % 2;
        if (legendre(n, p) != 1) return -1;

        if (p % 4 == 3) return power(n, (p + 1) / 4, p);

        ll s = 0, q = p - 1;
        while (q % 2 == 0) {
            q /= 2;
            s++;
        }

        ll z = 2;
        while (legendre(z, p) != p - 1) z++;

        ll m = s, c = power(z, q, p);
        ll t = power(n, q, p);
        ll r = power(n, (q + 1) / 2, p);

        while (t != 1) {
            ll i = 1, temp = t * t % p;
            while (temp != 1 && i < m) {
                temp = temp * temp % p;
                i++;
            }

            ll b = power(c, 1LL << (m - i - 1), p);
            m = i;
            c = b * b % p;
            t = t * c % p;
            r = r * b % p;
        }

        return r;
    }

    vector<ll> all_solutions(ll n, ll p) {
        ll x = solve(n, p);
        if (x == -1) return {};
        if (x == 0) return {0};
        return {x, p - x};
    }
};
]=]),

-- 04_Math\Number_Theory\Basic\Euler.h
ps("04_math_number_theory_basic_euler_h", [=[

using ll = long long;

// 基础欧拉函数
struct EulerPhi {
    // 单个数的欧拉函数 φ(n)
    ll phi(ll n) {
        ll res = n;
        for (ll i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                res = res / i * (i - 1);
                while (n % i == 0) n /= i;
            }
        }
        if (n > 1) res = res / n * (n - 1);
        return res;
    }
    
    // 快速幂
    ll power(ll a, ll b, ll mod) {
        ll res = 1;
        while (b > 0) {
            if (b & 1) res = res * a % mod;
            a = a * a % mod;
            b >>= 1;
        }
        return res;
    }
    
    // 欧拉定理：a^φ(m) ≡ 1 (mod m) when gcd(a,m)=1
    ll euler_theorem(ll a, ll m) {
        if (__gcd(a, m) != 1) return -1;
        return power(a, phi(m), m);
    }
    
    // 模逆元（使用欧拉定理）
    ll mod_inverse(ll a, ll m) {
        if (__gcd(a, m) != 1) return -1;
        return power(a, phi(m) - 1, m);
    }
};

// 线性筛欧拉函数
struct LinearEuler {
    vector<ll> phi, primes;
    vector<bool> is_prime;
    int n;
    
    LinearEuler(int maxn) : n(maxn) {
        phi.resize(n + 1);
        is_prime.resize(n + 1, true);
        
        phi[1] = 1;
        is_prime[0] = is_prime[1] = false;
        
        for (int i = 2; i <= n; i++) {
            if (is_prime[i]) {
                primes.push_back(i);
                phi[i] = i - 1;
            }
            
            for (ll p : primes) {
                if (i * p > n) break;
                is_prime[i * p] = false;
                
                if (i % p == 0) {
                    phi[i * p] = phi[i] * p;
                    break;
                } else {
                    phi[i * p] = phi[i] * (p - 1);
                }
            }
        }
    }
    
    ll get_phi(int x) {
        if (x <= n) return phi[x];
        EulerPhi ep;
        return ep.phi(x);
    }
    
    vector<ll> get_primes() { return primes; }
};

// 欧拉函数性质和应用
struct EulerApps {
    // 欧拉函数的因子和性质：∑φ(d) = n (d|n)
    ll sum_phi_divisors(ll n) {
        ll sum = 0;
        for (ll i = 1; i * i <= n; i++) {
            if (n % i == 0) {
                EulerPhi ep;
                sum += ep.phi(i);
                if (i != n / i) sum += ep.phi(n / i);
            }
        }
        return sum;
    }
    
    // 扩展欧拉定理
    ll extended_euler(ll a, ll n, ll m) {
        EulerPhi ep;
        ll phi_m = ep.phi(m);
        if (n < 32) return ep.power(a, n, m);
        return ep.power(a, n % phi_m + phi_m, m);
    }
    
    // 寻找φ(x) = n的所有x
    vector<ll> inverse_phi(ll n) {
        vector<ll> res;
        if (n == 1) {
            res.push_back(1);
            res.push_back(2);
            return res;
        }
        
        EulerPhi ep;
        for (ll x = n; x <= 2 * n; x++) {
            if (ep.phi(x) == n) res.push_back(x);
        }
        return res;
    }
};


]=]),

}
