-- Auto-generated LuaSnip snippets
local ls = require("luasnip")
local ps = ls.parser.parse_snippet

return {

-- 04_Math\Number_Theory\Basic\ExtendedGCD.h
ps("04_math_number_theory_basic_extendedgcd_h", [=[

using ll = long long;
using pll = pair<ll, ll>;

// 扩展欧几里得算法
struct ExtGCD {
    // 返回 {gcd(a,b), x, y} 使得 ax + by = gcd(a,b)
    array<ll, 3> exgcd(ll a, ll b) {
        if (!b) return {a, 1, 0};
        auto [g, x, y] = exgcd(b, a % b);
        return {g, y, x - a / b * y};
    }
    
    // 模逆元计算
    ll inv(ll a, ll m) {
        auto [g, x, y] = exgcd(a, m);
        if (g != 1) return -1;
        return (x % m + m) % m;
    }
    
    // 线性同余方程 ax ≡ b (mod m)
    pair<bool, ll> solve_congruence(ll a, ll b, ll m) {
        auto [g, x, y] = exgcd(a, m);
        if (b % g) return {false, 0};
        x = x * (b / g) % (m / g);
        if (x < 0) x += m / g;
        return {true, x};
    }
};

// 批量模逆元
struct BatchInverse {
    vector<ll> batch_inv(vector<ll> a, ll mod) {
        int n = a.size();
        vector<ll> s(n + 1, 1);
        for (int i = 0; i < n; i++) s[i + 1] = s[i] * a[i] % mod;
        
        ExtGCD egcd;
        ll inv_all = egcd.inv(s[n], mod);
        for (int i = n - 1; i >= 0; i--) {
            a[i] = s[i] * inv_all % mod;
            inv_all = inv_all * (a[i] * s[i] % mod) % mod;
        }
        return a;
    }
};

// 不定方程求解器
struct DiophantineEq {
    ll x, y, g;
    
    // 求解 ax + by = c
    bool solve(ll a, ll b, ll c) {
        ExtGCD egcd;
        auto [gcd, px, py] = egcd.exgcd(a, b);
        g = gcd;
        if (c % g) return false;
        x = px * (c / g);
        y = py * (c / g);
        return true;
    }
    
    // 通解：x = x0 + k*(b/g), y = y0 - k*(a/g)
    pair<ll, ll> general_solution(ll k, ll a, ll b) {
        return {x + k * (b / g), y - k * (a / g)};
    }
    
    // 最小正整数解
    ll min_positive_x(ll a, ll b) {
        ll step = abs(b / g);
        ll res = ((x % step) + step) % step;
        return res == 0 ? step : res;
    }
};
]=]),

-- 04_Math\Number_Theory\Basic\GCD_LCM.h
ps("04_math_number_theory_basic_gcd_lcm_h", [=[

using ll = long long;

// 基础GCD和LCM
struct BasicGCD {
    ll gcd(ll a, ll b) {
        while (b) {
            a %= b;
            swap(a, b);
        }
        return a;
    }
    
    ll lcm(ll a, ll b) {
        return a / gcd(a, b) * b;
    }
    
    ll gcd_multiple(const vector<ll>& arr) {
        ll res = arr[0];
        for (int i = 1; i < arr.size(); i++) {
            res = gcd(res, arr[i]);
            if (res == 1) break;
        }
        return res;
    }
    
    ll lcm_multiple(const vector<ll>& arr) {
        ll res = arr[0];
        for (int i = 1; i < arr.size(); i++) res = lcm(res, arr[i]);
        return res;
    }
};

// 二进制GCD（Stein算法）
struct BinaryGCD {
    ll gcd(ll a, ll b) {
        if (a == 0) return b;
        if (b == 0) return a;
        
        int shift = 0;
        while (((a | b) & 1) == 0) {
            shift++;
            a >>= 1;
            b >>= 1;
        }
        
        while ((a & 1) == 0) a >>= 1;
        
        do {
            while ((b & 1) == 0) b >>= 1;
            if (a > b) swap(a, b);
            b -= a;
        } while (b != 0);
        
        return a << shift;
    }
};

// 分数运算
struct Fraction {
    ll num, den;
    
    Fraction(ll n = 0, ll d = 1) {
        if (d < 0) {
            n = -n;
            d = -d;
        }
        ll g = __gcd(abs(n), abs(d));
        num = n / g;
        den = d / g;
    }
    
    Fraction operator+(const Fraction& b) const {
        return Fraction(num * b.den + b.num * den, den * b.den);
    }
    
    Fraction operator-(const Fraction& b) const {
        return Fraction(num * b.den - b.num * den, den * b.den);
    }
    
    Fraction operator*(const Fraction& b) const {
        return Fraction(num * b.num, den * b.den);
    }
    
    Fraction operator/(const Fraction& b) const {
        return Fraction(num * b.den, den * b.num);
    }
    
    bool operator<(const Fraction& b) const {
        return num * b.den < b.num * den;
    }
    
    bool operator==(const Fraction& b) const {
        return num == b.num && den == b.den;
    }
    
    string toString() const {
        if (den == 1) return to_string(num);
        return to_string(num) + "/" + to_string(den);
    }
};

// 范围GCD查询（稀疏表）
struct RangeGCD {
    vector<vector<ll>> st;
    vector<int> lg;
    
    RangeGCD(const vector<ll>& arr) {
        int n = arr.size();
        lg.resize(n + 1);
        lg[1] = 0;
        for (int i = 2; i <= n; i++) lg[i] = lg[i / 2] + 1;
        
        int k = lg[n] + 1;
        st.assign(k, vector<ll>(n));
        
        for (int i = 0; i < n; i++) st[0][i] = arr[i];
        
        for (int j = 1; j < k; j++) {
            for (int i = 0; i + (1 << j) <= n; i++) {
                st[j][i] = __gcd(st[j-1][i], st[j-1][i + (1 << (j-1))]);
            }
        }
    }
    
    ll query(int l, int r) {
        int j = lg[r - l + 1];
        return __gcd(st[j][l], st[j][r - (1 << j) + 1]);
    }
};


]=]),

-- 04_Math\Number_Theory\Basic\LinearSieve.h
ps("04_math_number_theory_basic_linearsieve_h", [=[

using ll = long long;

struct LinearSieve {
    vector<int> primes;
    vector<bool> is_prime;
    int n;

    LinearSieve(int limit) : n(limit) {
        is_prime.assign(n + 1, true);
        is_prime[0] = is_prime[1] = false;
        
        for (int i = 2; i <= n; i++) {
            if (is_prime[i]) primes.push_back(i);
            for (int j = 0; j < primes.size() && i * primes[j] <= n; j++) {
                is_prime[i * primes[j]] = false;
                if (i % primes[j] == 0) break;
            }
        }
    }

    bool check(int x) { return x <= n && is_prime[x]; }
    vector<int> get_primes() { return primes; }
};

struct EulerSieve {
    vector<int> primes, phi;
    int n;

    EulerSieve(int limit) : n(limit) {
        phi.assign(n + 1, 0);
        phi[1] = 1;
        vector<bool> is_prime(n + 1, true);
        is_prime[0] = is_prime[1] = false;

        for (int i = 2; i <= n; i++) {
            if (is_prime[i]) {
                primes.push_back(i);
                phi[i] = i - 1;
            }
            for (int j = 0; j < primes.size() && i * primes[j] <= n; j++) {
                int ip = i * primes[j];
                is_prime[ip] = false;
                if (i % primes[j] == 0) {
                    phi[ip] = phi[i] * primes[j];
                    break;
                } else {
                    phi[ip] = phi[i] * (primes[j] - 1);
                }
            }
        }
    }
};

struct MobiusSieve {
    vector<int> primes, mu;
    int n;

    MobiusSieve(int limit) : n(limit) {
        mu.assign(n + 1, 0);
        mu[1] = 1;
        vector<bool> is_prime(n + 1, true);
        is_prime[0] = is_prime[1] = false;

        for (int i = 2; i <= n; i++) {
            if (is_prime[i]) {
                primes.push_back(i);
                mu[i] = -1;
            }
            for (int j = 0; j < primes.size() && i * primes[j] <= n; j++) {
                int ip = i * primes[j];
                is_prime[ip] = false;
                if (i % primes[j] == 0) {
                    mu[ip] = 0;
                    break;
                } else {
                    mu[ip] = -mu[i];
                }
            }
        }
    }
};
]=]),

-- 04_Math\Number_Theory\Basic\Sieve.h
ps("04_math_number_theory_basic_sieve_h", [=[

using ll = long long;

// 埃拉托斯特尼筛法
struct Sieve {
    vector<bool> is_prime;
    vector<int> primes;
    int n;
    
    Sieve(int limit) : n(limit) {
        is_prime.assign(n + 1, true);
        is_prime[0] = is_prime[1] = false;
        
        for (int i = 2; i * i <= n; i++) {
            if (is_prime[i]) {
                for (int j = i * i; j <= n; j += i) is_prime[j] = false;
            }
        }
        
        for (int i = 2; i <= n; i++) {
            if (is_prime[i]) primes.push_back(i);
        }
    }
    
    bool check(int x) { return x <= n && is_prime[x]; }
    vector<int> get_primes() { return primes; }
    int count() { return primes.size(); }
};

// 区间筛
struct SegmentSieve {
    vector<bool> sieve(ll L, ll R) {
        vector<bool> is_prime(R - L + 1, true);
        for (ll p = 2; p * p <= R; p++) {
            ll start = max(p * p, (L + p - 1) / p * p);
            for (ll j = start; j <= R; j += p) is_prime[j - L] = false;
        }
        if (L == 1) is_prime[0] = false;
        return is_prime;
    }
};
]=]),

-- 04_Math\Number_Theory\Congruences\CRT.h
ps("04_math_number_theory_congruences_crt_h", [=[

// 中国剩余定理 (Chinese Remainder Theorem)
using ll = long long;

// 扩展欧几里得算法
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

// 求逆元
ll inv(ll a, ll m) {
    ll x, y;
    ll d = exgcd(a, m, x, y);
    if (d != 1) return -1;  // 不存在逆元
    return (x % m + m) % m;
}

// 中国剩余定理（模数互质）
// 解方程组: x ≡ a[i] (mod m[i])
ll crt(vector<ll> a, vector<ll> m) {
    int n = a.size();
    ll M = 1;
    for (int i = 0; i < n; i++) { M *= m[i]; }

    ll ans = 0;
    for (int i = 0; i < n; i++) {
        ll Mi = M / m[i];
        ll yi = inv(Mi, m[i]);
        if (yi == -1) return -1;  // 模数不互质
        ans = (ans + a[i] * Mi % M * yi) % M;
    }
    return (ans % M + M) % M;
}

// 扩展中国剩余定理（模数不要求互质）
// 解方程组: x ≡ a[i] (mod m[i])
ll excrt(vector<ll> a, vector<ll> m) {
    int n = a.size();
    ll x = a[0], lcm = m[0];

    for (int i = 1; i < n; i++) {
        ll A = lcm, B = m[i], C = (a[i] - x % B + B) % B;
        ll p, q;
        ll d = exgcd(A, B, p, q);

        if (C % d != 0) return -1;  // 无解

        ll t = C / d;
        p = (p * t % (B / d) + B / d) % (B / d);
        x += A * p;
        lcm = lcm / d * B;
    }
    return (x % lcm + lcm) % lcm;
}

// 解单个线性同余方程 ax ≡ b (mod m)
vector<ll> linear_congruence(ll a, ll b, ll m) {
    ll x, y;
    ll d = exgcd(a, m, x, y);

    if (b % d != 0) return {};  // 无解

    ll x0 = (x * (b / d) % m + m) % m;
    ll step = m / d;

    vector<ll> solutions;
    for (int i = 0; i < d; i++) { solutions.push_back((x0 + i * step) % m); }
    return solutions;
}

// Wilson定理：(p-1)! ≡ -1 (mod p) 当且仅当p是质数
bool wilson_test(ll p) {
    if (p <= 1) return false;
    if (p == 2) return true;
    if (p % 2 == 0) return false;

    ll fact = 1;
    for (ll i = 1; i < p; i++) { fact = (fact * i) % p; }
    return fact == p - 1;
}

// 快速幂
ll binpower(ll a, ll b, ll m) {
    ll res = 1;
    a %= m;
    while (b > 0) {
        if (b & 1) res = (res * a) % m;
        a = (a * a) % m;
        b >>= 1;
    }
    return res;
}

// 求解二次同余方程 x² ≡ n (mod p)，p为奇质数
vector<ll> quadratic_residue(ll n, ll p) {
    n %= p;
    if (n == 0) return {0};

    // 勒让德符号判断
    if (binpower(n, (p - 1) / 2, p) != 1) return {};  // n不是二次剩余

    // Tonelli-Shanks算法
    ll q = p - 1, s = 0;
    while (q % 2 == 0) {
        q /= 2;
        s++;
    }

    if (s == 1) {
        ll x = binpower(n, (p + 1) / 4, p);
        return {x, p - x};
    }

    // 找一个二次非剩余z
    ll z = 2;
    while (binpower(z, (p - 1) / 2, p) != p - 1) z++;

    ll m = s, c = binpower(z, q, p);
    ll t = binpower(n, q, p), r = binpower(n, (q + 1) / 2, p);

    while (t != 1) {
        ll i = 1;
        ll temp = (t * t) % p;
        while (temp != 1) {
            temp = (temp * temp) % p;
            i++;
        }

        ll b = binpower(c, 1LL << (m - i - 1), p);
        m = i;
        c = (b * b) % p;
        t = (t * c) % p;
        r = (r * b) % p;
    }

    return {r, p - r};
}
]=]),

-- 04_Math\Number_Theory\Congruences\DiscreteLog.h
ps("04_math_number_theory_congruences_discretelog_h", [=[

// 离散对数 (Discrete Logarithm)
using ll = long long;

// 快速幂
ll binpower(ll base, ll exp, ll mod) {
    ll result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

// Baby-step Giant-step 算法
// 求解 a^x ≡ b (mod m)，返回最小的非负整数x
ll bsgs(ll a, ll b, ll m) {
    if (b == 1) return 0;

    a %= m;
    b %= m;

    int n = sqrt(m) + 1;

    // Baby steps: 计算 a^j mod m, j = 0, 1, ..., n-1
    unordered_map<ll, int> baby_steps;
    ll gamma = 1;
    for (int j = 0; j < n; j++) {
        if (baby_steps.find(gamma) == baby_steps.end()) { baby_steps[gamma] = j; }
        gamma = (gamma * a) % m;
    }

    // Giant steps: 计算 b * (a^n)^(-i) mod m, i = 0, 1, ..., n-1
    ll an = binpower(a, n, m);
    ll inv_an = binpower(an, m - 2, m);  // 费马小定理求逆元（假设m是质数）

    ll y = b;
    for (int i = 0; i < n; i++) {
        if (baby_steps.find(y) != baby_steps.end()) {
            ll ans = i * n + baby_steps[y];
            if (ans > 0) return ans;
        }
        y = (y * inv_an) % m;
    }

    return -1;  // 无解
}

// 扩展欧几里得算法
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

// 求模逆元
ll modinv(ll a, ll m) {
    ll x, y;
    ll d = exgcd(a, m, x, y);
    if (d != 1) return -1;
    return (x % m + m) % m;
}

// 处理gcd(a,m) > 1的情况的BSGS
ll exbsgs(ll a, ll b, ll m) {
    a %= m;
    b %= m;
    if (b == 1) return 0;

    ll d, k = 0, ak = 1;

    // 处理gcd(a,m) > 1的情况
    while ((d = __gcd(a, m)) > 1) {
        if (b % d != 0) return -1;
        b /= d;
        m /= d;
        k++;
        ak = (ak * (a / d)) % m;
        if (b == ak) return k;
    }

    // 现在gcd(a,m) = 1，可以用标准BSGS
    int n = sqrt(m) + 1;

    unordered_map<ll, int> baby_steps;
    ll gamma = 1;
    for (int j = 0; j < n; j++) {
        if (baby_steps.find(gamma) == baby_steps.end()) { baby_steps[gamma] = j; }
        gamma = (gamma * a) % m;
    }

    ll an = binpower(a, n, m);
    ll inv_an = modinv(an, m);
    if (inv_an == -1) return -1;

    ll y = (b * modinv(ak, m)) % m;

    for (int i = 0; i < n; i++) {
        if (baby_steps.find(y) != baby_steps.end()) {
            ll ans = k + i * n + baby_steps[y];
            return ans;
        }
        y = (y * inv_an) % m;
    }

    return -1;
}

// Pohlig-Hellman算法（当模数p-1有小的质因数时）
ll pohlig_hellman(ll a, ll b, ll p) {
    // 对于素数p，求解 a^x ≡ b (mod p)
    ll order = p - 1;
    vector<pair<ll, int>> factors;

    // 分解order = p-1
    ll temp = order;
    for (ll i = 2; i * i <= temp; i++) {
        if (temp % i == 0) {
            int cnt = 0;
            while (temp % i == 0) {
                temp /= i;
                cnt++;
            }
            factors.push_back({i, cnt});
        }
    }
    if (temp > 1) factors.push_back({temp, 1});

    vector<ll> remainders, moduli;

    for (auto [q, e] : factors) {
        ll qe = 1;
        for (int i = 0; i < e; i++) qe *= q;

        ll a_reduced = binpower(a, order / qe, p);
        ll b_reduced = binpower(b, order / qe, p);

        ll x_mod_qe = bsgs(a_reduced, b_reduced, p);
        if (x_mod_qe == -1) return -1;

        remainders.push_back(x_mod_qe);
        moduli.push_back(qe);
    }

    // 使用中国剩余定理合并结果
    ll result = remainders[0], lcm = moduli[0];
    for (int i = 1; i < remainders.size(); i++) {
        ll a1 = lcm, b1 = moduli[i], c = (remainders[i] - result % b1 + b1) % b1;
        ll x, y;
        ll d = exgcd(a1, b1, x, y);
        if (c % d != 0) return -1;
        ll t = c / d;
        x = (x * t % (b1 / d) + b1 / d) % (b1 / d);
        result += a1 * x;
        lcm = lcm / d * b1;
    }

    return (result % order + order) % order;
}
]=]),

-- 04_Math\Number_Theory\Congruences\ExCRT.h
ps("04_math_number_theory_congruences_excrt_h", [=[

// 扩展中国剩余定理
using ll = long long;

// 扩展欧几里得算法
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

// 扩展中国剩余定理：解方程组 x ≡ a[i] (mod m[i])
// 不要求模数互质
struct ExCRT {
    vector<ll> a, m;

    void add_congruence(ll ai, ll mi) {
        a.push_back(ai);
        m.push_back(mi);
    }

    ll solve() {
        if (a.empty()) return 0;

        ll x = a[0], lcm = m[0];

        for (int i = 1; i < a.size(); i++) {
            ll A = lcm, B = m[i], C = (a[i] - x % B + B) % B;
            ll p, q;
            ll d = exgcd(A, B, p, q);

            if (C % d != 0) return -1;  // 无解

            ll t = C / d;
            p = (p * t % (B / d) + B / d) % (B / d);
            x += A * p;
            lcm = lcm / d * B;
        }

        return (x % lcm + lcm) % lcm;
    }

    // 检查解的存在性
    bool has_solution() {
        if (a.empty()) return true;

        ll x = a[0], lcm = m[0];

        for (int i = 1; i < a.size(); i++) {
            ll A = lcm, B = m[i], C = (a[i] - x % B + B) % B;
            ll p, q;
            ll d = exgcd(A, B, p, q);

            if (C % d != 0) return false;

            ll t = C / d;
            p = (p * t % (B / d) + B / d) % (B / d);
            x += A * p;
            lcm = lcm / d * B;
        }

        return true;
    }

    // 获取解的模数
    ll get_modulus() {
        if (a.empty()) return 1;

        ll lcm = m[0];
        for (int i = 1; i < m.size(); i++) {
            ll d = __gcd(lcm, m[i]);
            lcm = lcm / d * m[i];
        }
        return lcm;
    }

    void clear() {
        a.clear();
        m.clear();
    }
};

// 直接求解函数版本
ll excrt(vector<ll> a, vector<ll> m) {
    if (a.empty()) return 0;

    ll x = a[0], lcm = m[0];

    for (int i = 1; i < a.size(); i++) {
        ll A = lcm, B = m[i], C = (a[i] - x % B + B) % B;
        ll p, q;
        ll d = exgcd(A, B, p, q);

        if (C % d != 0) return -1;  // 无解

        ll t = C / d;
        p = (p * t % (B / d) + B / d) % (B / d);
        x += A * p;
        lcm = lcm / d * B;
    }

    return (x % lcm + lcm) % lcm;
}

// 二元线性同余方程组求解
pair<ll, ll> solve_binary_system(ll a1, ll b1, ll m1, ll a2, ll b2, ll m2) {
    // 求解: a1*x + b1*y ≡ 0 (mod m1)
    //       a2*x + b2*y ≡ 0 (mod m2)

    ll x, y;
    ll d = exgcd(m1, m2, x, y);

    if ((a1 * b2 - a2 * b1) % d != 0) {
        return {-1, -1};  // 无解
    }

    ll lcm = m1 / d * m2;
    ll sol_x = (x * (a1 * b2 - a2 * b1) / d) % lcm;
    ll sol_y = (a1 == 0 ? 0 : (-a1 * sol_x - b1) / m1);

    return {(sol_x % lcm + lcm) % lcm, sol_y};
}
]=]),

-- 04_Math\Number_Theory\Factorization\PollardRho.h
ps("04_math_number_theory_factorization_pollardrho_h", [=[

// Pollard-Rho 算法进行大数因数分解
using u64 = uint64_t;
using u128 = __uint128_t;

u64 mulmod(u64 a, u64 b, u64 mod) { return ((u128)a * b) % mod; }

u64 binpower(u64 base, u64 e, u64 mod) {
    u64 result = 1;
    base %= mod;
    while (e) {
        if (e & 1) result = mulmod(result, base, mod);
        base = mulmod(base, base, mod);
        e >>= 1;
    }
    return result;
}

bool miller_rabin(u64 n, u64 a) {
    if (n == a) return true;
    if (n % 2 == 0) return false;

    u64 d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }

    u64 x = binpower(a, d, n);
    if (x == 1 || x == n - 1) return true;

    for (int i = 0; i < r - 1; i++) {
        x = mulmod(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

bool is_prime(u64 n) {
    if (n < 2) return false;
    static const u64 bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (u64 a : bases) {
        if (n == a) return true;
        if (!miller_rabin(n, a)) return false;
    }
    return true;
}

u64 f(u64 x, u64 c, u64 mod) { return (mulmod(x, x, mod) + c) % mod; }

u64 rho(u64 n) {
    if (n % 2 == 0) return 2;

    u64 x = 2, y = 2, c = 1;
    u64 d = 1;

    while (d == 1) {
        x = f(x, c, n);
        y = f(f(y, c, n), c, n);
        d = gcd(abs((long long)(x - y)), n);
        if (d == n) {
            c++;
            x = y = 2;
            d = 1;
        }
    }
    return d;
}

vector<u64> factorize(u64 n) {
    if (n == 1) return {};
    if (is_prime(n)) return {n};

    u64 factor = rho(n);
    auto left = factorize(factor);
    auto right = factorize(n / factor);

    left.insert(left.end(), right.begin(), right.end());
    return left;
}

// 返回质因数及其指数
map<u64, int> factor_count(u64 n) {
    auto factors = factorize(n);
    map<u64, int> count;
    for (u64 p : factors) { count[p]++; }
    return count;
}

// 获取所有因数
vector<u64> get_divisors(u64 n) {
    auto factor_map = factor_count(n);
    vector<u64> divisors = {1};

    for (auto [p, cnt] : factor_map) {
        int old_size = divisors.size();
        u64 power = 1;
        for (int i = 0; i < cnt; i++) {
            power *= p;
            for (int j = 0; j < old_size; j++) { divisors.push_back(divisors[j] * power); }
        }
    }

    sort(divisors.begin(), divisors.end());
    return divisors;
}
]=]),

-- 04_Math\Number_Theory\Factorization\QuadraticSieve.h
ps("04_math_number_theory_factorization_quadraticsieve_h", [=[

// 二次筛法（Quadratic Sieve）- 大整数因式分解
struct QuadraticSieve {
    static vector<long long> small_primes;
    static bool primes_initialized;

    static void init_primes(int limit = 10000) {
        if (primes_initialized) return;

        vector<bool> is_prime(limit + 1, true);
        is_prime[0] = is_prime[1] = false;

        for (int i = 2; i * i <= limit; i++) {
            if (is_prime[i]) {
                for (int j = i * i; j <= limit; j += i) { is_prime[j] = false; }
            }
        }

        for (int i = 2; i <= limit; i++) {
            if (is_prime[i]) { small_primes.push_back(i); }
        }

        primes_initialized = true;
    }

    // 计算勒让德符号 (a/p)
    static int legendre_symbol(long long a, long long p) {
        if (a % p == 0) return 0;

        long long result = 1;
        a %= p;

        while (a != 0) {
            while (a % 2 == 0) {
                a /= 2;
                if ((p * p - 1) / 8 % 2 == 1) { result = -result; }
            }

            swap(a, p);
            if ((a - 1) / 2 % 2 == 1 && (p - 1) / 2 % 2 == 1) { result = -result; }
            a %= p;
        }

        return p == 1 ? result : 0;
    }

    // 生成因子基
    static vector<long long> generate_factor_base(long long n, int base_size) {
        init_primes();
        vector<long long> factor_base;
        factor_base.push_back(-1);  // 包含-1

        for (long long p : small_primes) {
            if (factor_base.size() >= base_size) break;
            if (legendre_symbol(n, p) != -1) { factor_base.push_back(p); }
        }

        return factor_base;
    }

    // 计算模平方根（Tonelli-Shanks算法的简化版本）
    static long long mod_sqrt(long long n, long long p) {
        if (p == 2) return n % 2;
        if (legendre_symbol(n, p) != 1) return -1;

        // 简单情况：p ≡ 3 (mod 4)
        if (p % 4 == 3) {
            long long r = 1;
            long long exp = (p + 1) / 4;
            long long base = n % p;

            while (exp > 0) {
                if (exp & 1) r = r * base % p;
                base = base * base % p;
                exp >>= 1;
            }
            return r;
        }

        // 更复杂的情况需要Tonelli-Shanks算法
        // 这里简化实现
        for (long long i = 1; i < p; i++) {
            if (i * i % p == n % p) { return i; }
        }

        return -1;
    }

    // 筛选平滑数
    static vector<pair<long long, vector<int>>> sieve_smooth_numbers(long long n,
                                                                     const vector<long long>& factor_base,
                                                                     int sieve_interval) {
        vector<pair<long long, vector<int>>> smooth_numbers;
        long long sqrt_n = (long long)sqrt(n);

        for (long long x = sqrt_n; x < sqrt_n + sieve_interval; x++) {
            long long val = x * x - n;
            if (val <= 0) continue;

            vector<int> exponents(factor_base.size(), 0);
            long long temp = val;
            bool is_smooth = true;

            // 处理-1
            if (temp < 0) {
                exponents[0] = 1;
                temp = -temp;
            }

            // 试除因子基中的素数
            for (int i = 1; i < factor_base.size() && is_smooth; i++) {
                long long p = factor_base[i];
                while (temp % p == 0) {
                    exponents[i]++;
                    temp /= p;
                }
            }

            if (temp == 1) { smooth_numbers.push_back({x, exponents}); }
        }

        return smooth_numbers;
    }

    // 高斯消元求解线性方程组（模2）
    static vector<vector<int>> gaussian_elimination_mod2(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        if (rows == 0) return {};
        int cols = matrix[0].size();

        vector<vector<int>> solutions;
        vector<int> pivot_col(rows, -1);

        for (int col = 0, row = 0; col < cols && row < rows; col++) {
            // 找到主元
            int pivot_row = -1;
            for (int i = row; i < rows; i++) {
                if (matrix[i][col] == 1) {
                    pivot_row = i;
                    break;
                }
            }

            if (pivot_row == -1) continue;

            // 交换行
            if (pivot_row != row) { swap(matrix[row], matrix[pivot_row]); }

            pivot_col[row] = col;

            // 消元
            for (int i = 0; i < rows; i++) {
                if (i != row && matrix[i][col] == 1) {
                    for (int j = 0; j < cols; j++) { matrix[i][j] ^= matrix[row][j]; }
                }
            }

            row++;
        }

        return solutions;
    }

    // 主要的二次筛法函数
    static vector<long long> factorize(long long n) {
        if (n <= 1) return {};
        if (n <= 3) return {n};

        // 先用试除法处理小因子
        vector<long long> factors;
        for (long long p = 2; p * p <= n && p <= 1000; p++) {
            while (n % p == 0) {
                factors.push_back(p);
                n /= p;
            }
        }

        if (n == 1) return factors;
        if (n <= 1000000) {
            factors.push_back(n);
            return factors;
        }

        // 对于大数使用二次筛法
        init_primes();

        // 生成因子基
        int base_size = min(100, (int)small_primes.size());
        vector<long long> factor_base = generate_factor_base(n, base_size);

        // 筛选平滑数
        vector<pair<long long, vector<int>>> smooth_numbers = sieve_smooth_numbers(n, factor_base, 1000);

        if (smooth_numbers.size() < factor_base.size() + 1) {
            // 如果平滑数不够，回退到其他方法
            factors.push_back(n);
            return factors;
        }

        // 构建矩阵并求解
        vector<vector<int>> matrix;
        for (const auto& [x, exponents] : smooth_numbers) {
            vector<int> row;
            for (int exp : exponents) { row.push_back(exp % 2); }
            matrix.push_back(row);
        }

        // 这里简化处理，实际应该求解线性方程组
        // 如果找到了非平凡因子，返回它

        factors.push_back(n);
        return factors;
    }
};

vector<long long> QuadraticSieve::small_primes;
bool QuadraticSieve::primes_initialized = false;
]=]),

-- 04_Math\Number_Theory\Factorization\TrialDivision.h
ps("04_math_number_theory_factorization_trialdivision_h", [=[

// 试除法进行因数分解
using ll = long long;

// 基础试除法
vector<pair<ll, int>> factorize(ll n) {
    vector<pair<ll, int>> factors;

    // 处理2的因子
    if (n % 2 == 0) {
        int cnt = 0;
        while (n % 2 == 0) {
            n /= 2;
            cnt++;
        }
        factors.push_back({2, cnt});
    }

    // 处理奇数因子
    for (ll i = 3; i * i <= n; i += 2) {
        if (n % i == 0) {
            int cnt = 0;
            while (n % i == 0) {
                n /= i;
                cnt++;
            }
            factors.push_back({i, cnt});
        }
    }

    // 如果n仍大于1，说明它是一个质数
    if (n > 1) { factors.push_back({n, 1}); }

    return factors;
}

// 获取所有因数
vector<ll> get_divisors(ll n) {
    auto factors = factorize(n);
    vector<ll> divisors = {1};

    for (auto [p, cnt] : factors) {
        int old_size = divisors.size();
        ll power = 1;
        for (int i = 0; i < cnt; i++) {
            power *= p;
            for (int j = 0; j < old_size; j++) { divisors.push_back(divisors[j] * power); }
        }
    }

    sort(divisors.begin(), divisors.end());
    return divisors;
}

// 计算因数个数
ll count_divisors(ll n) {
    auto factors = factorize(n);
    ll count = 1;
    for (auto [p, cnt] : factors) { count *= (cnt + 1); }
    return count;
}

// 计算因数和
ll sum_of_divisors(ll n) {
    auto factors = factorize(n);
    ll sum = 1;
    for (auto [p, cnt] : factors) {
        ll power_sum = 0;
        ll power = 1;
        for (int i = 0; i <= cnt; i++) {
            power_sum += power;
            power *= p;
        }
        sum *= power_sum;
    }
    return sum;
}

// 判断是否为完全数
bool is_perfect(ll n) { return sum_of_divisors(n) == 2 * n; }

// 判断是否为亏数
bool is_deficient(ll n) { return sum_of_divisors(n) < 2 * n; }

// 判断是否为过剩数
bool is_abundant(ll n) { return sum_of_divisors(n) > 2 * n; }
]=]),

-- 04_Math\Number_Theory\Primality\Fermat.h
ps("04_math_number_theory_primality_fermat_h", [=[

// 费马素性测试
using u64 = uint64_t;
using u128 = __uint128_t;

u64 binpower(u64 base, u64 e, u64 mod) {
    u64 result = 1;
    base %= mod;
    while (e) {
        if (e & 1) result = (u128)result * base % mod;
        base = (u128)base * base % mod;
        e >>= 1;
    }
    return result;
}

// 基础费马测试
bool fermat_test(u64 n, u64 a) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    return binpower(a, n - 1, n) == 1;
}

// 使用多个基数的费马测试
bool is_prime(u64 n, int k = 10) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

    for (int i = 0; i < k; i++) {
        u64 a = uniform_int_distribution<u64>(2, n - 2)(rng);
        if (!fermat_test(n, a)) return false;
    }
    return true;
}

// 检测卡迈克尔数（费马伪素数）
bool is_carmichael(u64 n) {
    if (n <= 1 || n % 2 == 0) return false;

    // 简单检测：对多个基数进行费马测试
    vector<u64> bases = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    for (u64 a : bases) {
        if (a >= n) break;
        if (__gcd(a, n) != 1) return false;
        if (binpower(a, n - 1, n) != 1) return false;
    }

    // 检查是否确实是合数
    for (u64 i = 2; i * i <= n; i++) {
        if (n % i == 0) return true;
    }
    return false;
}
]=]),

-- 04_Math\Number_Theory\Primality\MillerRabin.h
ps("04_math_number_theory_primality_millerrabin_h", [=[

// Miller-Rabin 素性测试
using u64 = uint64_t;
using u128 = __uint128_t;

u64 binpower(u64 base, u64 e, u64 mod) {
    u64 result = 1;
    base %= mod;
    while (e) {
        if (e & 1) result = (u128)result * base % mod;
        base = (u128)base * base % mod;
        e >>= 1;
    }
    return result;
}

bool check_composite(u64 n, u64 a, u64 d, int s) {
    u64 x = binpower(a, d, n);
    if (x == 1 || x == n - 1) return false;
    for (int r = 1; r < s; r++) {
        x = (u128)x * x % n;
        if (x == n - 1) return false;
    }
    return true;
}

bool is_prime(u64 n) {
    if (n < 2) return false;

    int r = 0;
    u64 d = n - 1;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }

    // 测试基底，对于64位数这些基底足够
    for (u64 a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
        if (n == a) return true;
        if (check_composite(n, a, d, r)) return false;
    }
    return true;
}

// 使用示例
bool isPrime(long long n) { return is_prime(n); }
]=]),

-- 04_Math\Number_Theory\Primality\SolovayStrassen.h
ps("04_math_number_theory_primality_solovaystrassen_h", [=[

// Solovay-Strassen 素性测试
using u64 = uint64_t;
using u128 = __uint128_t;

u64 binpower(u64 base, u64 e, u64 mod) {
    u64 result = 1;
    base %= mod;
    while (e) {
        if (e & 1) result = (u128)result * base % mod;
        base = (u128)base * base % mod;
        e >>= 1;
    }
    return result;
}

// 计算雅可比符号 (a/n)
int jacobi_symbol(u64 a, u64 n) {
    if (n <= 0 || n % 2 == 0) return 0;

    int result = 1;
    a %= n;

    while (a != 0) {
        while (a % 2 == 0) {
            a /= 2;
            if (n % 8 == 3 || n % 8 == 5) result = -result;
        }
        swap(a, n);
        if (a % 4 == 3 && n % 4 == 3) result = -result;
        a %= n;
    }

    return (n == 1) ? result : 0;
}

// Solovay-Strassen 测试
bool solovay_strassen_test(u64 n, u64 a) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    if (__gcd(a, n) != 1) return false;

    int jacobi = jacobi_symbol(a, n);
    if (jacobi == 0) return false;

    u64 exp = (n - 1) / 2;
    u64 mod_result = binpower(a, exp, n);

    // 将雅可比符号转换为模n的值
    if (jacobi == -1) jacobi = n - 1;

    return mod_result == jacobi;
}

// 使用多次随机测试
bool is_prime(u64 n, int k = 10) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

    for (int i = 0; i < k; i++) {
        u64 a = uniform_int_distribution<u64>(2, n - 2)(rng);
        if (!solovay_strassen_test(n, a)) return false;
    }
    return true;
}
]=]),

-- 04_Math\Polynomial\FFT.h
ps("04_math_polynomial_fft_h", [=[

// 快速傅里叶变换 (FFT) 模板
template <typename T = double>
struct FFT {
    const double PI = acos(-1.0);

    struct Complex {
        T real, imag;

        Complex(T r = 0, T i = 0) : real(r), imag(i) {}

        Complex operator+(const Complex& other) const { return Complex(real + other.real, imag + other.imag); }

        Complex operator-(const Complex& other) const { return Complex(real - other.real, imag - other.imag); }

        Complex operator*(const Complex& other) const {
            return Complex(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
        }

        Complex conj() const { return Complex(real, -imag); }

        T norm() const { return real * real + imag * imag; }

        T abs() const { return sqrt(norm()); }
    };

    // 位逆序置换
    void bit_reverse(vector<Complex>& a) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) { j ^= bit; }
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
    }

    // FFT主函数
    void fft(vector<Complex>& a, bool invert = false) {
        bit_reverse(a);
        int n = a.size();

        for (int len = 2; len <= n; len <<= 1) {
            T angle = 2 * PI / len * (invert ? -1 : 1);
            Complex wlen(cos(angle), sin(angle));

            for (int i = 0; i < n; i += len) {
                Complex w(1);
                for (int j = 0; j < len / 2; j++) {
                    Complex u = a[i + j];
                    Complex v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w = w * wlen;
                }
            }
        }

        if (invert) {
            for (Complex& x : a) {
                x.real /= n;
                x.imag /= n;
            }
        }
    }

    // 多项式乘法
    vector<T> multiply(vector<T>& a, vector<T>& b) {
        vector<Complex> fa(a.begin(), a.end());
        vector<Complex> fb(b.begin(), b.end());

        int n = 1;
        while (n < a.size() + b.size()) n <<= 1;

        fa.resize(n);
        fb.resize(n);

        fft(fa);
        fft(fb);

        for (int i = 0; i < n; i++) { fa[i] = fa[i] * fb[i]; }

        fft(fa, true);

        vector<T> result(a.size() + b.size() - 1);
        for (int i = 0; i < result.size(); i++) { result[i] = round(fa[i].real); }

        return result;
    }

    // 大整数乘法
    string multiply_bigint(string num1, string num2) {
        vector<T> a, b;

        for (int i = num1.length() - 1; i >= 0; i--) { a.push_back(num1[i] - '0'); }
        for (int i = num2.length() - 1; i >= 0; i--) { b.push_back(num2[i] - '0'); }

        vector<T> c = multiply(a, b);

        // 处理进位
        long long carry = 0;
        for (int i = 0; i < c.size(); i++) {
            long long val = (long long)round(c[i]) + carry;
            carry = val / 10;
            c[i] = val % 10;
        }

        while (carry) {
            c.push_back(carry % 10);
            carry /= 10;
        }

        // 去除前导零
        while (c.size() > 1 && c.back() == 0) { c.pop_back(); }

        string result;
        for (int i = c.size() - 1; i >= 0; i--) { result += char('0' + (int)round(c[i])); }

        return result;
    }

    // 多项式求逆（模x^n）
    vector<Complex> polynomial_inverse(vector<Complex>& a, int n) {
        if (n == 1) { return {Complex(1.0 / a[0].real)}; }

        vector<Complex> b = polynomial_inverse(a, (n + 1) / 2);

        int size = 1;
        while (size < 2 * n) size <<= 1;

        vector<Complex> fa(a.begin(), a.begin() + min(n, (int)a.size()));
        fa.resize(size);
        b.resize(size);

        fft(fa);
        fft(b);

        for (int i = 0; i < size; i++) { fa[i] = b[i] * (Complex(2) - fa[i] * b[i]); }

        fft(fa, true);
        fa.resize(n);

        return fa;
    }

    // 多项式除法
    pair<vector<T>, vector<T>> polynomial_division(vector<T>& dividend, vector<T>& divisor) {
        int n = dividend.size(), m = divisor.size();
        if (n < m) { return {{0}, dividend}; }

        // 计算商
        vector<T> rev_dividend(dividend.rbegin(), dividend.rend());
        vector<T> rev_divisor(divisor.rbegin(), divisor.rend());

        rev_dividend.resize(n - m + 1);
        rev_divisor.resize(n - m + 1);

        vector<T> quotient = multiply(rev_dividend, rev_divisor);
        quotient.resize(n - m + 1);
        reverse(quotient.begin(), quotient.end());

        // 计算余数
        vector<T> product = multiply(quotient, divisor);
        vector<T> remainder(max(0, m - 1));

        for (int i = 0; i < remainder.size(); i++) {
            remainder[i] = dividend[i] - (i < product.size() ? product[i] : 0);
        }

        // 去除前导零
        while (remainder.size() > 1 && remainder.back() == 0) { remainder.pop_back(); }

        return {quotient, remainder};
    }

    // 点值插值（拉格朗日插值）
    vector<T> lagrange_interpolation(vector<pair<T, T>>& points) {
        int n = points.size();
        vector<T> result(n, 0);

        for (int i = 0; i < n; i++) {
            vector<T> term = {1};
            T denominator = 1;

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    // term *= (x - points[j].first)
                    vector<T> factor = {-points[j].first, 1};
                    term = multiply(term, factor);

                    denominator *= (points[i].first - points[j].first);
                }
            }

            // term *= points[i].second / denominator
            for (T& coeff : term) { coeff *= points[i].second / denominator; }

            // result += term
            result.resize(max(result.size(), term.size()));
            for (int k = 0; k < term.size(); k++) { result[k] += term[k]; }
        }

        return result;
    }
};

// 使用示例
/*
FFT<double> fft;

// 多项式乘法
vector<double> a = {1, 2, 3};  // 1 + 2x + 3x^2
vector<double> b = {4, 5};     // 4 + 5x
vector<double> c = fft.multiply(a, b);  // 结果: 4 + 13x + 22x^2 + 15x^3

// 大整数乘法
string result = fft.multiply_bigint("123456789", "987654321");
cout << result << endl;
*/
]=]),

-- 04_Math\Polynomial\NTT.h
ps("04_math_polynomial_ntt_h", [=[

// 数论变换 (NTT) 模板
template <long long MOD = 998244353, long long G = 3>
struct NTT {
    static constexpr long long mod = MOD;
    static constexpr long long g = G;

    // 快速幂
    long long power(long long a, long long b, long long m = mod) {
        long long res = 1;
        a %= m;
        while (b > 0) {
            if (b & 1) res = res * a % m;
            a = a * a % m;
            b >>= 1;
        }
        return res;
    }

    // 位逆序置换
    void bit_reverse(vector<long long>& a) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) { j ^= bit; }
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
    }

    // NTT主函数
    void ntt(vector<long long>& a, bool invert = false) {
        bit_reverse(a);
        int n = a.size();

        for (int len = 2; len <= n; len <<= 1) {
            long long wlen = power(g, (mod - 1) / len);
            if (invert) wlen = power(wlen, mod - 2);

            for (int i = 0; i < n; i += len) {
                long long w = 1;
                for (int j = 0; j < len / 2; j++) {
                    long long u = a[i + j];
                    long long v = a[i + j + len / 2] * w % mod;
                    a[i + j] = (u + v) % mod;
                    a[i + j + len / 2] = (u - v + mod) % mod;
                    w = w * wlen % mod;
                }
            }
        }

        if (invert) {
            long long inv_n = power(n, mod - 2);
            for (long long& x : a) { x = x * inv_n % mod; }
        }
    }

    // 多项式乘法
    vector<long long> multiply(vector<long long> a, vector<long long> b) {
        if (a.empty() || b.empty()) return {};

        int n = 1;
        while (n < a.size() + b.size()) n <<= 1;

        a.resize(n);
        b.resize(n);

        ntt(a);
        ntt(b);

        for (int i = 0; i < n; i++) { a[i] = a[i] * b[i] % mod; }

        ntt(a, true);

        vector<long long> result(a.size() + b.size() - 1);
        for (int i = 0; i < result.size(); i++) { result[i] = a[i]; }

        // 去除前导零
        while (result.size() > 1 && result.back() == 0) { result.pop_back(); }

        return result;
    }

    // 多项式快速幂
    vector<long long> polynomial_power(vector<long long> a, long long k, int n) {
        vector<long long> result = {1};
        a.resize(n);

        while (k > 0) {
            if (k & 1) {
                result = multiply(result, a);
                result.resize(n);
            }
            a = multiply(a, a);
            a.resize(n);
            k >>= 1;
        }

        return result;
    }

    // 多项式求逆（模x^n）
    vector<long long> polynomial_inverse(vector<long long> a, int n) {
        if (n == 1) { return {power(a[0], mod - 2)}; }

        vector<long long> b = polynomial_inverse(a, (n + 1) / 2);

        int size = 1;
        while (size < 2 * n) size <<= 1;

        vector<long long> fa(a.begin(), a.begin() + min(n, (int)a.size()));
        fa.resize(size);
        b.resize(size);

        ntt(fa);
        ntt(b);

        for (int i = 0; i < size; i++) { fa[i] = b[i] * (2 - fa[i] * b[i] % mod + mod) % mod; }

        ntt(fa, true);
        fa.resize(n);

        return fa;
    }

    // 多项式除法
    pair<vector<long long>, vector<long long>> polynomial_division(vector<long long> dividend,
                                                                   vector<long long> divisor) {
        int n = dividend.size(), m = divisor.size();
        if (n < m) { return {{0}, dividend}; }

        // 计算商
        vector<long long> rev_dividend(dividend.rbegin(), dividend.rend());
        vector<long long> rev_divisor(divisor.rbegin(), divisor.rend());

        vector<long long> inv_divisor = polynomial_inverse(rev_divisor, n - m + 1);
        vector<long long> quotient = multiply(rev_dividend, inv_divisor);
        quotient.resize(n - m + 1);
        reverse(quotient.begin(), quotient.end());

        // 计算余数
        vector<long long> product = multiply(quotient, divisor);
        vector<long long> remainder(max(0, m - 1));

        for (int i = 0; i < remainder.size(); i++) {
            remainder[i] = (dividend[i] - (i < product.size() ? product[i] : 0) + mod) % mod;
        }

        // 去除前导零
        while (remainder.size() > 1 && remainder.back() == 0) { remainder.pop_back(); }

        return {quotient, remainder};
    }

    // 多项式对数
    vector<long long> polynomial_log(vector<long long> a, int n) {
        if (a[0] != 1) return {};  // 要求常数项为1

        // ln(a) = ∫(a'/a)dx
        vector<long long> a_derivative(n - 1);
        for (int i = 1; i < n; i++) { a_derivative[i - 1] = a[i] * i % mod; }

        vector<long long> a_inv = polynomial_inverse(a, n - 1);
        vector<long long> result = multiply(a_derivative, a_inv);
        result.resize(n - 1);

        // 积分
        vector<long long> log_a(n);
        for (int i = 1; i < n; i++) { log_a[i] = result[i - 1] * power(i, mod - 2) % mod; }

        return log_a;
    }

    // 多项式指数
    vector<long long> polynomial_exp(vector<long long> a, int n) {
        if (a[0] != 0) return {};  // 要求常数项为0

        vector<long long> result = {1};

        for (int len = 1; len < n; len <<= 1) {
            vector<long long> log_result = polynomial_log(result, 2 * len);

            for (int i = 0; i < 2 * len; i++) { log_result[i] = (a[i] - log_result[i] + mod) % mod; }
            log_result[0] = (log_result[0] + 1) % mod;

            result = multiply(result, log_result);
            result.resize(2 * len);
        }

        result.resize(n);
        return result;
    }

    // 多项式开根
    vector<long long> polynomial_sqrt(vector<long long> a, int n) {
        if (a[0] != 1) return {};  // 这里简化为常数项为1的情况

        vector<long long> result = {1};
        long long inv2 = power(2, mod - 2);

        for (int len = 1; len < n; len <<= 1) {
            vector<long long> inv_result = polynomial_inverse(result, 2 * len);

            vector<long long> temp(a.begin(), a.begin() + min(2 * len, (int)a.size()));
            temp.resize(2 * len);

            temp = multiply(temp, inv_result);
            temp.resize(2 * len);

            for (int i = 0; i < 2 * len; i++) { temp[i] = (temp[i] + result[i]) * inv2 % mod; }

            result = temp;
        }

        result.resize(n);
        return result;
    }

    // 多点求值
    vector<long long> multipoint_evaluation(vector<long long>& poly, vector<long long>& points) {
        int n = points.size();
        if (n == 0) return {};

        // 构建子积树
        vector<vector<vector<long long>>> tree(4 * n);
        function<void(int, int, int)> build = [&](int v, int tl, int tr) {
            if (tl == tr) {
                tree[v] = {(mod - points[tl]) % mod, 1};
            } else {
                int tm = (tl + tr) / 2;
                build(2 * v, tl, tm);
                build(2 * v + 1, tm + 1, tr);
                tree[v] = multiply(tree[2 * v], tree[2 * v + 1]);
            }
        };

        build(1, 0, n - 1);

        // 递归求值
        vector<long long> result(n);
        function<void(int, int, int, vector<long long>)> evaluate =
            [&](int v, int tl, int tr, vector<long long> remainder) {
                if (tl == tr) {
                    result[tl] = remainder.empty() ? 0 : remainder[0];
                } else {
                    int tm = (tl + tr) / 2;

                    auto [q1, r1] = polynomial_division(remainder, tree[2 * v]);
                    auto [q2, r2] = polynomial_division(remainder, tree[2 * v + 1]);

                    evaluate(2 * v, tl, tm, r1);
                    evaluate(2 * v + 1, tm + 1, tr, r2);
                }
            };

        evaluate(1, 0, n - 1, poly);
        return result;
    }
};

using NTT998 = NTT<998244353, 3>;
using NTT1004 = NTT<1004535809, 3>;
using NTT469 = NTT<469762049, 3>;

// 使用示例
/*
NTT998 ntt;

// 多项式乘法
vector<long long> a = {1, 2, 3};  // 1 + 2x + 3x^2
vector<long long> b = {4, 5};     // 4 + 5x
vector<long long> c = ntt.multiply(a, b);  // 结果: 4 + 13x + 22x^2 + 15x^3

// 多项式求逆
vector<long long> inv_a = ntt.polynomial_inverse(a, 10);
*/
]=]),

}
