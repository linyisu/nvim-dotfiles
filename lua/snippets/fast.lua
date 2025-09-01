local ls = require("luasnip") -- 引入 LuaSnipfa

-- 定义一些快捷函数
local ls = require("luasnip")
local fmt = require("luasnip.extras.fmt").fmt
local rep = require("luasnip.extras").rep
local s = ls.snippet
local sn = ls.snippet_node
local t = ls.text_node
local i = ls.insert_node
local f = ls.function_node
local c = ls.choice_node
local d = ls.dynamic_node
local ps = ls.parser.parse_snippet
local p = require("luasnip.extras.postfix").postfix
-- local same = function(index) return f(function(arg) return arg[1] end, { index }) end

return {
    s("sort", fmt([[sort({}.begin(), {}.end());]], { i(1), rep(1) })),
    s("uni", fmt([[sort({}.begin(), {}.end());
{}.erase(unique({}.begin(), {}.end()), {}.end());]], { i(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),
    s("lsh", fmt([[sort({}.begin(), {}.end());
{}.erase(unique({}.begin(), {}.end()), {}.end());
auto getRank = [&](int x) -> int {{ return lower_bound({}.begin(), {}.end(), x) - {}.begin() + 1; }};]],
        { i(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),

    s("sort1", fmt([[sort({}.begin() + 1, {}.end());]], { i(1), rep(1) })),
    s("uni1", fmt([[sort({}.begin() + 1, {}.end());
{}.erase(unique({}.begin() + 1, {}.end()), {}.end());]], { i(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),
    s("lsh1", fmt([[sort({}.begin() + 1, {}.end());
{}.erase(unique({}.begin() + 1, {}.end()), {}.end());
auto getRank = [&](int x) -> int {{ return lower_bound({}.begin() + 1, {}.end(), x) - {}.begin(); }};]],
        { i(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),

    s("all", d(1, function(_, parent)
        local var = parent.snippet.env.POSTFIX_MATCH
        return sn(nil, fmt("{}.begin(), {}.end()", { i(1, var), rep(1) }))
    end, {})),
    s("rall", d(1, function(_, parent)
        local var = parent.snippet.env.POSTFIX_MATCH
        return sn(nil, fmt("{}.rbegin(), {}.rend()", { i(1, var), rep(1) }))
    end, {})),
    s("all1", d(1, function(_, parent)
        local var = parent.snippet.env.POSTFIX_MATCH
        return sn(nil, fmt("{}.begin() + 1, {}.end()", { i(1, var), rep(1) }))
    end, {})),

    s("sum", d(1, function(_, parent)
        local var = parent.snippet.env.POSTFIX_MATCH
        return sn(nil, fmt("accumulate({}.begin(), {}.end(), {})", { i(1, var), rep(1), i(2, "0ll") }))
    end, {})),
    s("sum1", d(1, function(_, parent)
        local var = parent.snippet.env.POSTFIX_MATCH
        return sn(nil, fmt("accumulate({}.begin() + 1, {}.end(), {})", { i(1, var), rep(1), i(2, "0ll") }))
    end, {})),

    s("for",
        fmt([[for (int {} = {}; {} {}; {} {})]],
            { i(1, "i"), i(2, "1"), rep(1), i(3), rep(1), c(4, { t "++", t "--", t "" }) })),
    s("mod", fmt("const int MOD = {};", { c(1, { t "1e9 + 7", t "998244353", t "998244383", t "" }) })),
    s("yr", fmt([[cout << "{}\n";
return;]], { c(1, { t "Yes", t "YES" }) })),
    s("nr", fmt([[cout << "{}\n";
return;]], { c(1, { t "No", t "NO" }) })),
    s("yn", fmt([[cout << ({} ? "{}" : "{}") << "\n";]], {
        i(1, "ok"),
        c(2, { t "Yes", t "YES", t "No", t "NO" }),
        f(function(args)
            local input_text = args[1] and args[1][1] or ""
            local clean_input = string.gsub(input_text, "^%s*(.-)%s*$", "%1")
            if clean_input == "Yes" then
                return "No"
            elseif clean_input == "YES" then
                return "NO"
            elseif clean_input == "No" then
                return "Yes"
            elseif clean_input == "NO" then
                return "YES"
            else
                return input_text
            end
        end, { 2 })
    })),

    s("opxy", fmt(
        [[int opx[] = {{{}}};
int opy[] = {{{}}};
        ]],
        {
            i(1, "-1, 0, 1, 0"),
            f(function(args)
                local input_text = args[1] and args[1][1] or ""
                local clean_input = string.gsub(input_text, "^%s*(.-)%s*$", "%1")
                if clean_input == "-1, 0, 1, 0" then
                    return "0, -1, 0, 1"
                elseif clean_input == "0, -1, 0, 1" then
                    return "-1, 0, 1, 0"
                elseif clean_input == "1, 0, -1, 0" then
                    return "0, 1, 0, -1"
                elseif clean_input == "0, 1, 0, -1" then
                    return "1, 0, -1, 0"
                elseif clean_input == "-1, -1, 1, 1" then
                    return "-1, 1, -1, 1"
                elseif clean_input == "1, 1, -1, -1" then
                    return "-1, 1, -1, 1"
                elseif clean_input == "-1, -1, -1, 0, 0, 1, 1, 1" then
                    return "-1, 0, 1, -1, 1, -1, 0, 1"
                elseif clean_input == "-2, -2, -1, -1, 1, 1, 2, 2" then
                    return "-1, 1, -2, 2, -2, 2, -1, 1"
                else
                    return input_text
                end
            end, { 1 })
        }
    )),

    ps("qpow", [[
long long qpow(long long a, long long b) {
    long long t = 1;
    while (b) {
        if (b & 1) t = t * a;
        a = a * a;
        b >>= 1;
    }
    return t;
}
    ]]),
    ps("qpowM", [[
long long qpow(long long a, long long b) {
    long long t = 1;
    a %= MOD;
    while (b) {
        if (b & 1) t = t * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return t;
}
long long inv(long long x) { return qpow(x, MOD - 2);}
    ]]),
    ps("exgcd", [[
long long exgcd(long long a, long long b, long long& x, long long& y) {
	if (!b) { x = 1, y = 0; return a; }
	long long g = exgcd(b, a % b, x, y);
	tie(x, y) = tuple(y, x - a / b * y);
	return g;
}
long long inv(long long x) { long long p, q, g = exgcd(x, MOD, p, q); return g - 1 ? -1 : (p % MOD + MOD) % MOD; }
    ]]),
    ps("euler", [[
long long Phi(long long n) {
    long long res = n;
    for (int i = 2; i <= n / i; i ++)
        if (!(n % i)) {
            res = res / i * (i - 1);
            while (!(n % i))
                n /= i;
        }
    if (n > 1)
        res = res / n * (n - 1);
    return res;
}
    ]]),
    ps("eulersieve", [[
const int N = 1e5 + 5;
vector<int> primes, phi(N), pre(N);
vector<bool> isPrime(N, true);
void euler_sieve() {
    isPrime[0] = isPrime[1] = false;
    phi[1] = pre[1] = 1;
    for (int i = 2; i < N; i ++) {
        if (isPrime[i]) {
            primes.emplace_back(i);
            phi[i] = i - 1;
        }
        for (auto prime : primes) {
            if (i * prime >= N)
                break;
            isPrime[i * prime] = false;
            if (i % prime)
                phi[i * prime] = phi[i] * (prime - 1);
            else {
                phi[i * prime] = phi[i] * prime;
                break;
            }
        }
        pre[i] = pre[i - 1] + phi[i];
    }
}
    ]]),

    ps("qr", [[
void rd() {}
void wr() {}
void _wr(char c) { putchar(c); }
void _wr(const string &s) { fputs(s.c_str(), stdout); }
void _wr(const char *s) { fputs(s, stdout); }
template<typename T> void _rd(T &x) { x = 0; char ch = getchar(); int f = 1; while (ch < '0' || ch > '9') { if (ch == '-') f = -1; ch = getchar(); } while (ch >= '0' && ch <= '9') { x = x * 10 + ch - '0'; ch = getchar(); } x *= f; }
void _rd(double &x) { x = 0; char ch = getchar(); int f = 1; while (ch < '0' || ch > '9') { if (ch == '-') f = -1; ch = getchar(); } while (ch >= '0' && ch <= '9') { x = x * 10 + ch - '0'; ch = getchar(); } if (ch == '.') { double base = 0.1; ch = getchar(); while (ch >= '0' && ch <= '9') { x += base * (ch - '0'); base *= 0.1; ch = getchar(); } } x *= f; }
void _rd(char &c) { c = getchar(); while (isspace(c)) c = getchar(); }
void _rd(string &s) { s.clear(); char ch = getchar(); while (isspace(ch)) ch = getchar(); while (!isspace(ch) && ch != EOF) { s += ch; ch = getchar(); } }
template<typename T, typename... Args> void rd(T &x, Args&... args) { _rd(x); rd(args...); }
template<typename T> void _wr(T x) { if (x < 0) { putchar('-'); x = -x; } if (x == 0) { putchar('0'); return; } char buf[21]; int i = 0; while (x > 0) { buf[i++] = x % 10 + '0'; x /= 10; } while (i-- > 0) putchar(buf[i]); }
void _wr(double x, int precision = 6) { if (x < 0) { putchar('-'); x = -x; } long long int_part = (long long)x; _wr(int_part); putchar('.'); double dec_part = x - int_part; for (int i = 0; i < precision; i++) { dec_part *= 10; putchar((int)dec_part + '0'); dec_part -= (int)dec_part; } }
template<typename T, typename... Args> void wr(T x, Args... args) { _wr(x); if (sizeof...(args) > 0) { putchar(' '); } wr(args...); }
template<typename... Args> void wrln(Args... args) { wr(args...); putchar('\n'); }
    ]]),

    s("dbg", {
        t({
            "#ifndef ONLINE_JUDGE",
            "#include \"114514.h\"",
            "#else",
            "#define dbg(...) ((void)114514)",
            "#endif",
        }),
    }),

    s("fi", { t({ "first" }) }),
    s("se", { t({ "second" }) }),

    s("rng", { t("mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());"), }),

    s("vi", fmt([[vector<{}> {}{}]], { i(1, "int"), i(2, "v"),
        c(3, { sn(nil, { t("("), i(1, "n + 1"), t(");"), }), t(";"), t(""), }), })),
    s("vvi", { c(1, {
        sn(nil,
            { t("vector<vector<"), i(1, "int"), t(">> "), i(2, "v"), t("("), i(3, "n + 1"), t(", vector<"), f(
                function(args) return args[1][1] or "" end, { 1 }), t(">("), i(4, "m + 1"), t("));") }),
        sn(nil, { t("vector<vector<"), i(1, "int"), t(">> "), i(2, "v"), t("("), i(3, "n + 1"), t(");") }), }), }),

    s("pii", { t("pair<int, int>") }),
    -- s("vvi",
    --     { t("vector<vector<int>> "), i(1, "v"), t("("), i(2, "n + 1"), t(", vector<int>("), i(3, "m + 1"), t("));") }),
    s("pqi", { t("priority_queue<int> "), i(1), t(";") }),

    -- 输出多个变量
    s("write", { t("cout << "), i(1), t(" << '\\n';") }),

    -- vector<string>
    s("vs", { t("vector<string> "), i(1), t(";") }),

    -- map<int, int>
    s("mii", { t("map<int, int> "), i(1), t(";") }),

    -- set<int>
    s("si", { t("set<int> "), i(1), t(";") }),

    -- 常用常数定义
    s("const", { t("const int "), i(1, "N"), t(" = "), i(2, "1e5"), t(";") }),

    s("inf", { t("const int INF = "), i(1, "0x3f3f3f3f"), t(";") }),

    s("lambda", { t("auto "), i(1, "f"), t(" = [&]("), i(2, "int x"), t(") -> "), i(3, "int"), t(" {"), i(4), t("};") }),
    s("struct", {
        t("struct "), i(1, "node"), t({
        "",
        "{",
        "    ",
    }), i(2),
        t({
            "",
            "};"
        })
    }),
}
