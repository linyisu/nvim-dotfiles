return {
  {
    "L3MON4D3/LuaSnip",
    config = function(plugin, opts)
      require("astronvim.plugins.configs.luasnip")(plugin, opts)
      local ls = require("luasnip")
      local s = ls.snippet
      local sn = ls.snippet_node
      local t = ls.text_node
      local i = ls.insert_node
      local f = ls.function_node
      local c = ls.choice_node
      local d = ls.dynamic_node
      local ps = ls.parser.parse_snippet
      local fmt = require("luasnip.extras.fmt").fmt
      local rep = require("luasnip.extras").rep

      ls.add_snippets("cpp", {
        -- templates
        s("normal", {
          t({ "#include <bits/stdc++.h>", "", "void o() {", "\t" }),
          i(0),
          t({ "", "}", "", "int main() {",
            "\tstd::ios::sync_with_stdio(0), std::cin.tie(0);",
            "\to();", "\treturn 0;", "}" }),
        }),

        s("normals", {
          t({ "#include <bits/stdc++.h>", "", "void o() {", "\t" }),
          i(0),
          t({ "", "}", "", "int main() {",
            "\tstd::ios::sync_with_stdio(0), std::cin.tie(0);",
            "\tint t;", "\tfor (std::cin >> t; t--;) {", "\t\to();", "\t}",
            "\treturn 0;", "}" }),
        }),

        -- I/O
        s("cin",   { t("std::cin >> "), i(1), t(";"), i(0) }),
        s("cout",  { t("std::cout << "), i(1), t(' << "\\n";'), i(0) }),
        s("cerr",  { t("std::cerr << "), i(1), t(' << "\\n";'), i(0) }),
        s("write", { t("cout << "), i(1), t(' << "\\n";') }),

        -- containers (type-parameterized)
        s("vec",  { t("std::vector<"), i(1, "T"), t("> "), i(0) }),
        s("map",  { t("std::map<"), i(1, "K"), t(", "), i(2, "V"), t("> "), i(0) }),
        s("ump",  { t("std::unordered_map<"), i(1, "K"), t(", "), i(2, "V"), t("> "), i(0) }),
        s("set",  { t("std::set<"), i(1, "T"), t("> "), i(0) }),
        s("ust",  { t("std::unordered_set<"), i(1, "T"), t("> "), i(0) }),
        s("pq",   { t("std::priority_queue<"), i(1, "T"), t("> "), i(0) }),
        s("stk",  { t("std::stack<"), i(1, "T"), t("> "), i(0) }),
        s("que",  { t("std::queue<"), i(1, "T"), t("> "), i(0) }),
        s("dq",   { t("std::deque<"), i(1, "T"), t("> "), i(0) }),
        s("arr",  { t("std::array<"), i(1, "T"), t(", "), i(2, "N"), t("> "), i(0) }),
        s("str",  { t("std::string "), i(0) }),
        s("pair", { t("std::pair<"), i(1, "T1"), t(", "), i(2, "T2"), t("> "), i(0) }),

        -- concrete container declarations
        s("vi",  fmt([[vector<{}> {}{}]], { i(1, "int"), i(2, "v"),
          c(3, { sn(nil, { t("("), i(1, "n + 1"), t(");") }), t(";"), t("") }) })),
        s("vvi", { c(1, {
          sn(nil, { t("vector<vector<"), i(1, "int"), t(">> "), i(2, "v"), t("("), i(3, "n + 1"),
            t(", vector<"), f(function(args) return args[1][1] or "" end, { 1 }), t(">("), i(4, "m + 1"), t("));") }),
          sn(nil, { t("vector<vector<"), i(1, "int"), t(">> "), i(2, "v"), t("("), i(3, "n + 1"), t(");") }),
        }) }),
        s("vs",  { t("vector<string> "), i(1), t(";") }),
        s("pii", { t("pair<int, int>") }),
        s("pqi", { t("priority_queue<int> "), i(1), t(";") }),
        s("mii", { t("map<int, int> "), i(1), t(";") }),
        s("si",  { t("set<int> "), i(1), t(";") }),
        s("fi",  { t("first") }),
        s("se",  { t("second") }),

        -- constants
        s("mod", fmt("constexpr int mod = {};", { c(1, { t("1e9 + 7"), t("998244353"), t("") }) })),
        s("mod", fmt("constexpr int inf = {};", { c(1, { t("0x3f3f3f3f"), t("0x3f3f3f3f3f3f3f3f") }) })),
        s("const", { t("constexpr int "), i(1, "N"), t(" = "), i(2, "1e5"), t(";") }),
        s("rng", { t("mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());") }),

        -- control flow
        s("for", fmt([[for (int {} = {}; {} {}; {} {})]], {
          i(1, "i"), i(2, "1"), rep(1), i(3), rep(1),
          c(4, { t("++"), t("--"), t(" ") }),
        })),

        s("yr", fmt([[cout << "{}\n";
return;]], { c(1, { t("Yes"), t("YES") }) })),

        s("nr", fmt([[cout << "{}\n";
return;]], { c(1, { t("No"), t("NO") }) })),

        s("yn", fmt([[cout << ({} ? "{}" : "{}") << "\n";]], {
          i(1, "ok"),
          c(2, { t("Yes"), t("YES"), t("No"), t("NO") }),
          f(function(args)
            local v = args[1] and args[1][1] or ""
            if v == "Yes" then return "No"
            elseif v == "YES" then return "NO"
            elseif v == "No" then return "Yes"
            elseif v == "NO" then return "YES"
            else return v end
          end, { 2 }),
        })),

        -- lambda / struct
        s("lambda", { t("auto "), i(1, "f"), t(" = [&]("), i(2, "int x"), t(") -> "), i(3, "int"), t(" {"), i(4), t("};") }),
        s("struct", {
          t("struct "), i(1, "node"), t({ "", "{", "    " }), i(2), t({ "", "};" }),
        }),

        -- algorithms
        s("segt", fmt([[SegTree<{}, Info<{}>, Laz> seg({});]], { i(1, "int"), rep(1), i(2, "n") })),

        s("sort",  fmt([[sort({}.begin(), {}.end());]], { i(1), rep(1) })),
        s("sort1", fmt([[sort({}.begin() + 1, {}.end());]], { i(1), rep(1) })),

        s("uni", fmt([[sort({}.begin(), {}.end());
{}.erase(unique({}.begin(), {}.end()), {}.end());]], { i(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),

        s("uni1", fmt([[sort({}.begin() + 1, {}.end());
{}.erase(unique({}.begin() + 1, {}.end()), {}.end());]], { i(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),

        s("lsh", fmt([[sort({}.begin(), {}.end());
{}.erase(unique({}.begin(), {}.end()), {}.end());
auto getRank = [&](int x) -> int {{ return lower_bound({}.begin(), {}.end(), x) - {}.begin() + 1; }};]],
          { i(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),

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

        s("opxy", fmt([[int opx[] = {{{}}};
int opy[] = {{{}}};]], {
          i(1, "-1, 0, 1, 0"),
          f(function(args)
            local v = args[1] and args[1][1] or ""
            local m = {
              ["-1, 0, 1, 0"]              = "0, -1, 0, 1",
              ["0, -1, 0, 1"]              = "-1, 0, 1, 0",
              ["1, 0, -1, 0"]              = "0, 1, 0, -1",
              ["0, 1, 0, -1"]              = "1, 0, -1, 0",
              ["-1, -1, 1, 1"]             = "-1, 1, -1, 1",
              ["1, 1, -1, -1"]             = "-1, 1, -1, 1",
              ["-1, -1, -1, 0, 0, 1, 1, 1"] = "-1, 0, 1, -1, 1, -1, 0, 1",
              ["-2, -2, -1, -1, 1, 1, 2, 2"] = "-1, 1, -2, 2, -2, 2, -1, 1",
            }
            return m[v] or v
          end, { 1 }),
        })),

        -- math / number theory (parse snippets)
        ps("qpow", [[
long long qpow(long long a, long long b) {
    long long t = 1;
    while (b) {
        if (b & 1) t = t * a;
        a = a * a;
        b >>= 1;
    }
    return t;
}]]),

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
long long inv(long long x) { return qpow(x, MOD - 2); }]]),

        ps("exgcd", [[
long long exgcd(long long a, long long b, long long& x, long long& y) {
    if (!b) { x = 1, y = 0; return a; }
    long long g = exgcd(b, a % b, x, y);
    tie(x, y) = tuple(y, x - a / b * y);
    return g;
}
long long inv(long long x) { long long p, q, g = exgcd(x, MOD, p, q); return g - 1 ? -1 : (p % MOD + MOD) % MOD; }]]),

        ps("euler", [[
long long Phi(long long n) {
    long long res = n;
    for (int i = 2; i <= n / i; i++)
        if (!(n % i)) {
            res = res / i * (i - 1);
            while (!(n % i)) n /= i;
        }
    if (n > 1) res = res / n * (n - 1);
    return res;
}]]),

        ps("eulersieve", [[
const int N = 1e5 + 5;
vector<int> primes, phi(N), pre(N);
vector<bool> isPrime(N, true);
void euler_sieve() {
    isPrime[0] = isPrime[1] = false;
    phi[1] = pre[1] = 1;
    for (int i = 2; i < N; i++) {
        if (isPrime[i]) { primes.emplace_back(i); phi[i] = i - 1; }
        for (auto prime : primes) {
            if (i * prime >= N) break;
            isPrime[i * prime] = false;
            if (i % prime) phi[i * prime] = phi[i] * (prime - 1);
            else { phi[i * prime] = phi[i] * prime; break; }
        }
        pre[i] = pre[i - 1] + phi[i];
    }
}]]),

        ps("qr", [[
void rd() {}
void wr() {}
void _wr(char c) { putchar(c); }
void _wr(const string &s) { fputs(s.c_str(), stdout); }
void _wr(const char *s) { fputs(s, stdout); }
template<typename T> void _rd(T &x) { x = 0; char ch = getchar(); int f = 1; while (ch < '0' || ch > '9') { if (ch == '-') f = -1; ch = getchar(); } while (ch >= '0' && ch <= '9') { x = x * 10 + ch - '0'; ch = getchar(); } x *= f; }
void _rd(char &c) { c = getchar(); while (isspace(c)) c = getchar(); }
void _rd(string &s) { s.clear(); char ch = getchar(); while (isspace(ch)) ch = getchar(); while (!isspace(ch) && ch != EOF) { s += ch; ch = getchar(); } }
template<typename T, typename... Args> void rd(T &x, Args&... args) { _rd(x); rd(args...); }
template<typename T> void _wr(T x) { if (x < 0) { putchar('-'); x = -x; } if (x == 0) { putchar('0'); return; } char buf[21]; int i = 0; while (x > 0) { buf[i++] = x % 10 + '0'; x /= 10; } while (i-- > 0) putchar(buf[i]); }
template<typename T, typename... Args> void wr(T x, Args... args) { _wr(x); if (sizeof...(args) > 0) putchar(' '); wr(args...); }
template<typename... Args> void wrln(Args... args) { wr(args...); putchar('\n'); }]]),
      })
    end,
  },
}
