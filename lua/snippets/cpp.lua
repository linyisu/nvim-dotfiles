local ls = require("luasnip")

-- 定义一些快捷函数
local s = ls.snippet
local t = ls.text_node
local i = ls.insert_node
local c = ls.choice_node

return {
    s("ds", {
        t({
            "#include <iostream>",
            "using namespace std;",
            "",
            "void solve() {",
            "	"
        }),
        i(1, ""),
        t({
            "",
            "}",
            "",
            "",
        }),
        t({
            "int main() {",
            "	solve();",
            "	return 0;",
            "}"
        }),
        i(0)
    }),

    s("dss", {
        t({
            "#include <iostream>",
            "using namespace std;",
            "",
            "void solve() {",
            "	"
        }),
        i(1, ""),
        t({
            "",
            "}",
            "",
            "",
        }),
        t({
            "int main() {",
            "	int t;",
            "	for (cin >> t; t --; )",
            "		solve();",
            "	return 0;",
            "}"
        }),
        i(0)
    }),

    s("normal", {
        t({
            "#include <bits/stdc++.h>",
            "#define int long long",
            "#define inf 0x3f3f3f3f3f3f3f3f",
            "#define IOS ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);",
            "using namespace std;",
            "",
            "void solve() {",
            "	"
        }),
        i(1, ""),
        t({
            "",
            "}",
            "",
            "signed main() {",
            "	IOS;",
            "	solve();",
            "	return 0;",
            "}"
        }),
        i(0)
    }),

    s("normals", {
        t({
            "#include <bits/stdc++.h>",
            "#define int long long",
            "#define inf 0x3f3f3f3f3f3f3f3f",
            "#define IOS ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);",
            "using namespace std;",
            "",
            "void solve() {",
            "	"
        }),
        i(1, ""),
        t({
            "",
            "}",
            "",
            "signed main() {",
            "	IOS;",
            "	int t;",
            "	for (cin >> t; t --; )",
            "		solve();",
            "	return 0;",
            "}"
        }),
        i(0)
    }),

    s("normalcase", {
        t({
            "#include <bits/stdc++.h>",
            "#define int long long",
            "#define inf 0x3f3f3f3f3f3f3f3f",
            "#define IOS ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);",
            "using namespace std;",
            "",
            "void solve() {",
            "	"
        }),
        i(1, ""),
        t({
            "",
            "}",
            "",
            "signed main() {",
            "	IOS;",
            "	int t;",
            "	cin >> t;",
            "	for (int i = 1; i <= t; i ++) {",
            "		cout << \"Case \" << i << \": \";",
            "		solve();",
            "	}",
            "	return 0;",
            "}"

        }),
        i(0)
    }),

    s("headers", {
        t({
            "#include <iostream>",
            "#include <algorithm>",
            "#include <string.h>",
            "#include <sstream>",
            "#include <cctype>",
            "#include <string>",
            "#include <iomanip>",
            "#include <cmath>",
            "#include <vector>",
            "#include <queue>",
            "#include <deque>",
            "#include <stack>",
            "#include <map>",
            "#include <set>",
        }),
    }),

    s("DSU", {
        t({ "struct DSU {",
            "   vector<int> f, siz;",
            "   DSU() {}",
            "   DSU(int n) {init(n);}",
            "   void init(int n) {",
            "       f.resize(n + 1);",
            "       siz.assign(n + 1, 1);",
            "       iota(f.begin(), f.end(), 0);",
            "   }",
            "   int find(int x) {",
            "       while (x != f[x]) x = f[x] = f[f[x]];",
            "       return x;",
            "   }",
            "   bool same(int x, int y) {return find(x) == find(y);}",
            "   bool merge(int x, int y) {",
            "       x = find(x);",
            "       y = find(y);",
            "       if (x == y) return false;",
            "       siz[x] += siz[y];",
            "       f[y] = x;",
            "       return true;",
            "   }",
            "   int size(int x) {return siz[find(x)];}",
            "};" })
    }),
}
