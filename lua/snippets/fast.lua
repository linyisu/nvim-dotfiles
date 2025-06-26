local ls = require("luasnip") -- 引入 LuaSnip

-- 定义一些快捷函数
local s = ls.snippet
local t = ls.text_node
local i = ls.insert_node

return {
    -- vector<int>
    s("vii", { t("vector<int> "), i(1), t(";") }),

    -- pair<int, int>
    s("pii", { t("pair<int, int> "), i(1), t(";") }),
 
    -- for (auto &x : container)
    s("all", { t("for (auto &x : "), i(1), t(") "), i(2) }),

    -- rep(i, a, b)
    s("rep", { t("for (int "), i(1, "i"), t(" = "), i(2, "0"), t("; "), i(1), t(" < "), i(3, "n"), t("; ++"), i(1), t(") "), i(4) }),

    -- per(i, a, b)
    s("per", { t("for (int "), i(1, "i"), t(" = "), i(2, "n-1"), t("; "), i(1), t(" >= "), i(3, "0"), t("; --"), i(1), t(") "), i(4) }),

    -- debug(x)
    s("dbg", { t("cerr << \""), i(1, "x"), t(" = \" << "), i(1), t(" << '\\n';") }),

    -- 输出 Yes/No
    s("yesno", { t("cout << ("), i(1, "ok"), t(" ? \"Yes\" : \"No\") << '\\n';") }),

    -- 输入多个变量
    s("read", { t("cin >> "), i(1), t(";") }),

    -- 输出多个变量
    s("write", { t("cout << "), i(1), t(" << '\\n';") }),

    -- vector<long long>
    s("vll", { t("vector<long long> "), i(1), t(";") }),

    -- vector<string>
    s("vs", { t("vector<string> "), i(1), t(";") }),

    -- pair<long long, long long>
    s("pll", { t("pair<long long, long long> "), i(1), t(";") }),

    -- map<int, int>
    s("mii", { t("map<int, int> "), i(1), t(";") }),

    -- set<int>
    s("si", { t("set<int> "), i(1), t(";") }),

    -- priority_queue<int>
    s("pqi", { t("priority_queue<int> "), i(1), t(";") }),

    -- 常用常数定义
    s("const", { t("const int "), i(1, "N"), t(" = "), i(2, "1e5"), t(";") }),

    -- 模数定义
    s("mod", { t("const int MOD = "), i(1, "1e9 + 7"), t(";") }),

    -- 无穷大
    s("inf", { t("const int INF = "), i(1, "0x3f3f3f3f"), t(";") }),

    -- 二维数组
    s("arr2d", { t("vector<vector<int>> "), i(1, "arr"), t("("), i(2, "n"), t(", vector<int>("), i(3, "m"), t("));") }),

    -- 输出数组
    s("printa", { t("for (int i = 0; i < "), i(1, "n"), t("; i++) cout << "), i(2, "arr"), t("[i] << \" \"[i == "), i(1), t(" - 1];") }),

    -- 读取数组
    s("reada", { t("for (int i = 0; i < "), i(1, "n"), t("; i++) cin >> "), i(2, "arr"), t("[i];") }),

    -- lambda函数
    s("lambda", { t("auto "), i(1, "f"), t(" = [&]("), i(2, "int x"), t(") -> "), i(3, "int"), t(" {"), i(4), t("};") }),

    -- unique去重
    s("unique", { t("sort("), i(1, "arr"), t(".begin(), "), i(1), t(".end());"), t({""}, ""), i(1), t(".erase(unique("), i(1), t(".begin(), "), i(1), t(".end()), "), i(1), t(".end());") }),

    -- 二分查找
    s("bs", { t("binary_search("), i(1, "arr"), t(".begin(), "), i(1), t(".end(), "), i(2, "val"), t(")") }),

    -- 方向数组
    s("dir", {
        t({
            "int dx[] = {-1, 1, 0, 0};",
            "int dy[] = {0, 0, -1, 1};"
        })
    }),

    -- 8方向数组
    s("dir8", {
        t({
            "int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};",
            "int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};"
        })
    }),

    -- 结构体
    s("struct", {
        t("struct "), i(1, "Node"), t({
            " {",
            "    ",
        }), i(2),
        t({
            "",
            "};"
        })
    }),
}
