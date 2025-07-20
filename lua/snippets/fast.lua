local ls = require("luasnip") -- 引入 LuaSnip

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
local p = require("luasnip.extras.postfix").postfix
local same = function(index) return f(function(arg) return arg[1] end, { index }) end

return {
    s("sort", fmt([[sort({}.begin(), {}.end());]], { i(1), rep(1) })),
    s("unique", fmt([[sort({}.begin(), {}.end());
{}.erase(unique({}.begin(), {}.end()), {}.end());]], { i(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),
    s("lsh", fmt([[sort({}.begin(), {}.end());
{}.erase(unique({}.begin(), {}.end()), {}.end());
auto getRank = [&](int x) -> int {{ return lower_bound({}.begin(), {}.end(), x) - {}.begin() + 1; }};]], { i(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),

    s("sort1", fmt([[sort({}.begin() + 1, {}.end());]], { i(1), rep(1) })),
    s("unique1", fmt([[sort({}.begin() + 1, {}.end());
{}.erase(unique({}.begin() + 1, {}.end()), {}.end());]], { i(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),
    s("lsh1", fmt([[sort({}.begin() + 1, {}.end());
{}.erase(unique({}.begin() + 1, {}.end()), {}.end());
auto getRank = [&](int x) -> int {{ return lower_bound({}.begin() + 1, {}.end(), x) - {}.begin() + 1; }};]], { i(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1), rep(1) })),

    s("all", d(1, function(_, parent) local var = parent.snippet.env.POSTFIX_MATCH return sn(nil, fmt( "{}.begin(), {}.end()", { i(1, var), rep(1) })) end, {})),
    s("rall", d(1, function(_, parent) local var = parent.snippet.env.POSTFIX_MATCH return sn(nil, fmt( "{}.rbegin(), {}.rend()", { i(1, var), rep(1) })) end, {})),
    s("all1", d(1, function(_, parent) local var = parent.snippet.env.POSTFIX_MATCH return sn(nil, fmt( "{}.begin() + 1, {}.end()", { i(1, var), rep(1) })) end, {})),

    s("mod", fmt("const int MOD = {};", { c(1, { t"1e9 + 7", t"998244353", t"998244383" } ) })),

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
            end, {1})
        }
    )),

    -- debug(x)
    s("dbg", { t("cerr << \""), i(1, "x"), t(" = \" << "), i(1), t(" << '\\n';") }),

    -- 输出 Yes/No
    s("yesno", { t("cout << ("), i(1, "ok"), t(" ? \"Yes\" : \"No\") << '\\n';") }),

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

    -- 无穷大
    s("inf", { t("const int INF = "), i(1, "0x3f3f3f3f"), t(";") }),

    -- 二维数组
    s("arr2d", { t("vector<vector<int>> "), i(1, "arr"), t("("), i(2, "n"), t(", vector<int>("), i(3, "m"), t("));") }),

    -- lambda函数
    s("lambda", { t("auto "), i(1, "f"), t(" = [&]("), i(2, "int x"), t(") -> "), i(3, "int"), t(" {"), i(4), t("};") }),


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
