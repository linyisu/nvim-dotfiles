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
            { i(1, "i"), i(2, "1"), rep(1), i(3), rep(1), c(4, { t "++", t "--", t"" }) })),
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

    p(".write", {
        f(function(_, parent)
            local var = parent.snippet.env.POSTFIX_MATCH
            return 'for (int {} = 0; _ < ' .. var .. '.size(); _ ++) cerr << ' .. var .. '[_] << " "; cerr << \'\\n\';'
        end, {}),
    }),
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

    -- 无穷大
    s("inf", { t("const int INF = "), i(1, "0x3f3f3f3f"), t(";") }),

    -- lambda函数
    s("lambda", { t("auto "), i(1, "f"), t(" = [&]("), i(2, "int x"), t(") -> "), i(3, "int"), t(" {"), i(4), t("};") }),

    -- 结构体
    s("struct", {
        t("struct "), i(1, "node"), t({
        " {",
        "    ",
    }), i(2),
        t({
            "",
            "};"
        })
    }),

    -- p(".db", {
    --     f(function(_, parent)
    --         local var = parent.snippet.env.POSTFIX_MATCH
    --         return 'cerr << "' .. var .. ' = " << ' .. var .. ' << \'\\n\';'
    --     end, {}),
    -- }),
    -- p(".dbv", {
    --     f(function(_, parent)
    --         local var = parent.snippet.env.POSTFIX_MATCH
    --         return 'for (size_t _ = 0; _ < ' .. var .. '.size(); _ ++) cerr << ' .. var .. '[_] << " "; cerr << \'\\n\';'
    --     end, {}),
    -- }),
    -- p(".dbv1", {
    --     f(function(_, parent)
    --         local var = parent.snippet.env.POSTFIX_MATCH
    --         return 'for (size_t _ = 1; _ < ' .. var .. '.size(); _ ++) cerr << ' .. var .. '[_] << " "; cerr << \'\\n\';'
    --     end, {}),
    -- }),
}
