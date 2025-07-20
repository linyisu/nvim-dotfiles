return {
    {
        "L3MON4D3/LuaSnip",
        version = "v2.*",
        build = "make install_jsregexp", -- 如果你用 jsregexp

        config = function()
            local ls = require("luasnip")
            ls.config.set_config {
                history = true,
                updateevents = "TextChanged, TextChangedI",
                enabled_autosnippets = true,
            }
            ls.add_snippets("cpp", require("snippets"))
        end,
    }
}
