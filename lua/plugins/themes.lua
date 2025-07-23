return {
    {
        "folke/tokyonight.nvim",
        lazy = false,
        priority = 1000,
        opts = {
            on_highlights = function(hl, c)
                -- hl: 所有高亮组的表
                -- c: 主题的颜色调色板

                -- 自定义 MatchParen 的高亮
                hl.MatchParen = {
                    fg = c.comment,
                    bg = c.bg_highlight,
                    bold = true,
                    underline = true,
                }
            end,
        },
    },

    "RRethy/base16-nvim",
    "EdenEast/nightfox.nvim",
    "projekt0n/github-nvim-theme",
}
