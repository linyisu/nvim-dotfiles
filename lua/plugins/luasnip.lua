return {
    {
        "L3MON4D3/LuaSnip",
        version = "v2.*",                -- 使用最新的 v2 版本
        build = "make install_jsregexp", -- 安装 jsregexp（可选）

        -- 插件加载完成后，执行以下配置
        config = function()
            -- 使用统一的 snippets 管理器加载所有代码片段
            local snippet_manager = require("snippets")
            snippet_manager.load_all_snippets()

            -- 加载 snippets 管理工具
            require("snippets.manager")

            -- 可选：设置 LuaSnip 的其他配置
            local ls = require("luasnip")

            -- 设置快捷键映射
            vim.keymap.set({ "i", "s" }, "<C-S-j>", function() ls.jump(1) end, { silent = true })
            vim.keymap.set({ "i", "s" }, "<C-S-k>", function() ls.jump(-1) end, { silent = true })
            vim.keymap.set({ "i", "s" }, "<C-S-h>", function()
                if ls.choice_active() then
                    ls.change_choice(1)
                end
            end, { silent = true })
        end
    }
}

