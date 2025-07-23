return {
    -- 'onsails/lspkind-nvim',
    {
        'nvimdev/lspsaga.nvim',
        config = function()
            require('lspsaga').setup({
                lightbulb = {
                    enable = false,
                    sign = true,
                    virtual_text = false,
                    debounce = 10,
                    sign_priority = 40,
                }
            })
        end,
    },

    {
        'neovim/nvim-lspconfig',
        config = function()
            local capabilities = require('cmp_nvim_lsp').default_capabilities()
            local lspconfig = require('lspconfig')

            lspconfig.clangd.setup({
                capabilities = capabilities,
            })

            lspconfig.lua_ls.setup({
                capabilities = capabilities,
                settings = {
                    Lua = {
                        workspace = { preloadFileSize = 1000000 }
                    }
                }
            })

            vim.api.nvim_create_autocmd("LspAttach", {
                callback = function(args)
                    local buf = args.buf
                    vim.api.nvim_buf_set_keymap(buf, 'n', '<C-s>', "<cmd>lua vim.lsp.buf.format({ async = true })<CR>",
                        { noremap = true, silent = true, desc = "[L]SP [F]ormat" })
                end,
            })
        end,
    },
}
