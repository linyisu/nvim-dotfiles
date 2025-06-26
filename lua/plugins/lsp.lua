return {
    -- 'onsails/lspkind-nvim',
    -- 
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
            require('lspconfig').lua.setup {
                settings = { Lua = { workspace = { preloadFileSize = 1000000 } } }
            }

            vim.api.nvim_create_autocmd("LspAttach", {
                callback = function(args)
                    local buf = args.buf
                    vim.api.nvim_buf_set_keymap(buf, 'n', '<C-l>', "<cmd>lua vim.lsp.buf.format()<CR>",
                        { noremap = true })
                end,
            })
        end,
    },
}
