-- 文件位置: 你的 Neovim 配置中，例如 ~/.config/nvim/lua/custom_actions.lua
-- 然后在你的 init.lua 中通过 require('custom_actions') 来加载它

-- 处理器函数
-- @param params table LSP codeAction/context
-- @param bufnr number 缓冲区编号
local function create_copyright_action_handler(bufnr)
    return function(params)
        -- 这里可以添加更多逻辑，比如检查文件是否已经有版权头了
        -- for simplicity, we always offer the action.

        local copyright_header =
        [[// Copyright (C) 2025 My Company
// All rights reserved.
]]

        local text_edit = {
            range = {
                start = { line = 0, character = 0 },
                ['end'] = { line = 0, character = 0 },
            },
            newText = copyright_header,
        }

        local workspace_edit = {
            changes = {
                [vim.uri_from_bufnr(bufnr)] = { text_edit },
            },
        }

        local action = {
            title = '添加公司版权头 (自定义)',
            kind = 'refactor.rewrite', -- 使用更具体的类型
            edit = workspace_edit,
        }

        -- 返回一个列表
        return { action }
    end
end

-- 使用 LspAttach 事件来注册我们的处理器
local augroup = vim.api.nvim_create_augroup('MyCustomLspActions', { clear = true })

vim.api.nvim_create_autocmd('LspAttach', {
    group = augroup,
    desc = '为C/C++文件注册自定义Code Action',
    callback = function(args)
        local bufnr = args.buf
        local client = vim.lsp.get_client_by_id(args.data.client_id)

        -- 仅为 clangd 启动的缓冲区注册
        if client and client.name == 'clangd' then
            local handler = create_copyright_action_handler(bufnr)
            vim.lsp.buf.add_code_action_handler(bufnr, handler)
            vim.notify('已为 clangd 缓冲区注册自定义 Action', vim.log.levels.TRACE)
        end
    end,
})

print('自定义 Code Action 模块已加载')
