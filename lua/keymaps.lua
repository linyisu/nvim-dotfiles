local opts = {
    noremap = true, -- non-recursive
    silent = true,  -- do not show message
}
vim.keymap.set('n', '<C-Up>', ':resize -2<CR>', opts)
vim.keymap.set('n', '<C-Down>', ':resize +2<CR>', opts)
vim.keymap.set('n', '<C-Left>', ':vertical resize -2<CR>', opts)
vim.keymap.set('n', '<C-Right>', ':vertical resize +2<CR>', opts)

-- buffer 相关
vim.keymap.set('n', '<C-z>', ':BufferPrevious<CR>', opts)
vim.keymap.set('n', '<C-x>', ':BufferNext<CR>', opts)


-- 在 normal 模式下，使用 <leader>j 和 <leader>k 来移动行
vim.keymap.set('n', '<M-k>', ':m .-2<CR>', opts)
vim.keymap.set('n', '<M-j>', ':m .+1<CR>', opts)
vim.keymap.set('n', '<M-Down>', ':m .+1<CR>', opts)
vim.keymap.set('n', '<M-Up>', ':m .-2<CR>', opts)

-- 在 visual 模式下，使用 <leader>j 和 <leader>k 来移动选定的行
vim.keymap.set('v', '<M-j>', ":m '>+1<CR>`<my`>mzgv`yo`z", opts)
vim.keymap.set('v', '<M-k>', ":m '<-2<CR>`>my`<mzgv`yo`z", opts)
vim.keymap.set('v', '<M-Down>', ":m '>+1<CR>`<my`>mzgv`yo`z", opts)
vim.keymap.set('v', '<M-Up>', ":m '<-2<CR>`>my`<mzgv`yo`z", opts)

-- 在 visual 模式下， <leader>j 合并选中多行成一行，换行变空格
vim.keymap.set('v', '<C-S-j>', ':s/\\%V\\s*\\n\\s*/ /g<CR>', opts)


-- 与lsp相关的
vim.keymap.set('n', '<leader>re', ':LspRestart<CR>', opts)
vim.keymap.set('n', '<leader>hh', ':Lspsaga code_action<CR>', opts)
vim.keymap.set('n', '<leader>hi', ':Lspsaga hover_doc<CR>', opts)
vim.keymap.set('n', '<leader>e', ':Lspsaga show_buf_diagnostics<CR>', opts)
vim.keymap.set('n', '<leader>rn', ':lua vim.lsp.buf.rename()<CR>', opts)

vim.keymap.set('n', '<C-b>', ':NvimTreeToggle<CR>', opts)


-- 测试点相关的
vim.keymap.set('n', '<leader>g', ':CompetiTest receive problem<CR>', opts)
vim.keymap.set('n', '<leader>c', ':CompetiTest receive contest<CR>', opts)
vim.keymap.set('n', '<leader>tt', ':CompetiTest run<CR>', opts)
vim.keymap.set('n', '<leader>tg', ':CompetiTest receive testcases<CR>', opts)
-- vim.keymap.set('n', '<leader>ta', ':CompetiTest add_testcase<CR>', opts)
-- vim.keymap.set('n', '<leader>te', ':CompetiTest edit_testcase<CR>', opts)
-- vim.keymap.set('n', '<leader>td', ':CompetiTest delete_testcase<CR>', opts)


--telescope
vim.keymap.set('n', '<leader>/f', ':Telescope find_files<CR>', opts)
vim.keymap.set('n', '<leader>//', ':Telescope live_grep<CR>', opts)
vim.keymap.set('n', '<leader>/', ':Telescope<CR>', opts)


vim.keymap.set('n', '<leader>d', ":Dashboard<CR>", opts)


-- luasnip 相关
-- 插入 & 选择模式：跳到 下/上 一个 snippet 插入点
vim.keymap.set({ "i", "s" }, "<C-j>", function() local ls = require("luasnip") if ls.expand_or_jumpable() then ls.expand_or_jump() end end, opts)
vim.keymap.set({ "i", "s" }, "<C-k>", function() local ls = require("luasnip") if ls.jumpable(-1) then ls.jump(-1) end end, opts)
vim.keymap.set({ "i", "s" }, "<C-l>", function() local ls = require("luasnip") if ls.choice_active() then ls.change_choice(1) end end, opts)
vim.keymap.set("n", "<leader><leader>s", "<Cmd>source ~/.config/nvim/lua/plugins/luasnip.lua<CR>")

vim.keymap.set('n', '<leader>x', ":BufferClose<CR>", opts)
vim.keymap.set('n', '<leader<S-x>', ":BufferCloseAllButCurrent<CR>", opts)

vim.keymap.set('n', '<F5>', ":OverseerRun<CR>", opts)

-- vim.keymap.set('v', "<C-d>", function() vim.schedule(function() require('snippets.debug').debug_print() end) end, opts)


-- 新建测试文件
vim.keymap.set("n", '<leader>v', ":e /mnt/d/code/test.cpp<CR>", opts)

if vim.g.neovide then
    vim.keymap.set({ "n", "v" }, "<C-+>", ":lua vim.g.neovide_scale_factor = vim.g.neovide_scale_factor + 0.1<CR>", opts)
    vim.keymap.set({ "n", "v" }, "<C-->", ":lua vim.g.neovide_scale_factor = vim.g.neovide_scale_factor - 0.1<CR>", opts)
    vim.keymap.set({ "n", "v" }, "<C-0>", ":lua vim.g.neovide_scale_factor = 1<CR>")
    vim.keymap.set({ "n", "v" }, "<C-ScrollWheelUp>",
        ":lua vim.g.neovide_scale_factor = vim.g.neovide_scale_factor + 0.1<CR>", opts)
    vim.keymap.set({ "n", "v" }, "<C-ScrollWheelDown>",
        ":lua vim.g.neovide_scale_factor = vim.g.neovide_scale_factor - 0.1<CR>", opts)

end
