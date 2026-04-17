local map = vim.keymap.set

-- Add your custom mappings here.
-- Example:
-- map("n", "<Leader>w", "<Cmd>w<CR>", { desc = "Save file" })

map("n", "<Leader>m", function() Snacks.picker.zoxide() end, { desc = "Open zoxide picker" })

-- Alt + j/k to move lines up and down
map("n", "<A-Up>", ":m .-2<CR>==", { desc = "Move line up" })
map("n", "<A-Down>", ":m .+1<CR>==", { desc = "Move line down" })
map("n", "<A-k>", ":m .-2<CR>==", { desc = "Move line up" })
map("n", "<A-j>", ":m .+1<CR>==", { desc = "Move line down" })

-- Typst preview
map("n", "<Leader>tp", "<Cmd>TypstPreview<CR>", { desc = "TypstPreview" })

-- start Leetcode
map("n", "<Leader>ta", "<Cmd>Leet<CR>", { desc = "Leetcode start" })
map("n", "<Leader>tr", "<Cmd>Leet run<CR>", { desc = "Leetcode run" })
map("n", "<Leader>ts", "<Cmd>Leet submit<CR>", { desc = "Leetcode submit" })
map("n", "<Leader>tl", "<Cmd>Leet lang<CR>", { desc = "Leetcode language" })
