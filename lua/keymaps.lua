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

map("n", "<Leader>tp", function() vim.lsp.buf.execute_command({command = "tinymist.startDefaultPreview", arguments = {}}) end, { desc = "Typst Preview" })