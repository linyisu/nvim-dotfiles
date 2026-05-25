local map = vim.keymap.set

map("n", "<Esc>", "<cmd>nohlsearch<cr>", { desc = "Clear search highlight" })

map("n", "<leader>e", function()
  require("core.file_explorer").toggle()
end, { desc = "Toggle files" })

map("n", "<leader>o", function()
  require("core.file_explorer").open_workspace(vim.fn.getcwd())
end, { desc = "Open cwd as workspace" })

map("n", "<leader>ff", function()
  require("core.search").pick_files()
end, { desc = "Find files" })

map("n", "<leader>fg", function()
  require("core.search").pick_grep()
end, { desc = "Search project" })

map("n", "<leader>fb", function()
  require("lazy").load({ plugins = { "mini.nvim" } })
  require("mini.pick").builtin.buffers()
end, { desc = "Find buffers" })

map("n", "<leader>fr", function()
  require("core.search").oldfiles()
end, { desc = "Recent files" })

map("n", "<leader>fh", function()
  require("lazy").load({ plugins = { "mini.nvim" } })
  require("mini.pick").builtin.help()
end, { desc = "Help tags" })

map("n", "<leader>fR", function()
  require("lazy").load({ plugins = { "mini.nvim" } })
  require("mini.pick").builtin.resume()
end, { desc = "Resume picker" })

map("n", "<leader>f/", "/", { desc = "Search buffer" })

map("n", "<leader>H", "<cmd>Home<cr>", { desc = "Home page" })

map("n", "<C-h>", "<C-w>h", { desc = "Move to left window" })
map("n", "<C-j>", "<C-w>j", { desc = "Move to lower window" })
map("n", "<C-k>", "<C-w>k", { desc = "Move to upper window" })
map("n", "<C-l>", "<C-w>l", { desc = "Move to right window" })

map("n", "<leader>wv", "<cmd>vsplit<cr>", { desc = "Vertical split" })
map("n", "<leader>ws", "<cmd>split<cr>", { desc = "Horizontal split" })
map("n", "<leader>wc", "<cmd>close<cr>", { desc = "Close window" })
map("n", "<leader>wo", "<cmd>only<cr>", { desc = "Keep only window" })

map("n", "<leader>bn", "<cmd>bnext<cr>", { desc = "Next buffer" })
map("n", "<leader>bp", "<cmd>bprevious<cr>", { desc = "Previous buffer" })
map("n", "<leader>bd", "<cmd>bdelete<cr>", { desc = "Delete buffer" })

map("n", "]q", "<cmd>cnext<cr>", { desc = "Next quickfix item" })
map("n", "[q", "<cmd>cprevious<cr>", { desc = "Previous quickfix item" })
map("n", "]l", "<cmd>lnext<cr>", { desc = "Next location item" })
map("n", "[l", "<cmd>lprevious<cr>", { desc = "Previous location item" })
map("n", "]d", function()
  vim.diagnostic.jump({ count = 1, float = true })
end, { desc = "Next diagnostic" })
map("n", "[d", function()
  vim.diagnostic.jump({ count = -1, float = true })
end, { desc = "Previous diagnostic" })

map("n", "<leader>lm", "<cmd>Mason<cr>", { desc = "Manage language tools" })
map("n", "<leader>li", "<cmd>checkhealth vim.lsp<cr>", { desc = "LSP health" })
map("n", "<leader>lI", "<cmd>LspInstall<cr>", { desc = "Install LSP server" })
map("n", "<leader>lF", "<cmd>ClangFormatInit<cr>", { desc = "Create clang-format config" })

map("n", "<leader>rr", "<cmd>RunFile<cr>", { desc = "Run current file" })

map("v", "<", "<gv", { desc = "Indent left" })
map("v", ">", ">gv", { desc = "Indent right" })

map("n", "<A-Up>", "<cmd>move .-2<cr>==", { desc = "Move line up" })
map("n", "<A-Down>", "<cmd>move .+1<cr>==", { desc = "Move line down" })
map("x", "<A-Up>", ":move '<-2<cr>gv=gv", { desc = "Move selection up" })
map("x", "<A-Down>", ":move '>+1<cr>gv=gv", { desc = "Move selection down" })

map("n", "gc", function()
  require("core.comment").toggle_current_line()
end, { desc = "Toggle comment" })

map("x", "gc", function()
  require("core.comment").toggle_selection()
end, { desc = "Toggle comment" })
