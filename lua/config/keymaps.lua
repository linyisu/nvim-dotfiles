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
  require("mini.pick").builtin.buffers()
end, { desc = "Find buffers" })

map("n", "<leader>fr", function()
  require("core.search").oldfiles()
end, { desc = "Recent files" })

map("n", "<leader>fh", function()
  require("mini.pick").builtin.help()
end, { desc = "Help tags" })

map("n", "<leader>fR", function()
  require("mini.pick").builtin.resume()
end, { desc = "Resume picker" })

map("n", "<leader>f/", "/", { desc = "Search buffer" })

map("n", "<leader>H", function()
  require("snacks").dashboard.open()
end, { desc = "Home page" })

map("n", "<leader>lm", "<cmd>Mason<cr>", { desc = "Manage language tools" })
map("n", "<leader>li", "<cmd>checkhealth vim.lsp<cr>", { desc = "LSP health" })
map("n", "<leader>lI", "<cmd>LspInstall<cr>", { desc = "Install LSP server" })
map("n", "<leader>lF", "<cmd>ClangFormatInit<cr>", { desc = "Create clang-format config" })
map("n", "<leader>rn", vim.lsp.buf.rename, { desc = "Rename symbol" })
map({ "n", "x" }, "<leader>la", vim.lsp.buf.code_action, { desc = "Code action" })
map("n", "<leader>ld", vim.diagnostic.open_float, { desc = "Line diagnostic" })
map({ "n", "x" }, "<leader>lf", function()
  LazyVim.format({ force = true })
end, { desc = "Format buffer" })

map("n", "<leader>rr", "<cmd>RunFile<cr>", { desc = "Run current file or Rust project" })

map("v", "<", "<gv", { desc = "Indent left" })
map("v", ">", ">gv", { desc = "Indent right" })

map("n", "<A-Up>", "<cmd>move .-2<cr>==", { desc = "Move line up" })
map("n", "<A-Down>", "<cmd>move .+1<cr>==", { desc = "Move line down" })
map("x", "<A-Up>", ":move '<-2<cr>gv=gv", { desc = "Move selection up" })
map("x", "<A-Down>", ":move '>+1<cr>gv=gv", { desc = "Move selection down" })
