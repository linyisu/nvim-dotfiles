local map = vim.keymap.set
local opts = { noremap = true, silent = true }

map("n", "<Leader>m", function() Snacks.picker.zoxide() end, { desc = "Open zoxide picker" })
map("n", "<Leader>rn", function()
  for _, client in ipairs(vim.lsp.get_clients { bufnr = 0 }) do
    if client:supports_method("textDocument/rename", 0) then
      vim.lsp.buf.rename()
      return
    end
  end
  vim.notify("No LSP client supports rename in this buffer", vim.log.levels.WARN)
end, { desc = "Rename symbol" })

map({ "n", "v" }, "<A-Up>", "<A-k>", { remap = true, silent = true })
map({ "n", "v" }, "<A-Down>", "<A-j>", { remap = true, silent = true })

if vim.g.neovide then
  vim.g.neovide_scale_factor = 1.0
  map({ "n", "v", "i" }, "<C-ScrollWheelUp>",
    function() vim.g.neovide_scale_factor = vim.g.neovide_scale_factor * 1.1 end, opts)
  map({ "n", "v", "i" }, "<C-ScrollWheelDown>",
    function() vim.g.neovide_scale_factor = vim.g.neovide_scale_factor / 1.1 end, opts)
end

map("n", "<C-Up>", ":resize -2<CR>", opts)
map("n", "<C-Down>", ":resize +2<CR>", opts)
map("n", "<C-Left>", ":vertical resize -2<CR>", opts)
map("n", "<C-Right>", ":vertical resize +2<CR>", opts)

map("n", "<Leader>cpr", function()
  require("competitest.receive").stop_receiving()
  vim.cmd("CompetiTest receive problem")
end, { desc = "CompetiTest receive problem" })
map("n", "<Leader>cpc", function()
  require("competitest.receive").stop_receiving()
  vim.cmd("CompetiTest receive contest")
end, { desc = "CompetiTest receive contest" })
map("n", "<Leader>cpt", "<Cmd>CompetiTest run<CR>", { desc = "CompetiTest run" })
map("n", "<Leader>cpu", "<Cmd>CompetiTest show_ui<CR>", { desc = "CompetiTest show UI" })
map("n", "<Leader>tt", "<Cmd>CompetiTest run<CR>", { desc = "CompetiTest run" })
map("n", "<Leader>cpg", "<Cmd>CompetiTest receive testcases<CR>", { desc = "CompetiTest receive testcases" })
