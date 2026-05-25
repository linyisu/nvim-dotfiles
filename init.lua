vim.g.mapleader = " "
vim.g.maplocalleader = " "

if vim.loader then
  vim.loader.enable()
end

vim.cmd.filetype("plugin", "indent", "on")
vim.cmd.syntax("enable")

require("core.options")
require("core.keymaps")
require("core.autocmds")
require("core.commands")
require("core.neovide")
require("plugins")
