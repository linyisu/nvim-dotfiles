vim.g.mapleader = " "
vim.g.maplocalleader = " "

if vim.loader then
  vim.loader.enable()
end

require("core.neovide")
require("core.startup").setup()
require("config.lazy")
