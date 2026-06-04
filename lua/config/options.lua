vim.g.mapleader = " "
vim.g.maplocalleader = " "
vim.g.snacks_animate = false
vim.g.snacks_scroll = false

local opt = vim.opt

opt.number = true
opt.relativenumber = true
opt.signcolumn = "yes"
opt.cursorline = true

opt.expandtab = true
opt.shiftwidth = 2
opt.softtabstop = 2
opt.tabstop = 2
opt.smartindent = true
opt.breakindent = true

opt.wrap = false
opt.scrolloff = 8
opt.sidescrolloff = 8
opt.smoothscroll = false

opt.ignorecase = true
opt.smartcase = true
opt.inccommand = "split"
opt.path:append("**")
opt.wildmenu = true
opt.wildmode = { "longest:full", "full" }

opt.splitbelow = true
opt.splitright = true

opt.termguicolors = true
opt.mouse = "a"

local function has_linux_clipboard()
  return (vim.fn.executable("wl-copy") == 1 and vim.fn.executable("wl-paste") == 1)
    or vim.fn.executable("xclip") == 1
    or vim.fn.executable("xsel") == 1
    or vim.fn.executable("win32yank.exe") == 1
end

if vim.fn.has("win32") == 1 or vim.fn.has("mac") == 1 or has_linux_clipboard() then
  opt.clipboard = "unnamedplus"
else
  opt.clipboard = ""
end

opt.undofile = true
opt.swapfile = false
opt.confirm = true

opt.updatetime = 250
opt.timeoutlen = 300

opt.completeopt = { "menu", "menuone", "noselect" }
opt.list = true
opt.listchars = {
  tab = "> ",
  trail = ".",
  nbsp = "+",
}
