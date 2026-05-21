local lazypath = vim.env.LAZY or vim.fn.stdpath "data" .. "/lazy/lazy.nvim"
local uv = vim.uv or vim.loop

local function prepend_path(path)
  path = vim.fs.normalize(vim.fn.expand(path))
  if not uv.fs_stat(path) then return end
  local parts = vim.split(vim.env.PATH or "", ":", { plain = true })
  if not vim.tbl_contains(parts, path) then vim.env.PATH = path .. ":" .. (vim.env.PATH or "") end
end

prepend_path(vim.fn.stdpath "data" .. "/mason/bin")
prepend_path "~/.local/bin"
prepend_path "~/.cargo/bin"

if not (vim.env.LAZY or uv.fs_stat(lazypath)) then
  local result = vim.fn.system {
    "git",
    "clone",
    "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    "--branch=stable",
    lazypath,
  }
  if vim.v.shell_error ~= 0 then
    vim.api.nvim_echo(
      { { ("Error cloning lazy.nvim:\n%s\n"):format(result), "ErrorMsg" }, { "Press any key to exit...", "MoreMsg" } },
      true,
      {}
    )
    vim.fn.getchar()
    vim.cmd.quit()
  end
end

vim.opt.clipboard = "unnamedplus"

vim.opt.rtp:prepend(lazypath)

if not pcall(require, "lazy") then
  vim.api.nvim_echo(
    { { ("Unable to load lazy from: %s\n"):format(lazypath), "ErrorMsg" }, { "Press any key to exit...", "MoreMsg" } },
    true,
    {}
  )
  vim.fn.getchar()
  vim.cmd.quit()
end

require "compat"
require "lazy_setup"
require "health_filters"
require "polish"
require "keymaps"
