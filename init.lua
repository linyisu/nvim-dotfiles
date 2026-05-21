local uv = vim.uv or vim.loop
local is_windows = package.config:sub(1, 1) == "\\"
local path_sep = is_windows and ";" or ":"
local lazypath = vim.env.LAZY or vim.fs.joinpath(vim.fn.stdpath "data", "lazy", "lazy.nvim")

local function same_path(left, right)
  left = vim.fs.normalize(left)
  right = vim.fs.normalize(right)
  if is_windows then
    left = left:lower()
    right = right:lower()
  end
  return left == right
end

local function path_contains(paths, path)
  for _, existing in ipairs(paths) do
    if same_path(existing, path) then return true end
  end
  return false
end

local function prepend_path(path)
  path = vim.fs.normalize(vim.fn.expand(path))
  if not uv.fs_stat(path) then return end
  local parts = vim.split(vim.env.PATH or "", path_sep, { plain = true })
  if not path_contains(parts, path) then vim.env.PATH = path .. path_sep .. (vim.env.PATH or "") end
end

prepend_path(vim.fs.joinpath(vim.fn.stdpath "data", "mason", "bin"))
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
