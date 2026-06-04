local M = {}

local function augroup(name)
  return vim.api.nvim_create_augroup("user_" .. name, { clear = true })
end

function M.open_directory_arg()
  if vim.fn.argc() ~= 1 then
    return
  end

  local arg = vim.fn.argv(0)
  local path = vim.fs.normalize(vim.fn.fnamemodify(arg, ":p"))

  if vim.fn.isdirectory(path) ~= 1 then
    return
  end

  pcall(vim.cmd.bwipeout)
  vim.cmd.cd(vim.fn.fnameescape(path))
  require("core.file_explorer").open(path)
end

function M.setup()
  vim.api.nvim_create_autocmd("VimEnter", {
    group = augroup("open_directory_with_mini_files"),
    callback = M.open_directory_arg,
  })
end

return M
