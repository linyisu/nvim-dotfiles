local function augroup(name)
  return vim.api.nvim_create_augroup("user_" .. name, { clear = true })
end

local c_family_filetypes = {
  c = true,
  cpp = true,
  cuda = true,
  objc = true,
  objcpp = true,
}

local function setup_c_family_buffer(buf)
  if not c_family_filetypes[vim.bo[buf].filetype] then
    return
  end

  vim.bo[buf].expandtab = true
  vim.bo[buf].shiftwidth = 4
  vim.bo[buf].softtabstop = 4
  vim.bo[buf].tabstop = 4
  require("core.clang_format").ensure_for_buffer(buf)
end

vim.api.nvim_create_autocmd("TextYankPost", {
  group = augroup("highlight_yank"),
  callback = function()
    vim.highlight.on_yank()
  end,
})

vim.api.nvim_create_autocmd("VimResized", {
  group = augroup("resize_splits"),
  command = "tabdo wincmd =",
})

vim.api.nvim_create_autocmd({ "FocusGained", "TermClose", "TermLeave" }, {
  group = augroup("checktime"),
  command = "checktime",
})

vim.api.nvim_create_autocmd("BufReadPost", {
  group = augroup("last_location"),
  callback = function(event)
    local exclude = { gitcommit = true }
    local filetype = vim.bo[event.buf].filetype

    if exclude[filetype] then
      return
    end

    local mark = vim.api.nvim_buf_get_mark(event.buf, '"')
    local line_count = vim.api.nvim_buf_line_count(event.buf)

    if mark[1] > 0 and mark[1] <= line_count then
      pcall(vim.api.nvim_win_set_cursor, 0, mark)
    end
  end,
})

vim.api.nvim_create_autocmd("FileType", {
  group = augroup("close_with_q"),
  pattern = {
    "checkhealth",
    "help",
    "lspinfo",
    "man",
    "qf",
  },
  callback = function(event)
    vim.bo[event.buf].buflisted = false
    vim.keymap.set("n", "q", "<cmd>close<cr>", {
      buffer = event.buf,
      silent = true,
      desc = "Close window",
    })
  end,
})

vim.api.nvim_create_autocmd("FileType", {
  group = augroup("cpp_indent"),
  pattern = {
    "c",
    "cpp",
    "cuda",
    "objc",
    "objcpp",
  },
  callback = function(event)
    setup_c_family_buffer(event.buf)
  end,
})

vim.api.nvim_create_autocmd("BufEnter", {
  group = augroup("cpp_apply_current_buffer"),
  callback = function(event)
    setup_c_family_buffer(event.buf)
  end,
})

for _, buf in ipairs(vim.api.nvim_list_bufs()) do
  if vim.api.nvim_buf_is_loaded(buf) then
    setup_c_family_buffer(buf)
  end
end

vim.api.nvim_create_autocmd("VimEnter", {
  group = augroup("homepage"),
  callback = function()
    if vim.fn.argc() ~= 0 then
      return
    end

    if vim.api.nvim_buf_get_name(0) ~= "" or vim.bo.buftype ~= "" then
      return
    end

    require("core.home").open()
  end,
})

vim.api.nvim_create_autocmd("VimEnter", {
  group = augroup("open_directory_with_mini_files"),
  callback = function()
    if vim.fn.argc() ~= 1 then
      return
    end

    local path = vim.fs.normalize(vim.fn.fnamemodify(vim.fn.argv(0), ":p"))

    if vim.fn.isdirectory(path) ~= 1 then
      return
    end

    pcall(vim.cmd.bwipeout)
    vim.cmd.cd(vim.fn.fnameescape(path))
    require("core.file_explorer").open(path)
  end,
})
