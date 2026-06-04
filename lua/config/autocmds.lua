local function augroup(name)
  return vim.api.nvim_create_augroup("user_" .. name, { clear = true })
end

vim.api.nvim_create_user_command("ClangFormatInit", function(command)
  require("core.clang_format").write_default({ force = command.bang })
end, {
  bang = true,
  desc = "Create default .clang-format in the current workspace root",
})

vim.api.nvim_create_user_command("ClangFormatOpen", function()
  require("core.clang_format").open()
end, {
  desc = "Open .clang-format in the current workspace root",
})

vim.api.nvim_create_user_command("RunFile", function()
  require("core.run").current_file()
end, {
  desc = "Compile and run the current file",
})

vim.api.nvim_create_user_command("Home", function()
  require("snacks").dashboard.open()
end, {
  desc = "Open the start page",
})

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

vim.api.nvim_create_autocmd("LspAttach", {
  group = augroup("clang_format_on_attach"),
  callback = function(event)
    local client = vim.lsp.get_client_by_id(event.data.client_id)

    if client and client.name == "clangd" then
      setup_c_family_buffer(event.buf)
    end
  end,
})
