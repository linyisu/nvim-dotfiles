local shell = vim.env.SHELL
if shell and shell ~= "" and vim.fn.executable(shell) == 1 then
  vim.opt.shell = shell
end

local autosave_timers = {}
local autosave_group = vim.api.nvim_create_augroup("AutoSave", { clear = true })

local function can_autosave(bufnr)
  return vim.api.nvim_buf_is_valid(bufnr)
    and vim.bo[bufnr].modifiable
    and not vim.bo[bufnr].readonly
    and vim.bo[bufnr].modified
    and vim.api.nvim_buf_get_name(bufnr) ~= ""
end

local function autosave(bufnr)
  autosave_timers[bufnr] = nil
  if not can_autosave(bufnr) then return end

  vim.api.nvim_buf_call(bufnr, function()
    if can_autosave(bufnr) then
      vim.cmd "silent update"
    end
  end)
end

local function delayed_autosave(args)
  local bufnr = args.buf
  if autosave_timers[bufnr] then vim.fn.timer_stop(autosave_timers[bufnr]) end
  autosave_timers[bufnr] = vim.fn.timer_start(500, function() autosave(bufnr) end)
end

vim.api.nvim_create_autocmd({ "TextChanged", "InsertLeave", "BufLeave", "FocusLost" }, {
  group = autosave_group,
  pattern = "*",
  callback = delayed_autosave,
})
