-- This will run last in the setup process.
-- This is just pure lua so anything that doesn't
-- fit in the normal config locations above can go here

-- 设置 shell 为 NuShell
vim.opt.shell = "nu"
vim.opt.shellcmdflag = "-c"
vim.opt.shellquote = ""
vim.opt.shellxquote = ""
vim.opt.shellslash = true

-- 自动保存所有文件
local timer_id = nil
vim.api.nvim_create_autocmd({ "TextChanged", "TextChangedI" }, {
  group = vim.api.nvim_create_augroup("AutoSave", { clear = true }),
  pattern = "*",
  callback = function()
    if timer_id then
      vim.fn.timer_stop(timer_id)
    end
    timer_id = vim.fn.timer_start(200, function()
      if vim.bo.modified then
        vim.cmd "silent! write"
      end
      timer_id = nil
    end)
  end,
})
