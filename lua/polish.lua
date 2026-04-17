-- This will run last in the setup process.
-- This is just pure lua so anything that doesn't
-- fit in the normal config locations above can go here

-- 设置 shell 为 NuShell
vim.opt.shell = "nu"
vim.opt.shellcmdflag = "-c"
vim.opt.shellquote = ""
vim.opt.shellxquote = ""
vim.opt.shellslash = true

-- 延迟自动保存（不触发 BufWrite* 自动命令），因此不会触发 format_on_save。
-- 手动 :w 仍会正常触发格式化。
local autosave_timer = nil
local autosave_group = vim.api.nvim_create_augroup("AutoSave", { clear = true })

local function delayed_autosave()
	if autosave_timer then vim.fn.timer_stop(autosave_timer) end
	autosave_timer = vim.fn.timer_start(500, function()
		if vim.bo.modifiable and not vim.bo.readonly and vim.bo.modified and vim.fn.expand "%" ~= "" then
			vim.cmd "silent! noautocmd update"
		end
		autosave_timer = nil
	end)
end

vim.api.nvim_create_autocmd({ "TextChanged", "TextChangedI", "InsertLeave" }, {
	group = autosave_group,
	pattern = "*",
	callback = delayed_autosave,
})
