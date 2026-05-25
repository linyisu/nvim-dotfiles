local M = {}

local logo = {
  "██╗     ██╗███╗   ██╗██╗   ██╗██╗███████╗██╗   ██╗",
  "██║     ██║████╗  ██║╚██╗ ██╔╝██║██╔════╝██║   ██║",
  "██║     ██║██╔██╗ ██║ ╚████╔╝ ██║███████╗██║   ██║",
  "██║     ██║██║╚██╗██║  ╚██╔╝  ██║╚════██║██║   ██║",
  "███████╗██║██║ ╚████║   ██║   ██║███████║╚██████╔╝",
  "╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝ ╚═════╝ ",
}

local function center(text)
  local width = vim.api.nvim_win_get_width(0)
  local padding = math.max(math.floor((width - vim.fn.strdisplaywidth(text)) * 0.5), 0)
  return string.rep(" ", padding) .. text
end

local function load_mini()
  pcall(function()
    require("lazy").load({ plugins = { "mini.nvim" } })
  end)
end

local function can_reuse_current_buffer()
  local buf = vim.api.nvim_get_current_buf()

  if vim.api.nvim_buf_get_name(buf) ~= "" or vim.bo[buf].buftype ~= "" or vim.bo[buf].modified then
    return false
  end

  local lines = vim.api.nvim_buf_get_lines(buf, 0, -1, false)
  return #lines == 1 and lines[1] == ""
end

local function set_home_options(buf)
  vim.bo[buf].bufhidden = "wipe"
  vim.bo[buf].buflisted = false
  vim.bo[buf].buftype = "nofile"
  vim.bo[buf].filetype = "nvim-home"
  vim.bo[buf].modifiable = false
  vim.bo[buf].swapfile = false

  vim.wo.colorcolumn = ""
  vim.wo.cursorline = true
  vim.wo.foldcolumn = "0"
  vim.wo.list = false
  vim.wo.number = false
  vim.wo.relativenumber = false
  vim.wo.signcolumn = "no"
  vim.wo.wrap = false
end

local function action_open_files()
  load_mini()
  require("core.file_explorer").open_workspace(vim.fn.getcwd())
end

local function action_find_files()
  load_mini()
  require("core.search").pick_files()
end

local function action_search_text()
  load_mini()
  require("core.search").pick_grep()
end

local function action_recent_files()
  load_mini()

  local ok, extra = pcall(require, "mini.extra")

  if ok and extra.pickers and extra.pickers.oldfiles then
    extra.pickers.oldfiles()
    return
  end

  require("core.search").oldfiles()
end

local function action_new_file()
  vim.cmd.enew()
end

local function action_open_config()
  vim.cmd.edit(vim.fn.fnameescape(vim.fs.joinpath(vim.fn.stdpath("config"), "init.lua")))
end

local function action_mason()
  vim.cmd.Mason()
end

local function action_quit()
  vim.cmd.quit()
end

local menu = {
  { key = "1", label = "Open workspace", action = action_open_files },
  { key = "2", label = "Find file", action = action_find_files },
  { key = "3", label = "Search text", action = action_search_text },
  { key = "4", label = "Recent files", action = action_recent_files },
  { key = "5", label = "New file", action = action_new_file },
  { key = "6", label = "Open config", action = action_open_config },
  { key = "7", label = "Mason tools", action = action_mason },
  { key = "q", label = "Quit", action = action_quit },
}

local function render()
  local lines = {}
  local actions = {}
  local action_lines = {}

  for _ = 1, 3 do
    lines[#lines + 1] = ""
  end

  for _, line in ipairs(logo) do
    lines[#lines + 1] = center(line)
  end

  lines[#lines + 1] = ""
  lines[#lines + 1] = center("A small Neovim workspace")
  lines[#lines + 1] = ""
  lines[#lines + 1] = ""

  for _, item in ipairs(menu) do
    local line = center(string.format("%s  %s", item.key, item.label))
    lines[#lines + 1] = line
    actions[#lines] = item.action
    action_lines[#action_lines + 1] = #lines
  end

  return lines, actions, action_lines
end

local function highlight(buf, lines)
  local ns = vim.api.nvim_create_namespace("user_home")

  vim.api.nvim_buf_clear_namespace(buf, ns, 0, -1)

  for index = 4, 8 do
    vim.api.nvim_buf_add_highlight(buf, ns, "Title", index - 1, 0, -1)
  end

  for index, line in ipairs(lines) do
    local col = line:find("%S")

    if col and line:match("^%s*[%d,q]%s%s") then
      vim.api.nvim_buf_add_highlight(buf, ns, "Number", index - 1, col - 1, col)
      vim.api.nvim_buf_add_highlight(buf, ns, "NormalFloat", index - 1, col + 2, -1)
    end
  end
end

local function set_keymaps(buf, actions, action_lines)
  local function choose()
    local line = vim.api.nvim_win_get_cursor(0)[1]
    local action = actions[line]

    if action then
      action()
    end
  end

  local function move(step)
    local current = vim.api.nvim_win_get_cursor(0)[1]
    local target_index = step > 0 and 1 or #action_lines

    for index, line in ipairs(action_lines) do
      if step > 0 and line > current then
        target_index = index
        break
      end

      if step < 0 and line < current then
        target_index = index
      end
    end

    vim.api.nvim_win_set_cursor(0, { action_lines[target_index], 0 })
  end

  vim.keymap.set("n", "<CR>", choose, { buffer = buf, desc = "Home choose" })
  vim.keymap.set("n", "<Down>", function()
    move(1)
  end, { buffer = buf, desc = "Home next item" })
  vim.keymap.set("n", "j", function()
    move(1)
  end, { buffer = buf, desc = "Home next item" })
  vim.keymap.set("n", "<Up>", function()
    move(-1)
  end, { buffer = buf, desc = "Home previous item" })
  vim.keymap.set("n", "k", function()
    move(-1)
  end, { buffer = buf, desc = "Home previous item" })

  for _, item in ipairs(menu) do
    vim.keymap.set("n", item.key, item.action, { buffer = buf, desc = "Home " .. item.label })
  end
end

function M.open()
  if not can_reuse_current_buffer() then
    vim.cmd.enew()
  end

  local buf = vim.api.nvim_get_current_buf()
  local lines, actions, action_lines = render()

  vim.bo[buf].modifiable = true
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
  set_home_options(buf)
  highlight(buf, lines)
  set_keymaps(buf, actions, action_lines)
  vim.api.nvim_win_set_cursor(0, { action_lines[1], 0 })
end

return M
