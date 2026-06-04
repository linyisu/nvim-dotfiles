local M = {}
local neovide_motion_restore = nil

local neovide_motion_keys = {
  "neovide_cursor_animation_length",
  "neovide_cursor_short_animation_length",
  "neovide_cursor_trail_size",
  "neovide_scroll_animation_length",
  "neovide_position_animation_length",
}

local function notify(message, level)
  vim.notify(message, level or vim.log.levels.INFO)
end

function M.disable_neovide_animations()
  if not vim.g.neovide or neovide_motion_restore then
    return
  end

  local previous = {}

  for _, key in ipairs(neovide_motion_keys) do
    previous[key] = {
      exists = vim.g[key] ~= nil,
      value = vim.g[key],
    }
    vim.g[key] = 0
  end

  neovide_motion_restore = previous
end

function M.restore_neovide_animations()
  if not neovide_motion_restore then
    return
  end

  for key, state in pairs(neovide_motion_restore) do
    vim.g[key] = state.exists and state.value or nil
  end

  neovide_motion_restore = nil
end

function M.disable_buffer_animations(buf)
  vim.b[buf].snacks_animate = false
  vim.b[buf].snacks_scroll = false
end

local function stop_smooth_scroll(win)
  if not (_G.Snacks and Snacks.animate and Snacks.animate.del) then
    return
  end

  pcall(Snacks.animate.del, ("scroll_%d"):format(win))
  pcall(Snacks.animate.del, ("scroll_repeat_%d"):format(win))
end

local function mini_files()
  local ok, files = pcall(require, "mini.files")

  if not ok then
    return nil
  end

  return files
end

local function refresh()
  local files = mini_files()

  if files then
    files.refresh({ content = { filter = files.config.content.filter } })
  end
end

local function selected_entry()
  local files = mini_files()

  if not files then
    return nil
  end

  return files.get_fs_entry()
end

local function focused_directory()
  local files = mini_files()

  if not files then
    return vim.fn.getcwd()
  end

  local state = files.get_explorer_state()

  if state and state.branch and state.depth_focus then
    return state.branch[state.depth_focus]
  end

  local entry = selected_entry()

  if entry and entry.path then
    if entry.fs_type == "directory" then
      return entry.path
    end

    return vim.fs.dirname(entry.path)
  end

  return vim.fn.getcwd()
end

local function normalize_directory(path)
  if not path or path == "" then
    return nil
  end

  path = vim.fs.normalize(vim.fn.fnamemodify(path, ":p"))

  if vim.fn.isdirectory(path) ~= 1 then
    return nil
  end

  return path
end

local function create_path(kind)
  local base = focused_directory()

  vim.ui.input({ prompt = "New " .. kind .. ": " }, function(name)
    if not name or name == "" then
      return
    end

    local path = vim.fs.joinpath(base, name)

    if vim.uv.fs_stat(path) then
      notify("Path already exists: " .. path, vim.log.levels.WARN)
      return
    end

    if kind == "directory" then
      local ok = vim.fn.mkdir(path, "p") == 1

      if not ok then
        notify("Failed to create directory: " .. path, vim.log.levels.ERROR)
        return
      end
    else
      vim.fn.mkdir(vim.fs.dirname(path), "p")

      local fd = vim.uv.fs_open(path, "w", 420)

      if not fd then
        notify("Failed to create file: " .. path, vim.log.levels.ERROR)
        return
      end

      vim.uv.fs_close(fd)
    end

    refresh()
  end)
end

function M.toggle()
  local files = mini_files()

  if not files then
    vim.cmd.Explore()
    return
  end

  if files.close() then
    return
  end

  local current = vim.api.nvim_buf_get_name(0)
  files.open(current ~= "" and current or nil, false)
end

function M.open(path)
  local files = mini_files()

  if not files then
    vim.cmd.Explore(vim.fn.fnameescape(path or vim.fn.getcwd()))
    return
  end

  local target = path

  if target and target ~= "" then
    target = vim.fs.normalize(vim.fn.fnamemodify(target, ":p"))
  end

  files.open(target, false)
end

function M.open_workspace(path)
  local directory = normalize_directory(path)

  if not directory then
    notify("Not a valid directory: " .. tostring(path), vim.log.levels.WARN)
    return
  end

  vim.cmd.cd(vim.fn.fnameescape(directory))

  local files = mini_files()

  if files then
    files.close()
    files.open(directory, false)
  end

  notify("Workspace: " .. directory)
end

function M.open_selected_workspace()
  local entry = selected_entry()
  local directory

  if entry and entry.path then
    directory = entry.fs_type == "directory" and entry.path or vim.fs.dirname(entry.path)
  else
    directory = focused_directory()
  end

  M.open_workspace(directory)
end

function M.create_file()
  create_path("file")
end

function M.create_directory()
  create_path("directory")
end

function M.rename()
  local entry = selected_entry()

  if not entry or not entry.path then
    notify("Cursor is not on a file or directory", vim.log.levels.WARN)
    return
  end

  local old_path = entry.path
  local parent = vim.fs.dirname(old_path)

  vim.ui.input({ prompt = "Rename to: ", default = vim.fs.basename(old_path) }, function(name)
    if not name or name == "" then
      return
    end

    local new_path = vim.fs.joinpath(parent, name)

    if new_path == old_path then
      return
    end

    if vim.uv.fs_stat(new_path) then
      notify("Path already exists: " .. new_path, vim.log.levels.WARN)
      return
    end

    local ok, err = vim.uv.fs_rename(old_path, new_path)

    if not ok then
      notify("Failed to rename: " .. tostring(err), vim.log.levels.ERROR)
      return
    end

    if _G.Snacks and Snacks.rename then
      pcall(Snacks.rename.on_rename_file, old_path, new_path)
    end

    refresh()
  end)
end

function M.remove()
  local entry = selected_entry()

  if not entry or not entry.path then
    notify("Cursor is not on a file or directory", vim.log.levels.WARN)
    return
  end

  local path = entry.path
  local prompt = "Delete " .. path .. "? Type y to confirm: "

  vim.ui.input({ prompt = prompt }, function(input)
    if input ~= "y" then
      return
    end

    local flags = entry.fs_type == "directory" and "rf" or ""
    local result = vim.fn.delete(path, flags)

    if result ~= 0 then
      notify("Failed to delete: " .. path, vim.log.levels.ERROR)
      return
    end

    refresh()
  end)
end

function M.refresh()
  refresh()
end

function M.move_cursor(buf, step, count)
  local line_count = vim.api.nvim_buf_line_count(buf)

  if line_count == 0 then
    return
  end

  local current = vim.api.nvim_win_get_cursor(0)
  local raw_target = current[1] - 1 + step * (count or vim.v.count1)
  local target = (raw_target % line_count) + 1
  local wrapped = raw_target < 0 or raw_target >= line_count
  local line = vim.api.nvim_buf_get_lines(buf, target - 1, target, false)[1] or ""
  local col = math.min(current[2], math.max(#line - 1, 0))

  if wrapped then
    stop_smooth_scroll(vim.api.nvim_get_current_win())
  end

  vim.api.nvim_win_set_cursor(0, { target, col })
end

return M
