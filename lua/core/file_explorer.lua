local M = {}

local function notify(message, level)
  vim.notify(message, level or vim.log.levels.INFO)
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

  files.open(path, false)
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

return M
