local function with_file_explorer_stubs(callback)
  local previous_files = package.loaded["mini.files"]
  local previous_notify = vim.notify
  local previous_input = vim.ui.input
  local previous_cwd = vim.fn.getcwd()

  local opens = {}
  local closes = {}
  local refreshes = {}
  local notifications = {}
  local inputs = {}
  local responses = {}
  local close_result = false
  local explorer_state = nil
  local fs_entry = nil
  local filter = function()
    return true
  end

  local files = {
    config = {
      content = {
        filter = filter,
      },
    },
    close = function()
      closes[#closes + 1] = true
      return close_result
    end,
    open = function(path, use_latest)
      opens[#opens + 1] = {
        path = path,
        use_latest = use_latest,
      }
    end,
    refresh = function(opts)
      refreshes[#refreshes + 1] = opts
    end,
    get_explorer_state = function()
      return explorer_state
    end,
    get_fs_entry = function()
      return fs_entry
    end,
  }

  package.loaded["mini.files"] = files

  vim.notify = function(message, level, opts)
    notifications[#notifications + 1] = {
      message = message,
      level = level,
      opts = opts,
    }
  end

  vim.ui.input = function(opts, callback_input)
    inputs[#inputs + 1] = opts
    callback_input(table.remove(responses, 1))
  end

  local env = {
    files = files,
    opens = opens,
    closes = closes,
    refreshes = refreshes,
    notifications = notifications,
    inputs = inputs,
    set_close_result = function(value)
      close_result = value
    end,
    set_explorer_state = function(value)
      explorer_state = value
    end,
    set_fs_entry = function(value)
      fs_entry = value
    end,
    set_input = function(value)
      responses[#responses + 1] = value
    end,
  }

  local ok, result = xpcall(function()
    return callback(env, reload("core.file_explorer"))
  end, debug.traceback)

  package.loaded["mini.files"] = previous_files
  vim.notify = previous_notify
  vim.ui.input = previous_input
  vim.cmd.cd(vim.fn.fnameescape(previous_cwd))

  if not ok then
    error(result, 0)
  end

  return result
end

local function write_file(path, lines)
  vim.fn.mkdir(vim.fs.dirname(path), "p")
  expect.equal(vim.fn.writefile(type(lines) == "table" and lines or { lines }, path), 0)
end

local function read_file(path)
  return table.concat(vim.fn.readfile(path), "\n")
end

describe("core.file_explorer mini.files integration", function()
  it("toggle closes an open explorer without reopening it", function()
    with_file_explorer_stubs(function(env, explorer)
      env.set_close_result(true)

      explorer.toggle()

      expect.equal(#env.closes, 1)
      expect.equal(#env.opens, 0)
    end)
  end)

  it("toggle opens the current buffer file when the explorer is not already open", function()
    with_file_explorer_stubs(function(env, explorer)
      with_temp_dir(function(root)
        local path = vim.fs.joinpath(root, "current.txt")
        local buffer = vim.api.nvim_create_buf(true, false)

        write_file(path, "current")
        vim.api.nvim_buf_set_name(buffer, path)
        vim.api.nvim_set_current_buf(buffer)

        explorer.toggle()

        expect.equal(#env.closes, 1)
        expect.equal(#env.opens, 1)
        expect.equal(vim.fs.normalize(env.opens[1].path), vim.fs.normalize(path))
        expect.equal(env.opens[1].use_latest, false)
      end)
    end)
  end)

  it("open_workspace changes to a valid directory and reopens mini.files there", function()
    with_file_explorer_stubs(function(env, explorer)
      with_temp_dir(function(root)
        local workspace = vim.fs.joinpath(root, "workspace")

        expect.equal(vim.fn.mkdir(workspace, "p"), 1)

        explorer.open_workspace(workspace)

        expect.equal(vim.fs.normalize(vim.fn.getcwd()), vim.fs.normalize(workspace))
        expect.equal(#env.closes, 1)
        expect.equal(#env.opens, 1)
        expect.equal(vim.fs.normalize(env.opens[1].path), vim.fs.normalize(workspace))
        expect.equal(env.opens[1].use_latest, false)
        expect.equal(#env.notifications, 1)
        expect.contains(env.notifications[1].message, "Workspace: ")
      end)
    end)
  end)

  it("create_file creates a file under the focused directory and refreshes mini.files", function()
    with_file_explorer_stubs(function(env, explorer)
      with_temp_dir(function(root)
        env.set_explorer_state({
          branch = { root },
          depth_focus = 1,
        })
        env.set_input("nested/new.txt")

        explorer.create_file()

        local path = vim.fs.joinpath(root, "nested", "new.txt")

        expect.truthy(vim.uv.fs_stat(path))
        expect.equal(#env.refreshes, 1)
        expect.equal(env.refreshes[1].content.filter, env.files.config.content.filter)
      end)
    end)
  end)

  it("create_directory creates a directory under the focused directory and refreshes mini.files", function()
    with_file_explorer_stubs(function(env, explorer)
      with_temp_dir(function(root)
        env.set_explorer_state({
          branch = { root },
          depth_focus = 1,
        })
        env.set_input("nested/dir")

        explorer.create_directory()

        local path = vim.fs.joinpath(root, "nested", "dir")

        expect.equal(vim.fn.isdirectory(path), 1)
        expect.equal(#env.refreshes, 1)
        expect.equal(env.refreshes[1].content.filter, env.files.config.content.filter)
      end)
    end)
  end)

  it("rename warns and does not overwrite an existing target", function()
    with_file_explorer_stubs(function(env, explorer)
      with_temp_dir(function(root)
        local old_path = vim.fs.joinpath(root, "old.txt")
        local existing_path = vim.fs.joinpath(root, "existing.txt")

        write_file(old_path, "old")
        write_file(existing_path, "existing")
        env.set_fs_entry({
          path = old_path,
          fs_type = "file",
        })
        env.set_input("existing.txt")

        explorer.rename()

        expect.equal(read_file(old_path), "old")
        expect.equal(read_file(existing_path), "existing")
        expect.equal(#env.refreshes, 0)
        expect.equal(#env.notifications, 1)
        expect.equal(env.notifications[1].level, vim.log.levels.WARN)
        expect.contains(env.notifications[1].message, "Path already exists: ")
      end)
    end)
  end)

  it("remove only deletes after y confirmation", function()
    with_file_explorer_stubs(function(env, explorer)
      with_temp_dir(function(root)
        local path = vim.fs.joinpath(root, "delete-me.txt")

        write_file(path, "delete")
        env.set_fs_entry({
          path = path,
          fs_type = "file",
        })
        env.set_input("n")

        explorer.remove()

        expect.truthy(vim.uv.fs_stat(path))
        expect.equal(#env.refreshes, 0)

        env.set_input("y")

        explorer.remove()

        expect.falsy(vim.uv.fs_stat(path))
        expect.equal(#env.refreshes, 1)
      end)
    end)
  end)

  it("move_cursor preserves the current column while moving and wrapping", function()
    local explorer = reload("core.file_explorer")
    local previous = vim.api.nvim_get_current_buf()
    local buf = vim.api.nvim_create_buf(false, true)

    vim.api.nvim_buf_set_lines(buf, 0, -1, false, {
      "▣ alpha",
      "▣ beta",
      "▣ gamma",
    })
    vim.api.nvim_set_current_buf(buf)
    vim.api.nvim_win_set_cursor(0, { 1, 4 })

    local ok, err = xpcall(function()
      explorer.move_cursor(buf, 1)
      expect.equal(vim.api.nvim_win_get_cursor(0), { 2, 4 })

      explorer.move_cursor(buf, -1)
      expect.equal(vim.api.nvim_win_get_cursor(0), { 1, 4 })

      explorer.move_cursor(buf, -1)
      expect.equal(vim.api.nvim_win_get_cursor(0), { 3, 4 })

      explorer.move_cursor(buf, 1, 2)
      expect.equal(vim.api.nvim_win_get_cursor(0), { 2, 4 })
    end, debug.traceback)

    if vim.api.nvim_buf_is_valid(previous) then
      vim.api.nvim_set_current_buf(previous)
    end

    if vim.api.nvim_buf_is_valid(buf) then
      vim.api.nvim_buf_delete(buf, { force = true })
    end

    if not ok then
      error(err, 0)
    end
  end)

  it("disables file explorer animations and restores Neovide settings on close", function()
    local explorer = reload("core.file_explorer")
    local previous_neovide = vim.g.neovide
    local neovide_keys = {
      neovide_cursor_animation_length = 0.08,
      neovide_cursor_short_animation_length = 0.03,
      neovide_cursor_trail_size = 0.5,
      neovide_scroll_animation_length = 0.15,
      neovide_position_animation_length = 0.08,
    }
    local previous_values = {}
    local buf = vim.api.nvim_create_buf(false, true)

    for key, value in pairs(neovide_keys) do
      previous_values[key] = vim.g[key]
      vim.g[key] = value
    end

    vim.g.neovide = true

    local ok, err = xpcall(function()
      explorer.disable_buffer_animations(buf)
      expect.equal(vim.b[buf].snacks_animate, false)
      expect.equal(vim.b[buf].snacks_scroll, false)

      explorer.disable_neovide_animations()
      for key in pairs(neovide_keys) do
        expect.equal(vim.g[key], 0)
      end

      explorer.restore_neovide_animations()
      for key, value in pairs(neovide_keys) do
        expect.equal(vim.g[key], value)
      end
    end, debug.traceback)

    vim.g.neovide = previous_neovide

    for key, value in pairs(previous_values) do
      vim.g[key] = value
    end

    if vim.api.nvim_buf_is_valid(buf) then
      vim.api.nvim_buf_delete(buf, { force = true })
    end

    if not ok then
      error(err, 0)
    end
  end)

  it("restores nil Neovide animation settings after file explorer closes", function()
    local explorer = reload("core.file_explorer")
    local previous_neovide = vim.g.neovide
    local key = "neovide_cursor_short_animation_length"
    local previous = vim.g[key]

    vim.g.neovide = true
    vim.g[key] = nil

    local ok, err = xpcall(function()
      explorer.disable_neovide_animations()
      expect.equal(vim.g[key], 0)

      explorer.restore_neovide_animations()
      expect.equal(vim.g[key], nil)
    end, debug.traceback)

    vim.g.neovide = previous_neovide
    vim.g[key] = previous

    if not ok then
      error(err, 0)
    end
  end)

  it("move_cursor stops Snacks smooth scroll only on wrap jumps", function()
    local explorer = reload("core.file_explorer")
    local previous = vim.api.nvim_get_current_buf()
    local previous_snacks = _G.Snacks
    local calls = {}
    local win = vim.api.nvim_get_current_win()
    local buf = vim.api.nvim_create_buf(false, true)

    _G.Snacks = {
      animate = {
        del = function(id)
          calls[#calls + 1] = id
        end,
      },
    }

    vim.api.nvim_buf_set_lines(buf, 0, -1, false, {
      "  alpha",
      "  beta",
      "  gamma",
    })
    vim.api.nvim_set_current_buf(buf)
    vim.api.nvim_win_set_cursor(0, { 2, 3 })

    local ok, err = xpcall(function()
      explorer.move_cursor(buf, 1)
      expect.equal(vim.api.nvim_win_get_cursor(0), { 3, 3 })
      expect.equal(calls, {})

      explorer.move_cursor(buf, 1)
      expect.equal(vim.api.nvim_win_get_cursor(0), { 1, 3 })
      expect.equal(calls, {
        ("scroll_%d"):format(win),
        ("scroll_repeat_%d"):format(win),
      })
    end, debug.traceback)

    _G.Snacks = previous_snacks

    if vim.api.nvim_buf_is_valid(previous) then
      vim.api.nvim_set_current_buf(previous)
    end

    if vim.api.nvim_buf_is_valid(buf) then
      vim.api.nvim_buf_delete(buf, { force = true })
    end

    if not ok then
      error(err, 0)
    end
  end)
end)
