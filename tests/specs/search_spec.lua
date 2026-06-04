local function with_search_stubs(callback)
  local previous_pick = package.loaded["mini.pick"]
  local previous_notify = vim.notify
  local previous_defer_fn = vim.defer_fn
  local previous_cwd = vim.fn.getcwd()
  local starts = {}
  local notifications = {}
  local published = {}
  local deferred = {}
  local picker_active = true
  local querytick = 1

  package.loaded["mini.pick"] = {
    start = function(opts)
      starts[#starts + 1] = opts
    end,
    get_querytick = function()
      return querytick
    end,
    is_picker_active = function()
      return picker_active
    end,
    set_picker_items = function(items, opts)
      published[#published + 1] = {
        items = items,
        opts = opts,
      }
    end,
    default_show = function() end,
    default_choose = function() end,
    get_picker_query = function()
      return {}
    end,
  }

  vim.notify = function(message, level, opts)
    notifications[#notifications + 1] = {
      message = message,
      level = level,
      opts = opts,
    }
  end

  vim.defer_fn = function(fn)
    deferred[#deferred + 1] = fn
  end

  local env = {
    starts = starts,
    notifications = notifications,
    published = published,
    set_active = function(value)
      picker_active = value
    end,
    set_querytick = function(value)
      querytick = value
    end,
    run_deferred = function(limit)
      limit = limit or 1000

      while #deferred > 0 and limit > 0 do
        local fn = table.remove(deferred, 1)
        fn()
        limit = limit - 1
      end

      expect.truthy(limit > 0, "deferred queue did not drain")
    end,
  }

  local ok, result = xpcall(function()
    return callback(env)
  end, debug.traceback)

  package.loaded["mini.pick"] = previous_pick
  vim.notify = previous_notify
  vim.defer_fn = previous_defer_fn
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

local function write_binary(path, content)
  vim.fn.mkdir(vim.fs.dirname(path), "p")

  local fd, open_err = vim.uv.fs_open(path, "w", 420)

  expect.truthy(fd, open_err)
  expect.truthy(vim.uv.fs_write(fd, content, 0))
  vim.uv.fs_close(fd)
end

local function find_item(items, text)
  for _, item in ipairs(items) do
    if item.text == text then
      return item
    end
  end
end

local function item_text_blob(items)
  local texts = {}

  for _, item in ipairs(items) do
    texts[#texts + 1] = item.text
  end

  return table.concat(texts, "\n")
end

local function expect_no_picker_and_workspace_warning(env, root)
  expect.equal(#env.starts, 0)
  expect.equal(#env.notifications, 1)

  local message = env.notifications[1].message
  local notified_root = message:gsub("^No files under workspace:%s*", "")

  expect.contains(message, "No files under workspace: ")
  expect.equal(vim.fs.normalize(notified_root), vim.fs.normalize(root))
  expect.equal(env.notifications[1].level, vim.log.levels.WARN)
end

local function latest_publish(env)
  return env.published[#env.published]
end

local function with_case_options(ignorecase, smartcase, callback)
  local previous_ignorecase = vim.o.ignorecase
  local previous_smartcase = vim.o.smartcase

  vim.o.ignorecase = ignorecase
  vim.o.smartcase = smartcase

  local ok, result = xpcall(callback, debug.traceback)

  vim.o.ignorecase = previous_ignorecase
  vim.o.smartcase = previous_smartcase

  if not ok then
    error(result, 0)
  end

  return result
end

local function with_fake_project_scan(base, entries, callback)
  local previous_getcwd = vim.fn.getcwd
  local previous_scandir = vim.uv.fs_scandir
  local previous_scandir_next = vim.uv.fs_scandir_next
  local handles = {}

  vim.fn.getcwd = function()
    return base
  end

  vim.uv.fs_scandir = function(dir)
    local normalized = vim.fs.normalize(dir)
    local dir_entries = entries[normalized] or entries[dir]

    if not dir_entries then
      return nil
    end

    local handle = {}

    handles[handle] = {
      entries = dir_entries,
      index = 0,
    }

    return handle
  end

  vim.uv.fs_scandir_next = function(handle)
    local state = handles[handle]

    if not state then
      return nil
    end

    state.index = state.index + 1

    local entry = state.entries[state.index]

    if not entry then
      return nil
    end

    return entry.name, entry.type
  end

  local ok, result = xpcall(callback, debug.traceback)

  vim.fn.getcwd = previous_getcwd
  vim.uv.fs_scandir = previous_scandir
  vim.uv.fs_scandir_next = previous_scandir_next

  if not ok then
    error(result, 0)
  end

  return result
end

describe("core.search picker integration", function()
  it("pick_files scans the current workspace and includes ordinary and nested files", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        write_file(vim.fs.joinpath(root, "main.lua"), { "return true" })
        write_file(vim.fs.joinpath(root, "src", "README.md"), { "# test" })

        reload("core.search").pick_files()

        expect.equal(#env.notifications, 0)
        expect.equal(#env.starts, 1)

        local source = env.starts[1].source
        local main = find_item(source.items, "main.lua")
        local readme = find_item(source.items, "src/README.md")

        expect.equal(vim.fs.normalize(source.cwd), vim.fs.normalize(root))
        expect.equal(source.name, "Files: " .. vim.fs.basename(root))
        expect.truthy(main)
        expect.truthy(readme)
        expect.equal(vim.fs.normalize(main.path), vim.fs.normalize(vim.fs.joinpath(root, "main.lua")))
      end)
    end)
  end)

  it("pick_files keeps relative file names intact when the workspace is a filesystem root", function()
    with_search_stubs(function(env)
      local base = vim.fn.has("win32") == 1 and "C:/" or "/"

      with_fake_project_scan(vim.fs.normalize(base), {
        [vim.fs.normalize(base)] = {
          { name = "foo.txt", type = "file" },
        },
      }, function()
        reload("core.search").pick_files()
      end)

      expect.equal(#env.notifications, 0)
      expect.equal(#env.starts, 1)

      local source = env.starts[1].source
      local item = find_item(source.items, "foo.txt")

      expect.truthy(item)
      expect.equal(vim.fs.normalize(item.path), vim.fs.normalize(vim.fs.joinpath(base, "foo.txt")))
    end)
  end)

  it("pick_files skips generated and dependency directories", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        write_file(vim.fs.joinpath(root, "keep.txt"), { "keep" })

        for _, dir in ipairs({ ".git", ".hg", ".svn", ".next", ".venv", "node_modules", "target", "build", "dist" }) do
          write_file(vim.fs.joinpath(root, dir, "ignored-from-" .. dir .. ".txt"), { "ignore" })
        end

        reload("core.search").pick_files()

        expect.equal(#env.notifications, 0)
        expect.equal(#env.starts, 1)

        local items = env.starts[1].source.items
        local texts = item_text_blob(items)

        expect.truthy(find_item(items, "keep.txt"))
        expect.falsy(texts:find("ignored-from-", 1, true))
      end)
    end)
  end)

  it("pick_files treats workspaces with only skipped directories as empty", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        write_file(vim.fs.joinpath(root, ".git", "ignored.txt"), { "ignore" })
        write_file(vim.fs.joinpath(root, "node_modules", "ignored.txt"), { "ignore" })

        reload("core.search").pick_files()

        expect_no_picker_and_workspace_warning(env, root)
      end)
    end)
  end)

  it("pick_files notifies and does not open a picker for an empty workspace", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        reload("core.search").pick_files()

        expect_no_picker_and_workspace_warning(env, root)
      end)
    end)
  end)

  it("pick_grep creates a text picker source for the current workspace", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        write_file(vim.fs.joinpath(root, "notes.txt"), { "needle" })

        reload("core.search").pick_grep()

        expect.equal(#env.notifications, 0)
        expect.equal(#env.starts, 1)

        local source = env.starts[1].source

        expect.equal(vim.fs.normalize(source.cwd), vim.fs.normalize(root))
        expect.equal(source.name, "Text: " .. vim.fs.basename(root))
        expect.equal(source.items, {})
        expect.truthy(source.match)
        expect.truthy(source.show)
        expect.truthy(source.choose)
      end)
    end)
  end)

  it("pick_grep publishes literal text matches with file, line, and path data", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        write_file(vim.fs.joinpath(root, "src", "main.cpp"), {
          "int main() {",
          "    cout << 42;",
          "}",
        })
        write_file(vim.fs.joinpath(root, "notes.txt"), { "no match" })

        reload("core.search").pick_grep()

        local source = env.starts[1].source
        source.match(nil, nil, { "cout <<" })
        env.run_deferred()

        local publish = latest_publish(env)

        expect.truthy(publish, "expected grep results to be published")
        expect.equal(publish.opts.do_match, false)
        expect.equal(publish.opts.querytick, 1)
        expect.equal(#publish.items, 1)
        expect.contains(publish.items[1].text, "src/main.cpp:2:")
        expect.equal(vim.fs.normalize(publish.items[1].path), vim.fs.normalize(vim.fs.joinpath(root, "src", "main.cpp")))
        expect.equal(publish.items[1].lnum, 2)
        expect.equal(publish.items[1].col, 1)
      end)
    end)
  end)

  it("pick_grep handles empty and one-character queries without scanning files", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        write_file(vim.fs.joinpath(root, "notes.txt"), { "needle" })

        reload("core.search").pick_grep()

        local source = env.starts[1].source

        source.match(nil, nil, { "" })
        source.match(nil, nil, { "n" })

        expect.equal(#env.published, 2)
        expect.equal(env.published[1].items, {})
        expect.equal(env.published[2].items, {})
      end)
    end)
  end)

  it("pick_grep honors ignorecase and smartcase", function()
    with_case_options(true, true, function()
      with_search_stubs(function(env)
        with_temp_dir(function(root)
          write_file(vim.fs.joinpath(root, "case.txt"), {
            "alpha",
            "ALPHA",
          })

          reload("core.search").pick_grep()

          local source = env.starts[1].source

          source.match(nil, nil, { "alpha" })
          env.run_deferred()
          expect.equal(#latest_publish(env).items, 2)

          source.match(nil, nil, { "Alpha" })
          env.run_deferred()
          expect.equal(#latest_publish(env).items, 0)
        end)
      end)
    end)
  end)

  it("pick_grep skips large and binary files", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        write_file(vim.fs.joinpath(root, "small.txt"), { "needle" })
        write_binary(vim.fs.joinpath(root, "binary.txt"), "needle\0hidden")
        write_file(vim.fs.joinpath(root, "large.txt"), { string.rep("x", 1024 * 1024 + 1) .. "needle" })

        reload("core.search").pick_grep()

        local source = env.starts[1].source
        source.match(nil, nil, { "needle" })
        env.run_deferred()

        local texts = item_text_blob(latest_publish(env).items)

        expect.contains(texts, "small.txt")
        expect.falsy(texts:find("binary.txt", 1, true))
        expect.falsy(texts:find("large.txt", 1, true))
      end)
    end)
  end)

  it("pick_grep notifies and does not open a picker for an empty workspace", function()
    with_search_stubs(function(env)
      with_temp_dir(function(root)
        reload("core.search").pick_grep()

        expect_no_picker_and_workspace_warning(env, root)
      end)
    end)
  end)
end)
