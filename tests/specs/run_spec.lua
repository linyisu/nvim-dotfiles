local function join(...)
  return vim.fs.joinpath(...)
end

local function write_lines(path, lines)
  expect.equal(vim.fn.mkdir(vim.fs.dirname(path), "p"), 1)
  expect.equal(vim.fn.writefile(lines, path), 0)
end

local function with_buffer(path, filetype, callback)
  local previous = vim.api.nvim_get_current_buf()
  local buf = vim.api.nvim_create_buf(false, false)

  if path then
    vim.api.nvim_buf_set_name(buf, path)
  end

  vim.api.nvim_set_current_buf(buf)
  vim.bo[buf].filetype = filetype or ""

  local ok, result = xpcall(function()
    return callback(buf)
  end, debug.traceback)

  if vim.api.nvim_buf_is_valid(buf) then
    vim.bo[buf].modified = false
  end

  if vim.api.nvim_buf_is_valid(previous) then
    vim.api.nvim_set_current_buf(previous)
  end

  if vim.api.nvim_buf_is_valid(buf) then
    vim.api.nvim_buf_delete(buf, { force = true })
  end

  if not ok then
    error(result, 0)
  end

  return result
end

local function command_stub(calls)
  local cmd = {}

  return setmetatable(cmd, {
    __call = function(_, command)
      calls[#calls + 1] = command
    end,
    __index = function(_, name)
      return function(...)
        calls[#calls + 1] = name
      end
    end,
  })
end

local function with_run_stubs(options, callback)
  options = options or {}

  local saved = {
    notify = vim.notify,
    system = vim.system,
    schedule = vim.schedule,
    cmd = vim.cmd,
    executable = vim.fn.executable,
    stdpath = vim.fn.stdpath,
    termopen = vim.fn.termopen,
    setqflist = vim.fn.setqflist,
    keymap_set = vim.keymap.set,
  }

  local state = {
    commands = {},
    notifications = {},
    system_calls = {},
    terminal_calls = {},
    quickfix_calls = {},
    keymaps = {},
  }

  local executables = options.executables or {}
  local system_result = options.system_result or { code = 0, stdout = "", stderr = "" }

  vim.notify = function(message, level)
    state.notifications[#state.notifications + 1] = {
      message = message,
      level = level,
    }
  end

  vim.system = function(command, system_options, callback_fn)
    state.system_calls[#state.system_calls + 1] = {
      command = command,
      options = system_options,
    }

    callback_fn(system_result)
  end

  vim.schedule = function(fn)
    fn()
  end

  vim.cmd = command_stub(state.commands)

  vim.fn.executable = function(name)
    return executables[name] and 1 or 0
  end

  vim.fn.stdpath = function(name)
    if name == "cache" and options.cache then
      return options.cache
    end

    return saved.stdpath(name)
  end

  vim.fn.termopen = function(command, term_options)
    state.terminal_calls[#state.terminal_calls + 1] = {
      command = command,
      options = term_options,
    }

    return 1
  end

  vim.fn.setqflist = function(list, action, qf_options)
    state.quickfix_calls[#state.quickfix_calls + 1] = {
      list = list,
      action = action,
      options = qf_options,
    }

    return 0
  end

  vim.keymap.set = function(mode, lhs, rhs, opts)
    state.keymaps[#state.keymaps + 1] = {
      mode = mode,
      lhs = lhs,
      rhs = rhs,
      opts = opts or {},
    }
  end

  local ok, result = xpcall(function()
    return callback(state)
  end, debug.traceback)

  vim.notify = saved.notify
  vim.system = saved.system
  vim.schedule = saved.schedule
  vim.cmd = saved.cmd
  vim.fn.executable = saved.executable
  vim.fn.stdpath = saved.stdpath
  vim.fn.termopen = saved.termopen
  vim.fn.setqflist = saved.setqflist
  vim.keymap.set = saved.keymap_set

  if not ok then
    error(result, 0)
  end

  return result
end

local function run_file(state)
  reload("core.run").current_file()
  return state
end

local function expect_notification(state, message, level)
  expect.truthy(state.notifications[1], "expected notification")
  expect.equal(state.notifications[1].message, message)
  expect.equal(state.notifications[1].level, level)
end

local function expect_command(actual, expected_compiler, expected_standard, source)
  expect.equal(actual[1], expected_compiler)
  expect.equal(actual[2], expected_standard)
  expect.equal(actual[3], "-Wall")
  expect.equal(actual[4], "-Wextra")
  expect.equal(actual[5], source)
  expect.equal(actual[6], "-o")
  expect.truthy(actual[7], "expected output executable path")
end

local function find_keymap(state, mode, lhs)
  for _, keymap in ipairs(state.keymaps) do
    if keymap.mode == mode and keymap.lhs == lhs then
      return keymap
    end
  end
end

describe("core.run", function()
  it("notifies when the current buffer has not been saved", function()
    with_buffer(nil, "cpp", function()
      with_run_stubs({}, function(state)
        run_file(state)

        expect_notification(state, "Save the file before running it", vim.log.levels.WARN)
        expect.equal(#state.system_calls, 0)
      end)
    end)
  end)

  it("notifies when the filetype is not C, C++, or Rust", function()
    with_temp_dir(function(root)
      local source = join(root, "main.lua")
      write_lines(source, { "print('hello')" })

      with_buffer(source, "lua", function()
        with_run_stubs({}, function(state)
          run_file(state)

          expect_notification(state, "RunFile currently supports C, C++, and Rust project files", vim.log.levels.WARN)
          expect.equal(#state.system_calls, 0)
        end)
      end)
    end)
  end)

  it("writes modified buffers before compiling", function()
    with_temp_dir(function(root)
      local source = join(root, "main.cpp")
      write_lines(source, { "int main() { return 0; }" })

      with_buffer(source, "cpp", function(buf)
        vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "int main() { return 1; }" })
        vim.bo[buf].modified = true

        with_run_stubs({
          cache = join(root, "cache"),
          executables = { ["g++"] = true },
        }, function(state)
          run_file(state)

          expect.equal(state.commands[1], "write")
          expect.equal(#state.system_calls, 1)
        end)
      end)
    end)
  end)

  it("compiles C++ with g++ or clang++ using c++20", function()
    with_temp_dir(function(root)
      local source = join(root, "main.cpp")
      write_lines(source, { "int main() { return 0; }" })

      with_buffer(source, "cpp", function()
        with_run_stubs({
          cache = join(root, "cache-gxx"),
          executables = { ["g++"] = true, ["clang++"] = true },
        }, function(state)
          run_file(state)

          expect_command(state.system_calls[1].command, "g++", "-std=c++20", source)
        end)
      end)

      with_buffer(source, "cpp", function()
        with_run_stubs({
          cache = join(root, "cache-clangxx"),
          executables = { ["clang++"] = true },
        }, function(state)
          run_file(state)

          expect_command(state.system_calls[1].command, "clang++", "-std=c++20", source)
        end)
      end)
    end)
  end)

  it("compiles C with gcc or clang using c17", function()
    with_temp_dir(function(root)
      local source = join(root, "main.c")
      write_lines(source, { "int main(void) { return 0; }" })

      with_buffer(source, "c", function()
        with_run_stubs({
          cache = join(root, "cache-gcc"),
          executables = { gcc = true, clang = true },
        }, function(state)
          run_file(state)

          expect_command(state.system_calls[1].command, "gcc", "-std=c17", source)
        end)
      end)

      with_buffer(source, "c", function()
        with_run_stubs({
          cache = join(root, "cache-clang"),
          executables = { clang = true },
        }, function(state)
          run_file(state)

          expect_command(state.system_calls[1].command, "clang", "-std=c17", source)
        end)
      end)
    end)
  end)

  it("writes compile failures to quickfix and opens it", function()
    with_temp_dir(function(root)
      local source = join(root, "main.cpp")
      write_lines(source, { "int main() { return broken; }" })

      with_buffer(source, "cpp", function()
        with_run_stubs({
          cache = join(root, "cache"),
          executables = { ["g++"] = true },
          system_result = {
            code = 1,
            stdout = "first error",
            stderr = "second error",
          },
        }, function(state)
          run_file(state)

          expect.equal(#state.quickfix_calls, 1)
          expect.equal(state.quickfix_calls[1].action, " ")
          expect.equal(state.quickfix_calls[1].options.title, "Compile errors")
          expect.equal(state.quickfix_calls[1].options.lines, { "first error", "second error" })
          expect.equal(state.commands[1], "copen")
          expect_notification(state, "Compiling main.cpp...", vim.log.levels.INFO)
          expect.equal(state.notifications[2].message, "Compile failed")
          expect.equal(state.notifications[2].level, vim.log.levels.ERROR)
          expect.equal(#state.terminal_calls, 0)
        end)
      end)
    end)
  end)

  it("opens a terminal after a successful compile", function()
    with_temp_dir(function(root)
      local source = join(root, "main.cpp")
      write_lines(source, { "int main() { return 0; }" })

      with_buffer(source, "cpp", function()
        with_run_stubs({
          cache = join(root, "cache"),
          executables = { ["g++"] = true },
        }, function(state)
          run_file(state)

          expect.equal(state.commands[1], "botright 14split")
          expect.equal(state.commands[2], "enew")
          expect.equal(state.commands[3], nil)
          expect.equal(#state.terminal_calls, 1)
          expect.equal(state.terminal_calls[1].command, { state.system_calls[1].command[7] })
          expect.equal(state.terminal_calls[1].options.cwd, root)

          local terminal_escape = find_keymap(state, "t", "<Esc>")
          local close = find_keymap(state, "n", "q")

          expect.truthy(terminal_escape, "expected terminal escape keymap")
          expect.equal(terminal_escape.rhs, [[<C-\><C-n>]])
          expect.truthy(close, "expected close keymap")
          expect.equal(close.rhs, "<cmd>close<cr>")
          expect.equal(close.opts.silent, true)
        end)
      end)
    end)
  end)

  it("runs Rust files with cargo from the project root", function()
    with_temp_dir(function(root)
      local project = join(root, "project")
      local source = join(project, "src", "main.rs")

      write_lines(join(project, "Cargo.toml"), {
        "[package]",
        'name = "project"',
        'version = "0.1.0"',
        'edition = "2021"',
      })
      write_lines(source, { "fn main() { println!(\"hello\"); }" })

      with_buffer(source, "rust", function(buf)
        vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "fn main() { println!(\"updated\"); }" })
        vim.bo[buf].modified = true

        with_run_stubs({
          executables = { cargo = true },
        }, function(state)
          run_file(state)

          expect.equal(state.commands[1], "write")
          expect.equal(state.commands[2], "botright 14split")
          expect.equal(state.commands[3], "enew")
          expect.equal(state.commands[4], nil)
          expect.equal(#state.system_calls, 0)
          expect.equal(#state.terminal_calls, 1)
          expect.equal(state.terminal_calls[1].command, { "cargo", "run" })
          expect.equal(state.terminal_calls[1].options.cwd, project)
        end)
      end)
    end)
  end)

  it("notifies when a Rust file is outside a Cargo project", function()
    with_temp_dir(function(root)
      local source = join(root, "main.rs")
      write_lines(source, { "fn main() {}" })

      with_buffer(source, "rust", function()
        with_run_stubs({
          executables = { cargo = true },
        }, function(state)
          run_file(state)

          expect.truthy(state.notifications[1], "expected notification")
          expect.truthy(
            string.find(state.notifications[1].message, "Cargo.toml", 1, true),
            "expected notification to mention Cargo.toml"
          )
          expect.equal(#state.system_calls, 0)
          expect.equal(#state.terminal_calls, 0)
        end)
      end)
    end)
  end)
end)
