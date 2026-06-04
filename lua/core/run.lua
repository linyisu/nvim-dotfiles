local M = {}

local c_filetypes = {
  c = true,
  cpp = true,
}

local rust_filetypes = {
  rust = true,
}

local function notify(message, level)
  vim.notify(message, level or vim.log.levels.INFO)
end

local function first_executable(names)
  for _, name in ipairs(names) do
    if vim.fn.executable(name) == 1 then
      return name
    end
  end
end

local function output_path(file)
  local out_dir = vim.fs.joinpath(vim.fn.stdpath("cache"), "run")
  vim.fn.mkdir(out_dir, "p")

  local stem = vim.fn.fnamemodify(file, ":t:r")
  local suffix = vim.fn.sha256(file):sub(1, 8)
  local exe = string.format("%s-%s", stem, suffix)

  if vim.fn.has("win32") == 1 then
    exe = exe .. ".exe"
  end

  return vim.fs.joinpath(out_dir, exe)
end

local function compile_command(file, exe, filetype)
  if filetype == "c" then
    local compiler = first_executable({ "gcc", "clang" })

    if not compiler then
      return nil, "No C compiler found. Install gcc or clang."
    end

    return {
      compiler,
      "-std=c17",
      "-Wall",
      "-Wextra",
      file,
      "-o",
      exe,
    }
  end

  local compiler = first_executable({ "g++", "clang++" })

  if not compiler then
    return nil, "No C++ compiler found. Install g++ or clang++."
  end

  return {
    compiler,
    "-std=c++20",
    "-Wall",
    "-Wextra",
    file,
    "-o",
    exe,
  }
end

local function find_upwards(name, start)
  local dir = start

  if vim.fn.isdirectory(dir) ~= 1 then
    dir = vim.fs.dirname(dir)
  end

  while dir and dir ~= "" do
    local candidate = vim.fs.joinpath(dir, name)

    if vim.uv.fs_stat(candidate) then
      return candidate
    end

    local parent = vim.fs.dirname(dir)

    if not parent or parent == dir then
      return nil
    end

    dir = parent
  end
end

local function cargo_root(file)
  local manifest = find_upwards("Cargo.toml", vim.fs.dirname(file))

  if not manifest then
    return nil
  end

  return vim.fs.dirname(manifest)
end

local function open_compile_errors(output)
  local lines = vim.split(output, "\n", { plain = true, trimempty = true })

  if #lines == 0 then
    notify("Compile failed with no output", vim.log.levels.ERROR)
    return
  end

  vim.fn.setqflist({}, " ", {
    title = "Compile errors",
    lines = lines,
  })
  vim.cmd.copen()
  notify("Compile failed", vim.log.levels.ERROR)
end

local function open_terminal(command, cwd)
  vim.cmd("botright 14split")
  vim.cmd.enew()
  local buf = vim.api.nvim_get_current_buf()

  vim.bo[buf].bufhidden = "wipe"
  vim.bo[buf].filetype = "run-terminal"

  vim.fn.termopen(command, {
    cwd = cwd,
  })

  vim.keymap.set("t", "<Esc>", [[<C-\><C-n>]], {
    buffer = buf,
    desc = "Leave terminal input mode",
  })
  vim.keymap.set("n", "q", "<cmd>close<cr>", {
    buffer = buf,
    silent = true,
    desc = "Close run terminal",
  })
end

local function run_rust_project(file)
  local cargo = first_executable({ "cargo" })

  if not cargo then
    notify("No Rust toolchain found. Install cargo.", vim.log.levels.ERROR)
    return
  end

  local root = cargo_root(file)

  if not root then
    notify("No Cargo.toml found for this Rust file", vim.log.levels.WARN)
    return
  end

  notify("Running Rust project...")
  open_terminal({ cargo, "run" }, root)
end

function M.current_file()
  local file = vim.api.nvim_buf_get_name(0)

  if file == "" then
    notify("Save the file before running it", vim.log.levels.WARN)
    return
  end

  local filetype = vim.bo.filetype

  if not c_filetypes[filetype] and not rust_filetypes[filetype] then
    notify("RunFile currently supports C, C++, and Rust project files", vim.log.levels.WARN)
    return
  end

  if vim.bo.modified then
    vim.cmd.write()
  end

  if rust_filetypes[filetype] then
    run_rust_project(file)
    return
  end

  local exe = output_path(file)
  local command, err = compile_command(file, exe, filetype)

  if not command then
    notify(err, vim.log.levels.ERROR)
    return
  end

  local cwd = vim.fs.dirname(file)
  notify("Compiling " .. vim.fs.basename(file) .. "...")

  vim.system(command, { cwd = cwd, text = true }, function(result)
    vim.schedule(function()
      local output = table.concat({
        result.stdout or "",
        result.stderr or "",
      }, "\n")

      if result.code ~= 0 then
        open_compile_errors(output)
        return
      end

      if output:gsub("%s", "") ~= "" then
        vim.fn.setqflist({}, " ", {
          title = "Compile output",
          lines = vim.split(output, "\n", { plain = true, trimempty = true }),
        })
      end

      open_terminal({ exe }, cwd)
    end)
  end)
end

return M
