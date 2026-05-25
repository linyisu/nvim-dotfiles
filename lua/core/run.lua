local M = {}

local c_filetypes = {
  c = true,
  cpp = true,
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

local function open_terminal(exe, cwd)
  vim.cmd("botright 14split")
  vim.cmd.enew()
  vim.bo.bufhidden = "wipe"
  vim.bo.filetype = "terminal"

  vim.fn.termopen({ exe }, {
    cwd = cwd,
  })
  vim.cmd.startinsert()
end

function M.current_file()
  local file = vim.api.nvim_buf_get_name(0)

  if file == "" then
    notify("Save the file before running it", vim.log.levels.WARN)
    return
  end

  local filetype = vim.bo.filetype

  if not c_filetypes[filetype] then
    notify("RunFile currently supports C and C++ files", vim.log.levels.WARN)
    return
  end

  if vim.bo.modified then
    vim.cmd.write()
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

      open_terminal(exe, cwd)
    end)
  end)
end

return M
