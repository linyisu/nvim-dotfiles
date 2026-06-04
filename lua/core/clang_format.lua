local M = {}

M.lines = {
  "BasedOnStyle: Google",
  "Standard: Latest",
  "IndentWidth: 4",
  "ColumnLimit: 120",
  "AccessModifierOffset: -4",
  "InsertBraces: true",
  "",
}

local is_case_insensitive_fs = vim.fn.has("win32") == 1 or vim.fn.has("mac") == 1

local function normalize(path)
  return vim.fs.normalize(path):gsub("\\", "/")
end

local function comparable(path)
  path = normalize(path)
  return is_case_insensitive_fs and path:lower() or path
end

local function is_inside(path, root)
  local normalized_path = comparable(path)
  local normalized_root = comparable(root)

  return normalized_path == normalized_root or normalized_path:sub(1, #normalized_root + 1) == normalized_root .. "/"
end

function M.find(start, stop)
  local dir = start or vim.fn.getcwd()

  if vim.fn.isdirectory(dir) ~= 1 then
    dir = vim.fs.dirname(dir)
  end

  local stop_path = stop and comparable(stop)

  if stop_path and not is_inside(dir, stop) then
    return nil
  end

  while dir and dir ~= "" do
    local candidate = vim.fs.joinpath(dir, ".clang-format")

    if vim.uv.fs_stat(candidate) then
      return candidate
    end

    if stop_path and comparable(dir) == stop_path then
      return nil
    end

    local parent = vim.fs.dirname(dir)

    if not parent or parent == dir then
      return nil
    end

    dir = parent
  end
end

function M.path(root)
  return vim.fs.joinpath(root or vim.fn.getcwd(), ".clang-format")
end

function M.write_default(opts)
  opts = opts or {}

  local path = M.path(opts.root)
  local exists = vim.uv.fs_stat(path) ~= nil

  if exists and not opts.force then
    if not opts.quiet then
      vim.notify(".clang-format already exists: " .. path, vim.log.levels.INFO)
    end

    return
  end

  local ok, result = pcall(vim.fn.writefile, M.lines, path)

  if not ok or result ~= 0 then
    if not opts.quiet then
      vim.notify("Failed to write .clang-format: " .. path, vim.log.levels.ERROR)
    end

    return
  end

  if not opts.quiet then
    vim.notify((exists and "Updated " or "Created ") .. path, vim.log.levels.INFO)
  end
end

function M.ensure_for_buffer(_)
  -- Kept as a no-op for compatibility. Format files should be created manually
  -- at the project root with :ClangFormatInit or by editing .clang-format.
end

function M.open()
  vim.cmd.edit(vim.fn.fnameescape(M.path()))
end

return M
