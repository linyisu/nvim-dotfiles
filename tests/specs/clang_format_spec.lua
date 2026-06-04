local function normalize(path)
  return path and vim.fs.normalize(path):gsub("\\", "/") or path
end

local function join(...)
  return vim.fs.joinpath(...)
end

local function exists(path)
  return vim.uv.fs_stat(path) ~= nil
end

local function mkdir(path)
  expect.equal(vim.fn.mkdir(path, "p"), 1)
end

local function write_lines(path, lines)
  mkdir(vim.fs.dirname(path))
  expect.equal(vim.fn.writefile(lines, path), 0)
end

local function read_lines(path)
  return vim.fn.readfile(path)
end

local function with_named_buffer(path, callback)
  write_lines(path, { "int main() { return 0; }" })

  local buf = vim.api.nvim_create_buf(false, false)
  vim.api.nvim_buf_set_name(buf, path)

  local ok, result = xpcall(function()
    return callback(buf)
  end, debug.traceback)

  if vim.api.nvim_buf_is_valid(buf) then
    vim.api.nvim_buf_delete(buf, { force = true })
  end

  if not ok then
    error(result, 0)
  end

  return result
end

describe("core.clang_format", function()
  it("keeps the default lines in sync with the configured clang-format content", function()
    local clang_format = reload("core.clang_format")

    expect.equal(clang_format.lines, {
      "BasedOnStyle: Google",
      "Standard: Latest",
      "IndentWidth: 4",
      "ColumnLimit: 120",
      "AccessModifierOffset: -4",
      "InsertBraces: true",
      "",
    })
  end)

  it("writes the default .clang-format in the current workspace", function()
    with_temp_dir(function(root)
      local clang_format = reload("core.clang_format")
      local target = join(root, ".clang-format")

      clang_format.write_default({ quiet = true })

      expect.truthy(exists(target))
      expect.equal(read_lines(target), clang_format.lines)
    end)
  end)

  it("does not overwrite an existing .clang-format unless forced", function()
    with_temp_dir(function(root)
      local clang_format = reload("core.clang_format")
      local target = join(root, ".clang-format")
      local custom_lines = {
        "BasedOnStyle: LLVM",
        "IndentWidth: 2",
      }

      write_lines(target, custom_lines)

      clang_format.write_default({ quiet = true })
      expect.equal(read_lines(target), custom_lines)

      clang_format.write_default({ force = true, quiet = true })
      expect.equal(read_lines(target), clang_format.lines)
    end)
  end)

  it("finds .clang-format while walking up to stop but not above it", function()
    with_temp_dir(function(root)
      local clang_format = reload("core.clang_format")
      local workspace = join(root, "workspace")
      local nested = join(workspace, "src", "lib")
      local start = join(nested, "main.cpp")
      local workspace_config = join(workspace, ".clang-format")

      mkdir(nested)
      write_lines(workspace_config, { "BasedOnStyle: Google" })

      local found = clang_format.find(start, workspace)

      expect.truthy(found)
      expect.equal(normalize(found), normalize(workspace_config))

      vim.fn.delete(workspace_config)
      write_lines(join(root, ".clang-format"), { "BasedOnStyle: LLVM" })

      expect.equal(clang_format.find(start, workspace), nil)
    end)
  end)

  it("does not confuse sibling workspaces with similar path prefixes", function()
    with_temp_dir(function(root)
      local clang_format = reload("core.clang_format")
      local workspace = join(root, "workspace")
      local sibling = join(root, "workspace-other")
      local start = join(sibling, "src", "main.cpp")

      mkdir(vim.fs.dirname(start))
      write_lines(join(workspace, ".clang-format"), { "BasedOnStyle: Google" })

      expect.equal(clang_format.find(start, workspace), nil)
    end)
  end)

  it("does not create .clang-format automatically for named C-family buffers", function()
    with_temp_dir(function(root)
      local clang_format = reload("core.clang_format")
      local workspace = join(root, "workspace")
      local source_dir = join(workspace, "src")
      local source = join(source_dir, "main.cpp")

      mkdir(source_dir)
      vim.cmd.cd(vim.fn.fnameescape(workspace))

      with_named_buffer(source, function(buf)
        clang_format.ensure_for_buffer(buf)

        expect.falsy(exists(join(workspace, ".clang-format")))
        expect.falsy(exists(join(source_dir, ".clang-format")))
        expect.falsy(exists(join(root, ".clang-format")))
      end)
    end)
  end)

  it("does not create .clang-format for unnamed buffers", function()
    with_temp_dir(function(root)
      local clang_format = reload("core.clang_format")
      local buf = vim.api.nvim_create_buf(false, false)

      clang_format.ensure_for_buffer(buf)

      expect.falsy(exists(join(root, ".clang-format")))

      if vim.api.nvim_buf_is_valid(buf) then
        vim.api.nvim_buf_delete(buf, { force = true })
      end
    end)
  end)

  it("does not create .clang-format for unrelated named buffers", function()
    with_temp_dir(function(root)
      local clang_format = reload("core.clang_format")
      local workspace = join(root, "workspace")
      local external = join(root, "external")
      local source = join(external, "tool.cpp")

      mkdir(workspace)
      mkdir(external)
      vim.cmd.cd(vim.fn.fnameescape(workspace))

      with_named_buffer(source, function(buf)
        clang_format.ensure_for_buffer(buf)

        expect.falsy(exists(join(external, ".clang-format")))
        expect.falsy(exists(join(workspace, ".clang-format")))
      end)
    end)
  end)
end)
