local M = {}

local skip_dirs = {
  [".git"] = true,
  [".hg"] = true,
  [".svn"] = true,
  [".next"] = true,
  [".venv"] = true,
  ["build"] = true,
  ["dist"] = true,
  ["node_modules"] = true,
  ["target"] = true,
}

local live_search_id = 0
local max_grep_file_size = 1024 * 1024
local max_grep_results = 2000

local function root()
  return vim.fn.getcwd()
end

local function relative(path, base)
  local relpath = vim.fs.relpath(vim.fs.normalize(base), vim.fs.normalize(path))

  if relpath then
    return relpath
  end

  local prefix = vim.fs.normalize(base):gsub("\\", "/")
  local normalized_path = vim.fs.normalize(path):gsub("\\", "/")

  if prefix:sub(-1) ~= "/" then
    prefix = prefix .. "/"
  end

  return normalized_path:sub(#prefix + 1)
end

local function project_files()
  local base = root()
  local files = {}
  local uv = vim.uv or vim.loop

  local function scan(dir)
    local handle = uv.fs_scandir(dir)

    if not handle then
      return
    end

    while true do
      local name, type_name = uv.fs_scandir_next(handle)

      if not name then
        break
      end

      local path = vim.fs.joinpath(dir, name)

      if type_name == "directory" then
        if not skip_dirs[name] then
          scan(path)
        end
      elseif type_name == "file" then
        files[#files + 1] = relative(path, base)
      end
    end
  end

  scan(base)

  table.sort(files)
  return base, files
end

local function should_ignore_case(query)
  if not vim.o.ignorecase then
    return false
  end

  return not (vim.o.smartcase and query:find("%u"))
end

local function is_text_file(path, stat)
  if not stat or stat.type ~= "file" then
    return false
  end

  if stat.size > max_grep_file_size then
    return false
  end

  if stat.size == 0 then
    return true
  end

  local uv = vim.uv or vim.loop
  local fd = uv.fs_open(path, "r", 438)

  if not fd then
    return false
  end

  local chunk = uv.fs_read(fd, math.min(stat.size, 4096), 0) or ""
  uv.fs_close(fd)

  return not chunk:find("\0", 1, true)
end

local function copy_list(items)
  local copy = {}

  for i = 1, #items do
    copy[i] = items[i]
  end

  return copy
end

local function clean_line(line)
  return line:gsub("\t", string.rep(" ", vim.o.tabstop))
end

local function display_line(line)
  local text = clean_line(line)

  if #text <= 500 then
    return text
  end

  return text:sub(1, 500) .. " ..."
end

local function show_text_items(buf_id, items, query)
  local display_items = vim.tbl_map(function(item)
    return {
      text = item.display or item.text,
      path = item.path,
      lnum = item.lnum,
      col = item.col,
    }
  end, items)

  require("mini.pick").default_show(buf_id, display_items, query)
end

local function find_literal_column(line, prompt)
  if prompt == "" then
    return 1
  end

  local ignore_case = should_ignore_case(prompt)
  local haystack = ignore_case and line:lower() or line
  local needle = ignore_case and prompt:lower() or prompt
  local col = haystack:find(needle, 1, true)

  if col then
    return col
  end

  for part in prompt:gmatch("%S+") do
    needle = ignore_case and part:lower() or part
    col = haystack:find(needle, 1, true)

    if col then
      return col
    end
  end

  return 1
end

local function choose_text_item(item)
  if not item then
    return
  end

  local query = require("mini.pick").get_picker_query() or {}
  item.col = find_literal_column(item.line or "", table.concat(query))
  require("mini.pick").default_choose(item)
end

local function make_text_item(file, path, lnum, line)
  return {
    text = string.format("%s:%d: %s", file, lnum, clean_line(line)),
    display = string.format("%s:%d: %s", file, lnum, display_line(line)),
    path = path,
    lnum = lnum,
    col = 1,
    line = line,
  }
end

local function live_text_search(base, files, query, search_id, querytick)
  local pick = require("mini.pick")

  if vim.trim(query) == "" or #query < 2 then
    pick.set_picker_items({}, { do_match = false, querytick = querytick })
    return
  end

  local uv = vim.uv or vim.loop
  local ignore_case = should_ignore_case(query)
  local needle = ignore_case and query:lower() or query
  local items = {}
  local file_index = 1

  local function is_current_search()
    return live_search_id == search_id and pick.is_picker_active() and pick.get_querytick() == querytick
  end

  local function publish()
    if is_current_search() then
      pick.set_picker_items(copy_list(items), { do_match = false, querytick = querytick })
    end
  end

  local function scan_chunk()
    if not is_current_search() then
      return
    end

    local chunk_started = uv.hrtime()

    while file_index <= #files do
      local file = files[file_index]
      file_index = file_index + 1

      local path = vim.fs.joinpath(base, file)
      local stat = uv.fs_stat(path)

      if is_text_file(path, stat) then
        local ok, lines = pcall(vim.fn.readfile, path)

        if ok then
          for lnum, line in ipairs(lines) do
            local haystack = ignore_case and line:lower() or line

            if haystack:find(needle, 1, true) then
              items[#items + 1] = make_text_item(file, path, lnum, line)

              if #items >= max_grep_results then
                publish()
                return
              end
            end
          end
        end
      end

      if uv.hrtime() - chunk_started > 8000000 then
        publish()
        vim.defer_fn(scan_chunk, 1)
        return
      end
    end

    publish()
  end

  vim.defer_fn(scan_chunk, 80)
end

local function choose(items, prompt, callback)
  if #items == 0 then
    vim.notify("No items found", vim.log.levels.WARN)
    return
  end

  vim.ui.select(items, { prompt = prompt }, function(choice)
    if choice then
      callback(choice)
    end
  end)
end

function M.find_files()
  local base, files = project_files()

  vim.ui.input({ prompt = "Find file contains: " }, function(input)
    if input == nil then
      return
    end

    local query = input:lower()
    local matches = {}

    for _, file in ipairs(files) do
      if query == "" or file:lower():find(query, 1, true) then
        matches[#matches + 1] = file
      end
    end

    choose(matches, "Open file", function(choice)
      vim.cmd.edit(vim.fn.fnameescape(vim.fs.joinpath(base, choice)))
    end)
  end)
end

function M.pick_files()
  local base, files = project_files()

  if #files == 0 then
    vim.notify("No files under workspace: " .. base, vim.log.levels.WARN)
    return
  end

  local items = vim.tbl_map(function(file)
    return {
      text = file,
      path = vim.fs.joinpath(base, file),
    }
  end, files)

  require("mini.pick").start({
    source = {
      cwd = base,
      items = items,
      name = "Files: " .. vim.fs.basename(base),
    },
  })
end

function M.pick_grep()
  local base, files = project_files()

  if #files == 0 then
    vim.notify("No files under workspace: " .. base, vim.log.levels.WARN)
    return
  end

  require("mini.pick").start({
    source = {
      cwd = base,
      items = {},
      match = function(_, _, query)
        live_search_id = live_search_id + 1
        live_text_search(base, files, table.concat(query), live_search_id, require("mini.pick").get_querytick())
      end,
      show = show_text_items,
      choose = choose_text_item,
      name = "Text: " .. vim.fs.basename(base),
    },
  })
end

function M.grep_project()
  local base, files = project_files()

  vim.ui.input({ prompt = "Search text: " }, function(input)
    if input == nil or input == "" then
      return
    end

    local needle = input:lower()
    local items = {}
    local uv = vim.uv or vim.loop

    for _, file in ipairs(files) do
      local path = vim.fs.joinpath(base, file)
      local stat = uv.fs_stat(path)

      if stat and stat.size <= 1024 * 1024 then
        local ok, lines = pcall(vim.fn.readfile, path)

        if ok then
          for lnum, line in ipairs(lines) do
            local col = line:lower():find(needle, 1, true)

            if col then
              items[#items + 1] = {
                filename = path,
                lnum = lnum,
                col = col,
                text = line,
              }
            end
          end
        end
      end
    end

    if #items == 0 then
      vim.notify("No matches found", vim.log.levels.INFO)
      return
    end

    vim.fn.setqflist({}, " ", {
      title = "Search: " .. input,
      items = items,
    })
    vim.cmd.copen()
  end)
end

function M.buffers()
  local items = {}

  for _, buf in ipairs(vim.api.nvim_list_bufs()) do
    if vim.bo[buf].buflisted then
      local name = vim.api.nvim_buf_get_name(buf)
      items[#items + 1] = name ~= "" and name or ("[No Name] " .. buf)
    end
  end

  choose(items, "Open buffer", function(choice)
    for _, buf in ipairs(vim.api.nvim_list_bufs()) do
      local name = vim.api.nvim_buf_get_name(buf)

      if choice == name or choice == ("[No Name] " .. buf) then
        vim.api.nvim_set_current_buf(buf)
        return
      end
    end
  end)
end

function M.oldfiles()
  local files = vim.tbl_filter(function(file)
    return vim.fn.filereadable(file) == 1
  end, vim.v.oldfiles)

  choose(files, "Open recent file", function(choice)
    vim.cmd.edit(vim.fn.fnameescape(choice))
  end)
end

return M
