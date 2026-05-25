local M = {}

local function comment_parts()
  local commentstring = vim.bo.commentstring

  if commentstring == "" or not commentstring:find("%%s") then
    commentstring = "# %s"
  end

  local left, right = commentstring:match("^(.*)%%s(.*)$")
  return vim.trim(left or ""), vim.trim(right or "")
end

local function is_commented(line, left, right)
  local trimmed = vim.trim(line)

  if left ~= "" and not vim.startswith(trimmed, left) then
    return false
  end

  if right ~= "" and not vim.endswith(trimmed, right) then
    return false
  end

  return left ~= "" or right ~= ""
end

local function uncomment(line, left, right)
  local indent, body = line:match("^(%s*)(.*)$")

  if left ~= "" then
    body = body:gsub("^%s*" .. vim.pesc(left) .. "%s?", "", 1)
  end

  if right ~= "" then
    body = body:gsub("%s?" .. vim.pesc(right) .. "%s*$", "", 1)
  end

  return indent .. body
end

local function comment(line, left, right)
  local indent, body = line:match("^(%s*)(.*)$")
  local suffix = right ~= "" and (" " .. right) or ""
  return indent .. left .. " " .. body .. suffix
end

function M.toggle_lines(first, last)
  local left, right = comment_parts()
  local lines = vim.api.nvim_buf_get_lines(0, first - 1, last, false)
  local uncomment_all = true

  for _, line in ipairs(lines) do
    if vim.trim(line) ~= "" and not is_commented(line, left, right) then
      uncomment_all = false
      break
    end
  end

  for index, line in ipairs(lines) do
    if vim.trim(line) ~= "" then
      if uncomment_all then
        lines[index] = uncomment(line, left, right)
      else
        lines[index] = comment(line, left, right)
      end
    end
  end

  vim.api.nvim_buf_set_lines(0, first - 1, last, false, lines)
end

function M.toggle_current_line()
  local line = vim.fn.line(".")
  M.toggle_lines(line, line)
end

function M.toggle_selection()
  local first = vim.fn.line("v")
  local last = vim.fn.line(".")

  if first > last then
    first, last = last, first
  end

  M.toggle_lines(first, last)
end

return M
