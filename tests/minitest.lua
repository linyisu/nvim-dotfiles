local M = {}

local tests = {}
local current_suite = nil

local function fail(message, level)
  error(message, level or 2)
end

local function format_value(value)
  return vim.inspect(value)
end

local function assert_equal(actual, expected, message)
  if not vim.deep_equal(actual, expected) then
    fail((message or "values are not equal") .. "\nexpected: " .. format_value(expected) .. "\nactual: " .. format_value(actual), 3)
  end
end

local function assert_truthy(value, message)
  if not value then
    fail(message or ("expected truthy value, got " .. format_value(value)), 3)
  end
end

local function assert_falsy(value, message)
  if value then
    fail(message or ("expected falsy value, got " .. format_value(value)), 3)
  end
end

local function assert_contains(haystack, needle, message)
  if not tostring(haystack):find(tostring(needle), 1, true) then
    fail((message or "expected value to contain substring") .. "\nneedle: " .. needle .. "\nhaystack: " .. tostring(haystack), 3)
  end
end

function M.describe(name, callback)
  local previous = current_suite
  current_suite = previous and (previous .. " " .. name) or name
  callback()
  current_suite = previous
end

function M.it(name, callback)
  tests[#tests + 1] = {
    name = (current_suite and (current_suite .. " ") or "") .. name,
    callback = callback,
  }
end

M.assert = {
  equal = assert_equal,
  truthy = assert_truthy,
  falsy = assert_falsy,
  contains = assert_contains,
}

local function reset_package(module)
  package.loaded[module] = nil
end

function M.reload(module)
  reset_package(module)
  return require(module)
end

function M.with_temp_dir(callback)
  local root = vim.fs.joinpath(vim.fn.tempname())
  local created = vim.fn.mkdir(root, "p") == 1

  if not created then
    fail("failed to create temp dir: " .. root)
  end

  local previous = vim.fn.getcwd()

  vim.cmd.cd(vim.fn.fnameescape(root))

  local ok, result = pcall(callback, root)

  vim.cmd.cd(vim.fn.fnameescape(previous))
  vim.fn.delete(root, "rf")

  if not ok then
    fail(result, 0)
  end

  return result
end

function M.run()
  local failures = {}

  if #tests == 0 then
    print("FAIL no tests were registered")
    vim.cmd.cquit(1)
    return
  end

  for _, test in ipairs(tests) do
    local ok, err = xpcall(test.callback, debug.traceback)

    if ok then
      print("PASS " .. test.name)
    else
      failures[#failures + 1] = {
        name = test.name,
        error = err,
      }
      print("FAIL " .. test.name)
      print(err)
    end
  end

  print(string.format("Tests: %d passed, %d failed, %d total", #tests - #failures, #failures, #tests))

  if #failures > 0 then
    vim.cmd.cquit(1)
  end
end

return M
