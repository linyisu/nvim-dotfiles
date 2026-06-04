local source = debug.getinfo(1, "S").source:sub(2)
local config_root = vim.fn.fnamemodify(source, ":p:h:h")
local lua_root = vim.fs.normalize(config_root):gsub("\\", "/")

vim.go.loadplugins = true
vim.opt.runtimepath:prepend(config_root)
package.path = table.concat({
  lua_root .. "/?.lua",
  lua_root .. "/?/init.lua",
  package.path,
}, ";")

if not package.loaded["config.lazy"] then
  dofile(vim.fs.joinpath(config_root, "init.lua"))
end

local minitest = require("tests.minitest")

_G.describe = minitest.describe
_G.it = minitest.it
_G.expect = minitest.assert
_G.reload = minitest.reload
_G.with_temp_dir = minitest.with_temp_dir

local spec_pattern = vim.fs.joinpath(config_root, "tests", "specs", "*_spec.lua")
local spec_files = vim.fn.glob(spec_pattern, false, true)

table.sort(spec_files)

if #spec_files == 0 then
  error("no test specs found under " .. spec_pattern)
end

for _, file in ipairs(spec_files) do
  local relative = vim.fs.normalize(file):sub(#vim.fs.normalize(config_root) + 2)
  local spec = relative:gsub("%.lua$", ""):gsub("[/\\]", ".")

  require(spec)
end

minitest.run()
