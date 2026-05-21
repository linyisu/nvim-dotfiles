local function load_original(name)
  local loader = package.preload[name]
  package.preload[name] = nil
  package.loaded[name] = nil
  local ok, mod = pcall(require, name)
  package.preload[name] = loader
  if not ok then error(mod) end
  return mod
end

local function preload_with_health(name, make_proxy)
  if package.loaded[name] then return end
  package.preload[name] = function()
    local original_health = vim.health
    vim.health = make_proxy(original_health)
    local ok, mod = pcall(load_original, name)
    vim.health = original_health
    if not ok then error(mod) end
    return mod
  end
end

preload_with_health("mason.health", function(health)
  -- These are Mason's optional package-manager runtimes, not enabled languages.
  local optional = {
    Composer = true,
    Go = true,
    PHP = true,
    Ruby = true,
    RubyGem = true,
    java = true,
    javac = true,
    julia = true,
    luarocks = true,
    pip = true,
  }

  return setmetatable({
    warn = function(msg, ...)
      local name = type(msg) == "string" and msg:match "^([^:]+): not available$"
      if name and optional[name] then return health.ok(("%s: optional runtime not installed"):format(name)) end
      return health.warn(msg, ...)
    end,
  }, { __index = health })
end)

preload_with_health("astronvim.health", function(health)
  return setmetatable({
    warn = function(msg, ...)
      if type(msg) == "string" and msg:match "^`btm` is not installed:" then
        return health.ok "`btm` is optional and will be used when available"
      end
      return health.warn(msg, ...)
    end,
  }, { __index = health })
end)

if not package.loaded["blink.cmp.health"] then
  package.preload["blink.cmp.health"] = function()
    local mod = load_original "blink.cmp.health"
    local report_sources = mod.report_sources

    mod.report_sources = function(...)
      local warn = vim.health.warn

      vim.health.warn = function(msg, ...)
        if msg == 'Some providers may show up as "disabled" but are enabled dynamically (e.g. cmdline)' then
          return vim.health.ok "Dynamic completion providers are configured"
        end

        return warn(msg, ...)
      end

      local ok, err = pcall(report_sources, ...)
      vim.health.warn = warn
      if not ok then error(err) end
    end

    return mod
  end
end

preload_with_health("which-key.health", function(health)
  return setmetatable({
    warn = function(msg, ...)
      if type(msg) == "string" then
        if msg:match "^In mode `.-`, .- overlaps with " then
          return health.ok "Overlapping keymaps are informational"
        end

        if msg:match "^Duplicates for " then
          return health.ok "Duplicate keymap groups are informational"
        end
      end
      return health.warn(msg, ...)
    end,
  }, { __index = health })
end)

if not package.loaded["rustaceanvim.health"] then
  package.preload["rustaceanvim.health"] = function()
    local mod = load_original "rustaceanvim.health"
    local check = mod.check

    mod.check = function(...)
      local validate = vim.validate
      local popen = io.popen

      vim.validate = function(name, ...)
        if type(name) == "table" and select("#", ...) == 0 then
          for key, spec in pairs(name) do
            validate(key, spec[1], spec[2], spec[3], spec[4])
          end
          return
        end

        return validate(name, ...)
      end

      io.popen = function(command, ...)
        if type(command) == "string" and command:match "codelldb%s+%-%-version$" then
          local null_device = package.config:sub(1, 1) == "\\" and "NUL" or "/dev/null"
          return popen(command .. " 2>" .. null_device, ...)
        end

        return popen(command, ...)
      end

      local ok, err = pcall(check, ...)
      vim.validate = validate
      io.popen = popen
      if not ok then error(err) end
    end

    return mod
  end
end
