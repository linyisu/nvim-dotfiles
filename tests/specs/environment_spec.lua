local function table_contains(values, expected)
  for _, value in ipairs(values or {}) do
    if value == expected then
      return true
    end
  end

  return false
end

local function with_neovide_state(state, callback)
  local neovide_global_keys = {
    "neovide_theme",
    "neovide_scale_factor",
    "neovide_padding_top",
    "neovide_padding_bottom",
    "neovide_padding_left",
    "neovide_padding_right",
    "neovide_remember_window_size",
    "neovide_confirm_quit",
    "neovide_hide_mouse_when_typing",
    "neovide_scroll_animation_length",
    "neovide_position_animation_length",
    "neovide_cursor_animation_length",
    "neovide_cursor_short_animation_length",
    "neovide_cursor_trail_size",
  }
  local previous_neovide = vim.g.neovide
  local previous_font_env = vim.env.NEOVIDE_FONT
  local previous_guifont = vim.o.guifont
  local previous_background = vim.o.background
  local previous_globals = {}

  for _, key in ipairs(neovide_global_keys) do
    previous_globals[key] = vim.g[key]
  end

  vim.g.neovide = state.neovide
  vim.env.NEOVIDE_FONT = state.font_env
  vim.o.guifont = state.guifont or previous_guifont
  vim.o.background = state.background or previous_background

  local ok, result = pcall(function()
    package.loaded["core.neovide"] = nil
    return callback()
  end)

  vim.g.neovide = previous_neovide
  vim.env.NEOVIDE_FONT = previous_font_env
  vim.o.guifont = previous_guifont
  vim.o.background = previous_background
  for _, key in ipairs(neovide_global_keys) do
    vim.g[key] = previous_globals[key]
  end
  package.loaded["core.neovide"] = nil

  if not ok then
    error(result, 0)
  end

  return result
end

describe("environment configuration", function()
  it("sets stable editor options", function()
    local listchars = vim.opt.listchars:get()

    expect.equal(vim.o.number, true)
    expect.equal(vim.o.relativenumber, true)
    expect.equal(vim.o.signcolumn, "yes")
    expect.equal(vim.o.cursorline, true)

    expect.equal(vim.o.expandtab, true)
    expect.equal(vim.o.shiftwidth, 2)
    expect.equal(vim.o.softtabstop, 2)
    expect.equal(vim.o.tabstop, 2)
    expect.equal(vim.o.smartindent, true)
    expect.equal(vim.o.breakindent, true)

    expect.equal(vim.o.wrap, false)
    expect.equal(vim.o.scrolloff, 8)
    expect.equal(vim.o.sidescrolloff, 8)
    expect.equal(vim.o.smoothscroll, false)
    expect.equal(vim.o.splitbelow, true)
    expect.equal(vim.o.splitright, true)
    expect.equal(vim.o.mouse, "a")

    expect.equal(vim.o.undofile, true)
    expect.equal(vim.o.swapfile, false)
    expect.equal(vim.o.confirm, true)
    expect.equal(vim.o.list, true)
    expect.equal(listchars.tab, "> ")
    expect.equal(listchars.trail, ".")
    expect.equal(listchars.nbsp, "+")
    expect.truthy(table_contains(vim.opt.path:get(), "**"), "expected recursive path lookup")
    expect.equal(vim.g.snacks_animate, false)
    expect.equal(vim.g.snacks_scroll, false)
  end)

  it("disables animated motion in Snacks UI components", function()
    local opts = LazyVim.opts("snacks.nvim")

    expect.equal(opts.scroll and opts.scroll.enabled, false)
    expect.equal(opts.scope and opts.scope.enabled, false)
    expect.equal(opts.indent and opts.indent.animate and opts.indent.animate.enabled, false)
    expect.equal(opts.indent and opts.indent.scope and opts.indent.scope.enabled, false)
  end)

  it("configures mini.files navigation mappings and explorer behavior", function()
    local opts = LazyVim.opts("mini.files")
    local mappings = opts.mappings or {}

    expect.equal(mappings.go_in, "")
    expect.equal(mappings.go_in_plus, "")
    expect.equal(mappings.go_out, "")
    expect.equal(mappings.go_out_plus, "")
    expect.equal(mappings.close, "q")
    expect.equal(opts.options and opts.options.use_as_default_explorer, true)
    expect.equal(opts.windows and opts.windows.max_number, 1)
    expect.equal(opts.windows and opts.windows.preview, false)
  end)

  it("uses a centered mini.pick window with minimum dimensions", function()
    local opts = LazyVim.opts("mini.pick")
    local config = opts.window and opts.window.config
    local previous_lines = vim.o.lines
    local previous_columns = vim.o.columns

    expect.equal(type(config), "function")

    local ok, result = pcall(function()
      vim.o.lines = 40
      vim.o.columns = 100

      local large = config()

      expect.equal(large.anchor, "NW")
      expect.equal(large.border, "rounded")
      expect.equal(large.height, 28)
      expect.equal(large.width, 70)
      expect.equal(large.row, 6)
      expect.equal(large.col, 15)

      vim.o.lines = 8
      vim.o.columns = 20

      local small = config()

      expect.equal(small.height, 10)
      expect.equal(small.width, 40)
    end)

    vim.o.lines = previous_lines
    vim.o.columns = previous_columns

    if not ok then
      error(result, 0)
    end
  end)

  it("does not change guifont when Neovide is disabled or unset", function()
    with_neovide_state({ neovide = false, font_env = "UnitTestFont:h13", guifont = "BeforeFalse:h11" }, function()
      reload("core.neovide")
      expect.equal(vim.o.guifont, "BeforeFalse:h11")
    end)

    with_neovide_state({ neovide = nil, font_env = "UnitTestFont:h13", guifont = "BeforeNil:h11" }, function()
      reload("core.neovide")
      expect.equal(vim.o.guifont, "BeforeNil:h11")
    end)
  end)

  it("uses NEOVIDE_FONT and dark Neovide defaults when Neovide is enabled", function()
    with_neovide_state({ neovide = true, font_env = "UnitTestFont:h13", guifont = "BeforeEnabled:h11" }, function()
      reload("core.neovide")

      expect.equal(vim.o.guifont, "UnitTestFont:h13")
      expect.equal(vim.o.background, "dark")
      expect.equal(vim.g.neovide_theme, "dark")
      expect.equal(vim.g.neovide_scale_factor, 1.0)
      expect.equal(vim.g.neovide_padding_top, 4)
      expect.equal(vim.g.neovide_padding_bottom, 4)
      expect.equal(vim.g.neovide_padding_left, 6)
      expect.equal(vim.g.neovide_padding_right, 6)
      expect.equal(vim.g.neovide_scroll_animation_length, 0)
      expect.equal(vim.g.neovide_position_animation_length, 0)
      expect.equal(vim.g.neovide_cursor_animation_length, 0)
      expect.equal(vim.g.neovide_cursor_short_animation_length, 0)
      expect.equal(vim.g.neovide_cursor_trail_size, 0)
    end)
  end)

  it("uses the platform default Neovide font when NEOVIDE_FONT is not set", function()
    local expected_font = "monospace:h12"

    if vim.fn.has("win32") == 1 then
      expected_font = "JetBrainsMono NF:h12"
    elseif vim.fn.has("mac") == 1 then
      expected_font = "Menlo:h12"
    end

    with_neovide_state({ neovide = true, font_env = nil, guifont = "BeforeDefault:h11" }, function()
      reload("core.neovide")

      expect.equal(vim.o.guifont, expected_font)
    end)
  end)
end)
