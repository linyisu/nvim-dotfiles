local function ensure_very_lazy()
  if package.loaded["config.keymaps"] and package.loaded["config.autocmds"] then
    return
  end

  vim.api.nvim_exec_autocmds("User", {
    pattern = "VeryLazy",
    modeline = false,
  })
end

local function get_mapping(mode, lhs)
  local mapping = vim.fn.maparg(lhs, mode, false, true)

  if type(mapping) ~= "table" or next(mapping) == nil then
    return nil
  end

  return mapping
end

local function expect_mapping(mode, lhs)
  local mapping = get_mapping(mode, lhs)

  expect.truthy(mapping, string.format("expected %s-mode mapping for %s", mode, lhs))

  return mapping
end

local function is_noop_mapping(mapping)
  local rhs = (mapping.rhs or ""):lower():gsub("%s+", "")

  return not mapping.callback and (rhs == "" or rhs == "<nop>")
end

local function dashboard_key(opts, key)
  local keys = opts.dashboard and opts.dashboard.preset and opts.dashboard.preset.keys or {}

  for _, item in ipairs(keys) do
    if item.key == key then
      return item
    end
  end
end

local function list_has(values, expected)
  for _, value in ipairs(values or {}) do
    if value == expected then
      return true
    end
  end

  return false
end

local function config_root()
  local source = debug.getinfo(1, "S").source:sub(2)

  return vim.fn.fnamemodify(source, ":p:h:h:h")
end

local expected_logo = table.concat({
  "██╗     ██╗███╗   ██╗██╗   ██╗██╗███████╗██╗   ██╗",
  "██║     ██║████╗  ██║╚██╗ ██╔╝██║██╔════╝██║   ██║",
  "██║     ██║██╔██╗ ██║ ╚████╔╝ ██║███████╗██║   ██║",
  "██║     ██║██║╚██╗██║  ╚██╔╝  ██║╚════██║██║   ██║",
  "███████╗██║██║ ╚████║   ██║   ██║███████║╚██████╔╝",
  "╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝ ╚═════╝ ",
}, "\n")

describe("LazyVim config integration", function()
  it("loads deferred config through the VeryLazy event", function()
    ensure_very_lazy()

    expect.truthy(package.loaded["config.keymaps"], "expected config.keymaps to be loaded")
    expect.truthy(package.loaded["config.autocmds"], "expected config.autocmds to be loaded")
    expect_mapping("n", "<leader>e")
  end)

  it("uses space for leader and localleader", function()
    ensure_very_lazy()

    expect.equal(vim.g.mapleader, " ")
    expect.equal(vim.g.maplocalleader, " ")
  end)

  it("defines common leader and movement keymaps", function()
    ensure_very_lazy()

    for _, lhs in ipairs({ "<leader>e", "<leader>ff", "<leader>fg", "<leader>rr", "<leader>rn" }) do
      expect_mapping("n", lhs)
    end

    expect.equal(expect_mapping("n", "<leader>rr").desc, "Run current file or Rust project")
    expect.equal(expect_mapping("n", "<leader>rn").desc, "Rename symbol")

    for _, lhs in ipairs({ "<A-Up>", "<A-Down>" }) do
      expect_mapping("n", lhs)
      expect_mapping("x", lhs)
    end
  end)

  it("does not define unwanted single-key save and quit leader actions", function()
    ensure_very_lazy()

    for _, lhs in ipairs({ "<leader>w", "<leader>q", "<leader>Q", "<leader>lr" }) do
      local mapping = get_mapping("n", lhs)
      local allowed = mapping == nil or is_noop_mapping(mapping)

      expect.truthy(
        allowed,
        string.format("expected no actionable n-mode mapping for %s, got %s", lhs, vim.inspect(mapping))
      )
    end
  end)

  it("still keeps LazyVim's longer quit/session/window mappings available", function()
    ensure_very_lazy()

    expect_mapping("n", "<leader>qq")
    expect_mapping("n", "<leader>qs")
    expect_mapping("n", "<leader>wd")
  end)

  it("defines user commands", function()
    ensure_very_lazy()

    local commands = vim.api.nvim_get_commands({})

    for _, name in ipairs({ "RunFile", "ClangFormatInit", "ClangFormatOpen", "Home" }) do
      expect.truthy(commands[name], "expected user command " .. name)
    end
  end)

  it("loads the upstream CompetiTest plugin with testcase keymaps", function()
    ensure_very_lazy()

    local plugin = require("lazy.core.config").plugins["competitest.nvim"]
    local opts = LazyVim.opts("competitest.nvim")
    local uv = vim.uv or vim.loop
    local home = vim.fs.normalize((uv.os_homedir and uv.os_homedir()) or vim.fn.expand("~"))
    local acm = vim.fs.joinpath(home, "acm")
    local target = package.config:sub(1, 1) == "\\" and "$(FNOEXT).exe" or "$(FNOEXT)"
    local run = package.config:sub(1, 1) == "\\" and "./$(FNOEXT).exe" or "./$(FNOEXT)"

    expect.truthy(plugin, "expected competitest.nvim plugin spec")
    expect.equal(plugin[1], "xeluxee/competitest.nvim")
    expect.equal(expect_mapping("n", "<leader>tt").desc, "Run testcases")
    expect.equal(expect_mapping("n", "<leader>ta").desc, "Add testcase")
    expect.equal(expect_mapping("n", "<leader>te").desc, "Edit testcase")
    expect.equal(expect_mapping("n", "<leader>td").desc, "Delete testcase")
    expect.equal(expect_mapping("n", "<leader>tc").desc, "Receive contest")
    expect.equal(opts.testcases_use_single_file, true)
    expect.equal(opts.received_files_extension, "cpp")
    expect.equal(opts.received_contests_prompt_extension, false)
    expect.equal(opts.runner_ui and opts.runner_ui.interface, "split")
    expect.equal(opts.split_ui and opts.split_ui.position, "right")
    expect.equal(opts.split_ui and opts.split_ui.total_width, 0.4)
    expect.equal(opts.compile_command and opts.compile_command.cpp and opts.compile_command.cpp.args, {
      "-std=c++23",
      "-O2",
      "-Wall",
      "$(FNAME)",
      "-o",
      target,
    })
    expect.equal(opts.received_problems_path, vim.fs.joinpath(acm, "problems", "$(JUDGE)", "$(PROBLEM)", "$(PROBLEM).$(FEXT)"))
    expect.equal(opts.received_contests_directory, vim.fs.joinpath(acm, "contests", "$(JUDGE)", "$(CONTEST)"))
    expect.equal(opts.received_contests_problems_path, "$(PROBLEM)/$(PROBLEM).$(FEXT)")
    expect.equal(opts.run_command and opts.run_command.cpp and opts.run_command.cpp.exec, run)
  end)

  it("opens a directory argument through the configured explorer on VimEnter", function()
    with_temp_dir(function(root)
      local workspace = vim.fs.joinpath(root, "workspace")
      expect.equal(vim.fn.mkdir(workspace, "p"), 1)

      local root_path = config_root()
      local lua_root = vim.fs.normalize(root_path):gsub("\\", "/")
      local prelude = string.format(
        "lua vim.opt.runtimepath:prepend(%q); package.path = %q .. '/?.lua;' .. %q .. '/?/init.lua;' .. package.path",
        root_path,
        lua_root,
        lua_root
      )
      local report_cwd = "autocmd VimEnter * lua vim.schedule(function() print('STARTUP_CWD=' .. vim.fn.getcwd()); vim.cmd('qa!') end)"
      local result = vim.system({
        vim.v.progpath,
        "--headless",
        "-i",
        "NONE",
        "--cmd",
        prelude,
        "--cmd",
        report_cwd,
        "-u",
        vim.fs.joinpath(root_path, "init.lua"),
        workspace,
      }, {
        cwd = root_path,
        text = true,
      }):wait()
      local output = table.concat({ result.stdout or "", result.stderr or "" }, "\n")
      local cwd = output:match("STARTUP_CWD=([^\r\n]+)")

      expect.equal(result.code, 0, result.stderr)
      expect.truthy(cwd, output)
      expect.equal(vim.fs.normalize(cwd), vim.fs.normalize(workspace))
    end)
  end)

  it("configures LazyVim theme and final dashboard actions", function()
    ensure_very_lazy()

    local lazyvim_opts = LazyVim.opts("LazyVim")
    local snacks_opts = LazyVim.opts("snacks.nvim")
    local header = snacks_opts.dashboard and snacks_opts.dashboard.preset and snacks_opts.dashboard.preset.header

    expect.equal(lazyvim_opts.colorscheme, "tokyonight")
    expect.equal(header, expected_logo)

    for _, key in ipairs({ "f", "g", "e", "n", "r", "c", "l", "q" }) do
      expect.truthy(dashboard_key(snacks_opts, key), "expected dashboard key " .. key)
    end

    expect.contains(dashboard_key(snacks_opts, "f").action, "core.search")
    expect.contains(dashboard_key(snacks_opts, "g").action, "pick_grep")
    expect.contains(dashboard_key(snacks_opts, "e").action, "core.file_explorer")

    local expected_icons = {
      f = " ",
      g = " ",
      e = " ",
      n = " ",
      r = " ",
      c = " ",
      l = "󰒲 ",
      q = " ",
    }

    for key, icon in pairs(expected_icons) do
      expect.equal(dashboard_key(snacks_opts, key).icon, icon)
    end
  end)

  it("keeps the bufferline visible when only one buffer is open", function()
    ensure_very_lazy()

    local opts = LazyVim.opts("bufferline.nvim")

    expect.equal(opts.options and opts.options.always_show_bufferline, true)
  end)

  it("configures final clangd options with the expected style flags", function()
    ensure_very_lazy()

    local opts = LazyVim.opts("nvim-lspconfig")
    local cmd = opts.servers and opts.servers.clangd and opts.servers.clangd.cmd
    local filetypes = opts.servers and opts.servers.clangd and opts.servers.clangd.filetypes

    expect.equal(cmd, {
      "clangd",
      "--background-index",
      "--clang-tidy",
      "--completion-style=detailed",
      "--fallback-style=Google",
      "--header-insertion=never",
    })
    expect.equal(filetypes, { "c", "cpp", "objc", "objcpp", "cuda" })
  end)

  it("configures Rust parsing and rust-analyzer diagnostics", function()
    ensure_very_lazy()

    local treesitter_opts = LazyVim.opts("nvim-treesitter")
    local lsp_opts = LazyVim.opts("nvim-lspconfig")
    local rust = lsp_opts.servers and lsp_opts.servers.rust_analyzer
    local settings = rust and rust.settings and rust.settings["rust-analyzer"]

    expect.truthy(list_has(treesitter_opts.ensure_installed, "rust"), "expected rust treesitter parser")
    expect.truthy(list_has(treesitter_opts.ensure_installed, "toml"), "expected Cargo.toml treesitter parser")
    expect.equal(settings and settings.cargo and settings.cargo.allFeatures, true)
    expect.equal(settings and settings.diagnostics and settings.diagnostics.enable, true)
    expect.equal(settings and settings.procMacro and settings.procMacro.enable, true)
  end)

  it("formats C-family files with clang-format from Mason", function()
    ensure_very_lazy()

    local conform_opts = LazyVim.opts("conform.nvim")
    local mason_opts = LazyVim.opts("mason.nvim")

    for _, filetype in ipairs({ "c", "cpp", "cuda", "objc", "objcpp" }) do
      expect.equal(conform_opts.formatters_by_ft and conform_opts.formatters_by_ft[filetype], { "clang_format" })
    end

    expect.equal(conform_opts.formatters and conform_opts.formatters.clang_format and conform_opts.formatters.clang_format.prepend_args, {
      "--style=file",
    })
    expect.truthy(list_has(mason_opts.ensure_installed, "clang-format"), "expected Mason to install clang-format")
  end)

  it("uses LuaSnip for snippets and loads the C++ snippet module", function()
    ensure_very_lazy()

    local plugins = require("lazy.core.config").plugins
    local luasnip = plugins["LuaSnip"]
    local friendly_snippets = plugins["friendly-snippets"]
    local blink_opts = LazyVim.opts("blink.cmp")
    local luasnip_opts = LazyVim.opts("LuaSnip")

    expect.truthy(luasnip, "expected LuaSnip plugin spec")
    expect.truthy(
      friendly_snippets == nil or friendly_snippets.enabled == false,
      "expected friendly-snippets to be disabled"
    )
    expect.truthy(list_has(type(luasnip.event) == "table" and luasnip.event or { luasnip.event }, "InsertEnter"))
    expect.equal(luasnip.dependencies and #luasnip.dependencies or 0, 0)
    expect.equal(luasnip_opts.history, true)
    expect.equal(luasnip_opts.delete_check_events, "TextChanged")
    expect.equal(blink_opts.keymap and blink_opts.keymap.preset, "super-tab")
    expect.equal(blink_opts.snippets and blink_opts.snippets.preset, "luasnip")

    local source = table.concat(vim.fn.readfile(vim.fs.joinpath(config_root(), "lua", "plugins", "snippets.lua")), "\n")

    expect.equal(source:find("from_vscode", 1, true), nil)

    require("lazy").load({ plugins = { "LuaSnip" }, wait = true })

    expect.truthy(package.loaded["core.cpp_snippets"], "expected LuaSnip to load the C++ snippet module")
  end)
end)
