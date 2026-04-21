-- if true then return {} end -- WARN: REMOVE THIS LINE TO ACTIVATE THIS FILE

-- AstroCore provides a central place to modify mappings, vim options, autocommands, and more!
-- Configuration documentation can be found with `:h astrocore`
-- NOTE: We highly recommend setting up the Lua Language Server (`:LspInstall lua_ls`)
--       as this provides autocomplete and documentation while editing

---@type LazySpec
return {
  "AstroNvim/astrocore",
  ---@type AstroCoreOpts
  opts = {
    -- Configure core features of AstroNvim
    features = {
      large_buf = { size = 1024 * 256, lines = 10000 }, -- set global limits for large files for disabling features like treesitter
      autopairs = true, -- enable autopairs at start
      cmp = true, -- enable completion at start
      diagnostics = { virtual_text = true, virtual_lines = false }, -- diagnostic settings on startup
      highlighturl = true, -- highlight URLs at start
      notifications = true, -- enable notifications at start
    },
    -- Diagnostics configuration (for vim.diagnostics.config({...})) when diagnostics are on
    diagnostics = {
      virtual_text = true,
      underline = true,
    },
    -- Autocommands
    autocmds = {
      no_comment_continuation = {
        {
          event = "FileType",
          desc = "Disable comment continuation on o/O",
          callback = function()
            vim.schedule(function()
              vim.opt_local.formatoptions:remove({ "o", "r" })
            end)
          end,
        },
      },
      cpp_indent = {
        {
          event = "FileType",
          pattern = { "c", "cpp" },
          desc = "Set 4-space indent for C/C++",
          callback = function()
            vim.opt_local.tabstop = 4
            vim.opt_local.shiftwidth = 4
            vim.opt_local.softtabstop = 4
          end,
        },
      },
    },
    -- vim options can be configured here
    options = {
      opt = { -- vim.opt.<key>
        relativenumber = true, -- sets vim.opt.relativenumber
        number = true, -- sets vim.opt.number
        spell = false, -- sets vim.opt.spell
        signcolumn = "yes", -- sets vim.opt.signcolumn to yes
        wrap = false, -- sets vim.opt.wrap
        guifont = "JetBrainsMono Nerd Font:h14",
        clipboard = "unnamedplus", -- sync with system clipboard
      },
      g = { -- vim.g.<key>
        -- configure global vim variables (vim.g)
        -- NOTE: `mapleader` and `maplocalleader` must be set in the AstroNvim opts or before `lazy.setup`
        -- This can be found in the `lua/lazy_setup.lua` file
      },
    },
    -- Mappings can be configured through AstroCore as well.
    -- NOTE: keycodes follow the casing in the vimdocs. For example, `<Leader>` must be capitalized
    mappings = {
      -- first key is the mode
      n = {
        -- second key is the lefthand side of the map

        -- navigate buffer tabs
        ["<C-x>"] = { function() require("astrocore.buffer").nav(vim.v.count1) end, desc = "Next buffer" },
        ["<C-z>"] = { function() require("astrocore.buffer").nav(-vim.v.count1) end, desc = "Previous buffer" },

        -- mappings seen under group name "Buffer"
        ["<Leader>bd"] = {
          function()
            require("astroui.status.heirline").buffer_picker(
              function(bufnr) require("astrocore.buffer").close(bufnr) end
            )
          end,
          desc = "Close buffer from tabline",
        },

        -- tables with just a `desc` key will be registered with which-key if it's installed
        -- this is useful for naming menus
        -- ["<Leader>b"] = { desc = "Buffers" },

        -- disable basic keymaps
        ["<Leader>w"] = false,
        ["<Leader>q"] = false,
        ["<Leader>Q"] = false,
        ["<Leader>n"] = false,
        ["<Leader>/"] = false,

        -- move terminal keymaps from <Leader>t to <Leader>T
        ["<Leader>t"]  = false,
        ["<Leader>tf"] = false,
        ["<Leader>th"] = false,
        ["<Leader>tv"] = false,
        ["<Leader>tt"] = false,
        ["<Leader>tn"] = false,
        ["<Leader>tp"] = false,
        ["<Leader>tu"] = false,
        ["<Leader>tl"] = false,
        ["<Leader>T"]  = { desc = "Terminal" },
        ["<Leader>Tf"] = { "<Cmd>ToggleTerm direction=float<CR>",            desc = "ToggleTerm float" },
        ["<Leader>Th"] = { "<Cmd>ToggleTerm size=10 direction=horizontal<CR>", desc = "ToggleTerm horizontal" },
        ["<Leader>Tv"] = { "<Cmd>ToggleTerm size=80 direction=vertical<CR>",   desc = "ToggleTerm vertical" },
        ["<Leader>Tg"] = { function() require("astrocore").toggle_term_cmd "lazygit" end, desc = "Lazygit" },

        -- disable git keymaps
        ["<Leader>g"]  = false,
        ["<Leader>gg"] = false,
        ["<Leader>gb"] = false,
        ["<Leader>gc"] = false,
        ["<Leader>gC"] = false,
        ["<Leader>gt"] = false,
        ["<Leader>gT"] = false,
        ["<Leader>go"] = false,
        ["<Leader>gl"] = false,
        ["<Leader>gp"] = false,
        ["<Leader>gr"] = false,
        ["<Leader>gR"] = false,
        ["<Leader>gs"] = false,
        ["<Leader>gS"] = false,
        ["<Leader>gd"] = false,
      },
      x = {
        ["<Leader>/"] = false,
        ["<Leader>go"] = false,
        ["<Leader>la"] = false,
      },
    },
  },
}
