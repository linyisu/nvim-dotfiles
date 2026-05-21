return {
  "AstroNvim/astrocore",
  opts = function(_, opts)
    local user_opts = {
      features = {
        large_buf = { size = 1024 * 256, lines = 10000 },
        autopairs = true,
        cmp = true,
        diagnostics = { virtual_text = true, virtual_lines = false },
        highlighturl = true,
        notifications = true,
      },
      diagnostics = {
        virtual_text = true,
        underline = true,
        jump = {
          on_jump = function(_, bufnr)
            vim.diagnostic.open_float {
              bufnr = bufnr,
              scope = "cursor",
              focus = false,
            }
          end,
        },
      },
      autocmds = {
        no_comment_continuation = {
          {
            event = "FileType",
            desc = "Disable comment continuation on o/O",
            callback = function()
              vim.schedule(function() vim.opt_local.formatoptions:remove { "o", "r" } end)
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
              vim.opt_local.expandtab = true
              vim.opt_local.autoindent = true
            end,
          },
        },
      },
      options = {
        opt = {
          relativenumber = true,
          number = true,
          spell = false,
          signcolumn = "yes",
          wrap = false,
          guifont = "JetBrainsMono Nerd Font:h14",
          clipboard = "unnamedplus",
        },
        g = {
          loaded_node_provider = 0,
          loaded_perl_provider = 0,
          loaded_python3_provider = 0,
          loaded_ruby_provider = 0,
        },
      },
      mappings = {
        n = {
          ["<C-x>"] = { function() require("astrocore.buffer").nav(vim.v.count1) end, desc = "Next buffer" },
          ["<C-z>"] = { function() require("astrocore.buffer").nav(-vim.v.count1) end, desc = "Previous buffer" },

          ["<Leader>bd"] = {
            function()
              require("astroui.status.heirline").buffer_picker(
                function(bufnr) require("astrocore.buffer").close(bufnr) end
              )
            end,
            desc = "Close buffer from tabline",
          },

          ["<Leader>w"] = false,
          ["<Leader>q"] = false,
          ["<Leader>Q"] = false,
          ["<Leader>n"] = false,
          ["<Leader>/"] = false,
          ["<Leader>c"] = false,

          ["<Leader>t"] = false,
          ["<Leader>tf"] = false,
          ["<Leader>th"] = false,
          ["<Leader>tv"] = false,
          ["<Leader>tt"] = false,
          ["<Leader>tn"] = false,
          ["<Leader>tp"] = false,
          ["<Leader>tu"] = false,
          ["<Leader>tl"] = false,
          ["<Leader>T"] = { desc = "Terminal" },
          ["<Leader>Tf"] = { "<Cmd>ToggleTerm direction=float<CR>", desc = "ToggleTerm float" },
          ["<Leader>Th"] = { "<Cmd>ToggleTerm size=10 direction=horizontal<CR>", desc = "ToggleTerm horizontal" },
          ["<Leader>Tv"] = { "<Cmd>ToggleTerm size=80 direction=vertical<CR>", desc = "ToggleTerm vertical" },
          ["<Leader>Tg"] = { function() require("astrocore").toggle_term_cmd "lazygit" end, desc = "Lazygit" },

          ["<Leader>g"] = false,
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
    }

    opts = vim.tbl_deep_extend("force", opts or {}, user_opts)
    if opts.diagnostics and opts.diagnostics.jump then opts.diagnostics.jump.float = nil end
    return opts
  end,
}
