return {
  {
    "nvim-mini/mini.nvim",
    version = "*",
    event = "VeryLazy",
    config = function()
      require("mini.icons").setup()
      require("mini.files").setup({
        mappings = {
          go_in = "",
          go_in_plus = "",
          go_out = "",
          go_out_plus = "",
          reset = "",
          close = "q",
          mark_goto = "'",
          mark_set = "m",
          reveal_cwd = "@",
          show_help = "g?",
          synchronize = "=",
          trim_left = "<",
          trim_right = ">",
        },
      })
      vim.api.nvim_create_autocmd("User", {
        pattern = "MiniFilesExplorerOpen",
        callback = function()
          local files = require("mini.files")
          files.set_bookmark("~", vim.fn.expand("~"), { desc = "Home directory" })
          files.set_bookmark("c", vim.fn.stdpath("config"), { desc = "Config directory" })
          files.set_bookmark("w", vim.fn.getcwd, { desc = "Working directory" })
        end,
      })
      vim.api.nvim_create_autocmd("User", {
        pattern = "MiniFilesBufferCreate",
        callback = function(event)
          local buf = event.data.buf_id
          local explorer = require("core.file_explorer")
          local jump = function(path)
            require("mini.files").set_branch({ vim.fn.expand(path) })
          end
          local move = function(step)
            local line_count = vim.api.nvim_buf_line_count(buf)

            if line_count == 0 then
              return
            end

            local current = vim.api.nvim_win_get_cursor(0)[1]
            local count = vim.v.count1
            local target = ((current - 1 + step * count) % line_count) + 1

            vim.api.nvim_win_set_cursor(0, { target, 0 })
          end

          vim.keymap.set("n", "j", function()
            move(1)
          end, { buffer = buf, desc = "Move down" })

          vim.keymap.set("n", "<Down>", function()
            move(1)
          end, { buffer = buf, desc = "Move down" })

          vim.keymap.set("n", "k", function()
            move(-1)
          end, { buffer = buf, desc = "Move up" })

          vim.keymap.set("n", "<Up>", function()
            move(-1)
          end, { buffer = buf, desc = "Move up" })

          vim.keymap.set("n", "<CR>", function()
            require("mini.files").go_in({ close_on_file = true })
          end, { buffer = buf, desc = "Open file or directory" })

          vim.keymap.set("n", "<Right>", function()
            require("mini.files").go_in({ close_on_file = true })
          end, { buffer = buf, desc = "Open file or directory" })

          vim.keymap.set("n", "<BS>", function()
            require("mini.files").go_out()
          end, { buffer = buf, desc = "Go to parent directory" })

          vim.keymap.set("n", "-", function()
            require("mini.files").go_out()
          end, { buffer = buf, desc = "Go to parent directory" })

          vim.keymap.set("n", "<Left>", function()
            require("mini.files").go_out()
          end, { buffer = buf, desc = "Go to parent directory" })

          vim.keymap.set("n", "o", explorer.open_selected_workspace, { buffer = buf, desc = "Open as workspace" })
          vim.keymap.set("n", "a", explorer.create_file, { buffer = buf, desc = "New file" })
          vim.keymap.set("n", "A", explorer.create_directory, { buffer = buf, desc = "New directory" })
          vim.keymap.set("n", "r", explorer.rename, { buffer = buf, desc = "Rename" })
          vim.keymap.set("n", "d", explorer.remove, { buffer = buf, desc = "Delete" })
          vim.keymap.set("n", "R", explorer.refresh, { buffer = buf, desc = "Refresh" })

          for _, key in ipairs({ "i", "I", "O", "c", "C", "s", "S", "x", "X" }) do
            vim.keymap.set("n", key, "<Nop>", { buffer = buf })
          end

          vim.keymap.set("n", "gh", function()
            jump("~")
          end, { buffer = buf, desc = "Go to home directory" })

          vim.keymap.set("n", "gc", function()
            jump(vim.fn.stdpath("config"))
          end, { buffer = buf, desc = "Go to config directory" })

          vim.keymap.set("n", "gw", function()
            jump(vim.fn.getcwd())
          end, { buffer = buf, desc = "Go to working directory" })
        end,
      })
      local pick_window = function()
        local height = math.floor(vim.o.lines * 0.7)
        local width = math.floor(vim.o.columns * 0.7)

        return {
          anchor = "NW",
          border = "rounded",
          height = height,
          width = width,
          row = math.floor((vim.o.lines - height) * 0.5),
          col = math.floor((vim.o.columns - width) * 0.5),
        }
      end

      require("mini.pick").setup({
        window = {
          config = pick_window,
        },
      })
      require("mini.extra").setup()
      require("mini.comment").setup()
      require("mini.ai").setup()
      require("mini.surround").setup()
      require("mini.pairs").setup()
      require("mini.statusline").setup({ use_icons = true })

      local clue = require("mini.clue")
      clue.setup({
        triggers = {
          { mode = "n", keys = "<Leader>" },
          { mode = "x", keys = "<Leader>" },
          { mode = "n", keys = "<C-w>" },
          { mode = "n", keys = "[" },
          { mode = "n", keys = "]" },
          { mode = "n", keys = "z" },
          { mode = "x", keys = "z" },
        },
        clues = {
          { mode = "n", keys = "<Leader>b", desc = "+buffer" },
          { mode = "n", keys = "<Leader>f", desc = "+find" },
          { mode = "n", keys = "<Leader>g", desc = "+git" },
          { mode = "n", keys = "<Leader>gh", desc = "+hunk" },
          { mode = "n", keys = "<Leader>H", desc = "home page" },
          { mode = "n", keys = "<Leader>l", desc = "+lsp" },
          { mode = "n", keys = "<Leader>o", desc = "open workspace" },
          { mode = "n", keys = "<Leader>r", desc = "+run" },
          { mode = "n", keys = "<Leader>w", desc = "+window" },
          clue.gen_clues.square_brackets(),
          clue.gen_clues.windows(),
          clue.gen_clues.z(),
        },
        window = {
          delay = 250,
          config = {
            width = "auto",
          },
        },
      })
    end,
  },
  {
    "lewis6991/gitsigns.nvim",
    event = { "BufReadPre", "BufNewFile" },
    keys = {
      { "]h", function() require("gitsigns").nav_hunk("next") end, desc = "Next git hunk" },
      { "[h", function() require("gitsigns").nav_hunk("prev") end, desc = "Previous git hunk" },
      { "<leader>ghs", function() require("gitsigns").stage_hunk() end, desc = "Stage hunk" },
      { "<leader>ghr", function() require("gitsigns").reset_hunk() end, desc = "Reset hunk" },
      {
        "<leader>ghs",
        function()
          require("gitsigns").stage_hunk({ vim.fn.line("."), vim.fn.line("v") })
        end,
        mode = "v",
        desc = "Stage selected hunk",
      },
      {
        "<leader>ghr",
        function()
          require("gitsigns").reset_hunk({ vim.fn.line("."), vim.fn.line("v") })
        end,
        mode = "v",
        desc = "Reset selected hunk",
      },
      { "<leader>ghu", function() require("gitsigns").undo_stage_hunk() end, desc = "Undo stage hunk" },
      { "<leader>ghp", function() require("gitsigns").preview_hunk() end, desc = "Preview hunk" },
      { "<leader>ghb", function() require("gitsigns").blame_line({ full = true }) end, desc = "Blame line" },
      { "<leader>ghd", function() require("gitsigns").diffthis() end, desc = "Diff this" },
    },
    opts = {
      signs = {
        add = { text = "+" },
        change = { text = "~" },
        delete = { text = "_" },
        topdelete = { text = "^" },
        changedelete = { text = "~" },
        untracked = { text = "?" },
      },
    },
  },
}
