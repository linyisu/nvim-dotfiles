local function centered_pick_window()
  local height = math.max(10, math.floor(vim.o.lines * 0.7))
  local width = math.max(40, math.floor(vim.o.columns * 0.7))

  return {
    anchor = "NW",
    border = "rounded",
    height = height,
    width = width,
    row = math.floor((vim.o.lines - height) * 0.5),
    col = math.floor((vim.o.columns - width) * 0.5),
  }
end

return {
  {
    "nvim-mini/mini.files",
    opts = {
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
      options = {
        use_as_default_explorer = true,
      },
      windows = {
        max_number = 1,
        preview = false,
        width_focus = 32,
        width_preview = 40,
      },
    },
    config = function(_, opts)
      require("mini.files").setup(opts)

      vim.api.nvim_create_autocmd("User", {
        pattern = "MiniFilesExplorerOpen",
        callback = function()
          local files = require("mini.files")
          require("core.file_explorer").disable_neovide_animations()

          files.set_bookmark("~", vim.fn.expand("~"), { desc = "Home directory" })
          files.set_bookmark("c", vim.fn.stdpath("config"), { desc = "Config directory" })
          files.set_bookmark("w", vim.fn.getcwd, { desc = "Working directory" })
        end,
      })

      vim.api.nvim_create_autocmd("User", {
        pattern = "MiniFilesExplorerClose",
        callback = function()
          require("core.file_explorer").restore_neovide_animations()
        end,
      })

      vim.api.nvim_create_autocmd("User", {
        pattern = "MiniFilesBufferCreate",
        callback = function(event)
          local buf = event.data.buf_id
          local explorer = require("core.file_explorer")

          explorer.disable_buffer_animations(buf)

          local jump = function(path)
            require("mini.files").set_branch({ vim.fn.expand(path) })
          end

          vim.keymap.set("n", "j", function()
            explorer.move_cursor(buf, 1)
          end, { buffer = buf, desc = "Move down" })

          vim.keymap.set("n", "<Down>", function()
            explorer.move_cursor(buf, 1)
          end, { buffer = buf, desc = "Move down" })

          vim.keymap.set("n", "k", function()
            explorer.move_cursor(buf, -1)
          end, { buffer = buf, desc = "Move up" })

          vim.keymap.set("n", "<Up>", function()
            explorer.move_cursor(buf, -1)
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
    end,
  },

  {
    "nvim-mini/mini.pick",
    opts = {
      window = {
        config = centered_pick_window,
      },
    },
    config = function(_, opts)
      require("mini.pick").setup(opts)
    end,
  },

  {
    "nvim-mini/mini.extra",
    dependencies = { "nvim-mini/mini.pick" },
    config = function()
      require("mini.extra").setup()
    end,
  },
}
