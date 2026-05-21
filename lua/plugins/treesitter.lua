return {
  {
    "nvim-treesitter/nvim-treesitter",
    branch = "master",
    commit = false,
    dependencies = {
      {
        "nvim-treesitter/nvim-treesitter-textobjects",
        branch = "master",
        commit = false,
        lazy = false,
      },
    },
    opts = function(_, opts)
      opts.ensure_installed =
        require("astrocore").list_insert_unique(opts.ensure_installed, { "html", "jsonc", "regex" })
      opts.indent = opts.indent or {}
      opts.indent.disable =
        require("astrocore").list_insert_unique(opts.indent.disable, { "c", "cpp", "cuda", "objc" })
      opts.auto_install = true
      return opts
    end,
  },
}
