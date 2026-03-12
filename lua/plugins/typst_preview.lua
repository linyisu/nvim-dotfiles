---@type LazySpec
return {
  {
    "chomosuke/typst-preview.nvim",
    opts = function(_, opts)
      local data = vim.fn.stdpath "data"
      local mason_tinymist = data .. "/mason/packages/tinymist/tinymist-win32-x64.exe"
      local bundled_websocat = data .. "/typst-preview/websocat.x86_64-pc-windows-gnu.exe"

      opts.dependencies_bin = opts.dependencies_bin or {}
      opts.dependencies_bin.tinymist = vim.uv.fs_stat(mason_tinymist) and mason_tinymist or opts.dependencies_bin.tinymist
      opts.dependencies_bin.websocat = vim.uv.fs_stat(bundled_websocat) and bundled_websocat or opts.dependencies_bin.websocat
    end,
  },
}