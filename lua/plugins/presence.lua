return {
  "andweeb/presence.nvim",
  cond = function() return #vim.api.nvim_list_uis() > 0 end,
  event = "VeryLazy",
}
