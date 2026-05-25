if not vim.g.neovide then
  return
end

local function default_font()
  if vim.env.NEOVIDE_FONT and vim.env.NEOVIDE_FONT ~= "" then
    return vim.env.NEOVIDE_FONT
  end

  if vim.fn.has("win32") == 1 then
    return "JetBrainsMono NFM:h12"
  end

  if vim.fn.has("mac") == 1 then
    return "Menlo:h12"
  end

  return "monospace:h12"
end

vim.o.guifont = default_font()
vim.o.linespace = 0
vim.o.background = "dark"

vim.g.neovide_theme = "dark"
vim.g.neovide_scale_factor = 1.0

vim.g.neovide_padding_top = 4
vim.g.neovide_padding_bottom = 4
vim.g.neovide_padding_left = 6
vim.g.neovide_padding_right = 6

vim.g.neovide_remember_window_size = true
vim.g.neovide_confirm_quit = true
vim.g.neovide_hide_mouse_when_typing = true

vim.g.neovide_scroll_animation_length = 0.15
vim.g.neovide_position_animation_length = 0.08
vim.g.neovide_cursor_animation_length = 0.08
vim.g.neovide_cursor_short_animation_length = 0.03
vim.g.neovide_cursor_trail_size = 0.5
