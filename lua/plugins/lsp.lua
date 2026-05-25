local c_family_filetypes = {
  c = true,
  cpp = true,
  cuda = true,
  objc = true,
  objcpp = true,
}

local function lsp_keymaps(event)
  local map = vim.keymap.set
  local opts = function(desc)
    return { buffer = event.buf, desc = desc }
  end

  map("n", "gd", vim.lsp.buf.definition, opts("Go to definition"))
  map("n", "gD", vim.lsp.buf.declaration, opts("Go to declaration"))
  map("n", "gi", vim.lsp.buf.implementation, opts("Go to implementation"))
  map("n", "gr", vim.lsp.buf.references, opts("References"))
  map("n", "K", vim.lsp.buf.hover, opts("Hover documentation"))
  map("n", "<leader>lr", vim.lsp.buf.rename, opts("Rename symbol"))
  map({ "n", "x" }, "<leader>la", vim.lsp.buf.code_action, opts("Code action"))
  map("n", "<leader>ld", vim.diagnostic.open_float, opts("Line diagnostic"))
  map("n", "<leader>lf", function()
    vim.lsp.buf.format({ async = true })
  end, opts("Format buffer"))
end

local format_group = vim.api.nvim_create_augroup("UserLspFormat", { clear = true })

local function enable_format_on_save(event)
  local filetype = vim.bo[event.buf].filetype

  if not c_family_filetypes[filetype] then
    return
  end

  local client = vim.lsp.get_client_by_id(event.data.client_id)

  if not client or client.name ~= "clangd" or not client:supports_method("textDocument/formatting", event.buf) then
    return
  end

  require("core.clang_format").ensure_for_buffer(event.buf)

  vim.api.nvim_clear_autocmds({ group = format_group, buffer = event.buf })
  vim.api.nvim_create_autocmd("BufWritePre", {
    group = format_group,
    buffer = event.buf,
    callback = function()
      vim.lsp.buf.format({
        bufnr = event.buf,
        timeout_ms = 2000,
        filter = function(format_client)
          return format_client.name == "clangd"
        end,
      })
    end,
  })
end

local function configure_lsp()
  local capabilities = vim.lsp.protocol.make_client_capabilities()

  if capabilities.workspace then
    capabilities.workspace.didChangeWatchedFiles = nil
  end

  local ok_cmp, cmp_lsp = pcall(require, "cmp_nvim_lsp")
  if ok_cmp then
    capabilities = cmp_lsp.default_capabilities(capabilities)
  end

  vim.lsp.config("*", {
    capabilities = capabilities,
  })

  vim.lsp.config("lua_ls", {
    settings = {
      Lua = {
        completion = {
          callSnippet = "Replace",
        },
        diagnostics = {
          globals = { "vim" },
        },
        workspace = {
          checkThirdParty = false,
        },
      },
    },
  })

  vim.lsp.config("clangd", {
    cmd = {
      "clangd",
      "--background-index",
      "--clang-tidy",
      "--completion-style=detailed",
      "--fallback-style=Google",
      "--header-insertion=never",
    },
  })

  vim.diagnostic.config({
    underline = true,
    virtual_text = {
      prefix = ">",
      spacing = 2,
    },
    severity_sort = true,
    float = {
      border = "rounded",
      source = true,
    },
  })

  vim.api.nvim_create_autocmd("LspAttach", {
    group = vim.api.nvim_create_augroup("UserLspKeymaps", { clear = true }),
    callback = function(event)
      lsp_keymaps(event)
      enable_format_on_save(event)
    end,
  })
end

return {
  {
    "mason-org/mason.nvim",
    cmd = "Mason",
    opts = {
      ui = {
        border = "rounded",
      },
    },
  },
  {
    "neovim/nvim-lspconfig",
    event = { "BufReadPre", "BufNewFile" },
    dependencies = {
      "hrsh7th/cmp-nvim-lsp",
    },
    config = configure_lsp,
  },
  {
    "mason-org/mason-lspconfig.nvim",
    event = { "BufReadPre", "BufNewFile" },
    cmd = {
      "LspInstall",
      "LspUninstall",
    },
    dependencies = {
      "mason-org/mason.nvim",
      "neovim/nvim-lspconfig",
    },
    opts = {
      ensure_installed = {
        "lua_ls",
        "clangd",
      },
      automatic_enable = {
        "lua_ls",
        "clangd",
      },
    },
  },
  {
    "hrsh7th/nvim-cmp",
    event = "InsertEnter",
    dependencies = {
      "hrsh7th/cmp-buffer",
      "hrsh7th/cmp-nvim-lsp",
      "hrsh7th/cmp-path",
    },
    config = function()
      local cmp = require("cmp")

      cmp.setup({
        snippet = {
          expand = function(args)
            vim.snippet.expand(args.body)
          end,
        },
        completion = {
          completeopt = "menu,menuone,noinsert",
        },
        window = {
          completion = cmp.config.window.bordered(),
          documentation = cmp.config.window.bordered(),
        },
        mapping = {
          ["<C-Space>"] = cmp.mapping.complete(),
          ["<C-e>"] = cmp.mapping.abort(),
          ["<C-n>"] = cmp.mapping.select_next_item({ behavior = cmp.SelectBehavior.Insert }),
          ["<C-p>"] = cmp.mapping.select_prev_item({ behavior = cmp.SelectBehavior.Insert }),
          ["<C-f>"] = cmp.mapping.scroll_docs(4),
          ["<C-b>"] = cmp.mapping.scroll_docs(-4),
          ["<CR>"] = cmp.mapping.confirm({ select = true }),
        },
        sources = cmp.config.sources({
          { name = "nvim_lsp" },
          { name = "path" },
        }, {
          { name = "buffer" },
        }),
      })
    end,
  },
}
