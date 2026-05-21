if vim.treesitter and not vim.g.linyisu_treesitter_range_compat then
  vim.g.linyisu_treesitter_range_compat = true
  local get_range = vim.treesitter.get_range

  local function has_method(value, key)
    local ok, method = pcall(function() return value[key] end)
    return ok and type(method) == "function"
  end

  local function range_from_nodes(first, last)
    if not (has_method(first, "start") and has_method(last, "end_")) then return end
    local srow, scol, sbytes = first:start()
    local erow, ecol, ebytes = last:end_()
    return { srow, scol, sbytes, erow, ecol, ebytes }
  end

  vim.treesitter.get_range = function(node, source, metadata)
    local ok, range = pcall(get_range, node, source, metadata)
    if ok then return range end

    if type(node) == "userdata" then
      local fallback = range_from_nodes(node, node)
      if fallback then return fallback end
    end

    if type(node) == "table" and node[1] and node[#node] then
      local fallback = range_from_nodes(node[1], node[#node])
      if fallback then return fallback end
    end

    error(range)
  end
end

if vim.lsp and vim.lsp.codelens and vim.lsp.codelens.enable then
  vim.lsp.codelens.refresh = function(opts)
    vim.validate("opts", opts, "table", true)
    opts = opts or {}
    vim.lsp.codelens.enable(true, { bufnr = opts.bufnr })
  end

  vim.lsp.codelens.clear = function(client_id, bufnr)
    vim.lsp.codelens.enable(false, { bufnr = bufnr, client_id = client_id })
  end
end

if not vim.g.linyisu_validate_compat then
  vim.g.linyisu_validate_compat = true
  local validate = vim.validate

  vim.validate = function(name, value, validator, optional, message)
    if type(name) == "table" and value == nil and validator == nil and optional == nil and message == nil then
      for key, spec in pairs(name) do
        validate(key, spec[1], spec[2], spec[3], spec[4])
      end
      return
    end

    return validate(name, value, validator, optional, message)
  end
end

if vim._str_utfindex and not vim.g.linyisu_str_utfindex_compat then
  vim.g.linyisu_str_utfindex_compat = true
  local str_utfindex = vim.str_utfindex

  vim.str_utfindex = function(s, encoding, index, strict_indexing)
    if encoding == nil or type(encoding) == "number" then
      local col32, col16 = vim._str_utfindex(s, encoding)
      if not col32 or not col16 then error "index out of range" end
      return col32, col16
    end

    return str_utfindex(s, encoding, index, strict_indexing)
  end
end
