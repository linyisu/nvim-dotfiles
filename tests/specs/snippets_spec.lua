local function with_luasnip_stubs(callback)
  local saved = {
    fmt = package.loaded["luasnip.extras.fmt"],
    extras = package.loaded["luasnip.extras"],
  }

  package.loaded["luasnip.extras.fmt"] = {
    fmt = function(pattern, nodes)
      return {
        kind = "fmt",
        pattern = pattern,
        nodes = nodes,
      }
    end,
  }
  package.loaded["luasnip.extras"] = {
    rep = function(index)
      return {
        kind = "rep",
        index = index,
      }
    end,
  }

  local ok, result = xpcall(callback, debug.traceback)

  package.loaded["luasnip.extras.fmt"] = saved.fmt
  package.loaded["luasnip.extras"] = saved.extras

  if not ok then
    error(result, 0)
  end

  return result
end

local function luasnip_stub()
  local ls = {}

  ls.snippet = function(trigger, nodes)
    return {
      trigger = trigger,
      nodes = nodes,
    }
  end
  ls.snippet_node = function(position, nodes)
    return {
      kind = "snippet_node",
      position = position,
      nodes = nodes,
    }
  end
  ls.text_node = function(text)
    return {
      kind = "text",
      text = text,
    }
  end
  ls.insert_node = function(position, default)
    return {
      kind = "insert",
      position = position,
      default = default,
    }
  end
  ls.function_node = function(fn, args)
    return {
      kind = "function",
      fn = fn,
      args = args,
    }
  end
  ls.choice_node = function(position, choices)
    return {
      kind = "choice",
      position = position,
      choices = choices,
    }
  end
  ls.dynamic_node = function(position, fn, args)
    return {
      kind = "dynamic",
      position = position,
      fn = fn,
      args = args,
    }
  end
  ls.parser = {
    parse_snippet = function(trigger, body)
      return {
        trigger = trigger,
        body = body,
        parsed = true,
      }
    end,
  }

  return ls
end

local function find_snippet(snippets, trigger)
  for _, snippet in ipairs(snippets) do
    if snippet.trigger == trigger then
      return snippet
    end
  end
end

describe("C++ LuaSnip snippets", function()
  it("defines the competitive programming snippets for cpp buffers", function()
    with_luasnip_stubs(function()
      local snippets = reload("core.cpp_snippets").snippets(luasnip_stub())

      for _, trigger in ipairs({
        "normal",
        "normals",
        "cin",
        "cout",
        "vec",
        "vi",
        "yn",
        "all",
        "sum",
        "opxy",
        "qpow",
        "qpowM",
      }) do
        expect.truthy(find_snippet(snippets, trigger), "expected snippet trigger " .. trigger)
      end

      expect.truthy(#snippets >= 45, "expected the full C++ snippet set")
      expect.contains(find_snippet(snippets, "normal").nodes[1].text[1], "#include <bits/stdc++.h>")
      expect.contains(find_snippet(snippets, "qpow").body, "long long qpow")
    end)
  end)
end)
