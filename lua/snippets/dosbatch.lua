local ls = require("luasnip")

local s = ls.snippet
local t = ls.text_node
local i = ls.insert_node
local c = ls.choice_node
local rep = require("luasnip.extras").rep
local fmt = require("luasnip.extras.fmt").fmt

return {
    s("dp", fmt([[
    @echo off
    echo Compiling {}.cpp...
    g++ -std=c++2c {}.cpp -o {}.exe
    if %errorlevel% neq 0 (
        echo {}.cpp compilation failed.
        pause
        goto :eof
    )

    echo.
    echo Compiling {}.cpp...
    g++ -std=c++2c {}.cpp -o {}.exe
    if %errorlevel% neq 0 (
        echo {}.cpp compilation failed.
        pause
        goto :eof
    )

    echo.
    echo Both files compiled successfully. Starting tests...
    echo.

    :loop
    python {}.py > {}.in
    {} < {}.in > {}.out
    {} < {}.in > {}.out
    fc {}.out {}.out > nul
    if %errorlevel% == 1 (
        echo.
        echo =================================
        echo      Difference Found!
        echo =================================
        echo.
        echo ----- Input: {}.in -----
        type {}.in
        echo.
        echo ----- Output from {}: {}.out -----
        type {}.out
        echo.
        echo ----- Output from {}: {}.out -----
        type {}.out
        echo.
        pause
    ) else (
        echo No difference found.
    )
    goto loop]],
        { i(1, "brute"), rep(1), rep(1), rep(1), i(2, "test"), rep(2), rep(2), rep(2), i(3, "gen"), i(4, "in"), rep(1),
            rep(4), i(5, "out1"), rep(2), rep(4), i(6, "out2"), rep(5), rep(6), rep(4), rep(4), rep(1), rep(5), rep(5),
            rep(2), rep(6), rep(6) })),

    s("dpcpp", fmt([[
    @echo off
    echo Compiling {}.cpp...
    g++ -std=c++2c {}.cpp -o {}.exe
    if %errorlevel% neq 0 (
        echo {}.cpp compilation failed.
        pause
        goto :eof
    )

    echo.
    echo Compiling {}.cpp...
    g++ -std=c++2c {}.cpp -o {}.exe
    if %errorlevel% neq 0 (
        echo {}.cpp compilation failed.
        pause
        goto :eof
    )

    echo.
    echo Compiling {}.cpp...
    g++ -std=c++2c {}.cpp -o {}.exe
    if %errorlevel% neq 0 (
        echo {}.cpp compilation failed.
        pause
        goto :eof
    )

    echo.
    echo All files compiled successfully. Starting tests...
    echo.

    :loop
    {} > {}.in
    {} < {}.in > {}.out
    {} < {}.in > {}.out
    fc {}.out {}.out > nul
    if %errorlevel% == 1 (
        echo.
        echo =================================
        echo      Difference Found!
        echo =================================
        echo.
        echo ----- Input: {}.in -----
        type {}.in
        echo.
        echo ----- Output from {}: {}.out -----
        type {}.out
        echo.
        echo ----- Output from {}: {}.out -----
        type {}.out
        echo.
        pause
    ) else (
        echo No difference found.
    )
    goto loop]],
        { i(1, "brute"), rep(1), rep(1), rep(1), i(2, "test"), rep(2), rep(2), rep(2), i(3, "gen"), rep(3), rep(3), rep(3),
            rep(3), i(4, "in"), rep(1), rep(4), i(5, "out1"), rep(2), rep(4), i(6, "out2"), rep(5), rep(6), rep(4), rep(4),
            rep(1), rep(5), rep(5), rep(2), rep(6), rep(6) })),
}
