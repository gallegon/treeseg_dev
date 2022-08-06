@ECHO OFF

call python setup.py build

if %ERRORLEVEL% == 0 (
    echo.
    echo == Successfully compiled module
    echo.

    call python setup.py install

    if %ERRORLEVEL% == 0 (
        echo.
        echo == Successfully deployed module
        echo.
    ) else (
        echo.
        echo == Failed to deploy module
        echo.
    )
) else (
    echo.
    echo == Compilation failed
    echo.
)
