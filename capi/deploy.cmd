@ECHO OFF

python setup.py build

echo Error level %errorLevel%

if errorLevel 0 (
    echo.
    echo == Successfully compiled module
    echo.

    python setup.py install
    if errorLevel 0 (
        echo.
        echo == Successfully deployed module
        echo.
    )
) else (
    echo.
    echo == Compilation failed
    echo.
)