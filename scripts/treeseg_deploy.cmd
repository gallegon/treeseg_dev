@echo off

setlocal

pushd "%~dp0"\..\capi

call conda activate treeseg_dev

call python setup.py build

if %ERRORLEVEL% == 0 (
    echo.
    echo == Successfully compiled module
    echo.

    call python setup.py bdist

    if %ERRORLEVEL% == 0 (
        echo.
        echo == Successfully built module distributable
        echo.

        call python setup.py install
         if %ERRORLEVEL% == 0 (
            echo.
            echo == Successuly deployed module
            echo.
         ) else (
            echo.
            echo == Failed to install module
            echo.
         )

    ) else (
        echo.
        echo == Failed to build module
        echo.
    )
) else (
    echo.
    echo == Failed to compile module
    echo.
)

popd

endlocal