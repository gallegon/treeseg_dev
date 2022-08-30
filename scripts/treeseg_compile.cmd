@echo off

setlocal

pushd "%~dp0"\..\capi

call conda activate treeseg_dev

call python setup.py build

if %ERRORLEVEL% == 0 (
    echo.
    echo == Successfully compiled module
    echo.
) else (
    echo.
    echo == Failed to compile module
    echo.
)

popd

endlocal