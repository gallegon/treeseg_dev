@echo off

setlocal

pushd "%~dp0"\..\capi

echo == Compiling standalone
echo.

call python compile_standalone.py

if %ERRORLEVEL% == 0 (
    echo.
    echo == Successfully compiled standalone
    echo.
) else (
    echo.
    echo == Failed to compile standalone
    echo.
)

popd

endlocal
