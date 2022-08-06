#!/bin/bash

CC="g++"

if python setup.py build then
    echo
    echo == Successfully compiled module
    echo

    if python setup.py install then
        echo
        echo == Successfully deployed module
        echo
    else
        echo
        echo == Failed to deploy module
        echo
    fi
else
    echo
    echo == Compilation failed
    echo
fi

# @ECHO OFF

# call python setup.py build

# if %ERRORLEVEL% == 0 (
#     echo.
#     echo == Successfully compiled module
#     echo.

#     call python setup.py install

#     if %ERRORLEVEL% == 0 (
#         echo.
#         echo == Successfully deployed module
#         echo.
#     ) else (
#         echo.
#         echo == Failed to deploy module
#         echo.
#     )
# ) else (
#     echo.
#     echo == Compilation failed
#     echo.
# )
