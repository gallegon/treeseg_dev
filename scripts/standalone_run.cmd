@echo off

call conda activate treeseg_dev

call ..\capi\build\standalone\standalone.exe ..\sample_data\hard_nno.las
