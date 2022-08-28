@echo off

call conda activate treeseg_dev

call ..\capi\build\standalone.exe ..\python\sample_data\hard_nno.las
