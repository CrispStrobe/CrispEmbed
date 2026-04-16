@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "C:\Users\stc\Downloads\code\CrispEmbed\build-cuda"
ninja 2>&1
echo NINJA_EXIT_CODE=%ERRORLEVEL%
