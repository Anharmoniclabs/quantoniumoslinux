@echo off
REM QUE Guild — The Bronx — one-click launcher.
REM Double-click or run from terminal. No Godot editor needed.

setlocal

set "REPO=%~dp0"
set "PROJECT=%REPO%godot\bronx-rpg"
set "GODOT="

REM 1) Check PATH
where godot >nul 2>&1 && set "GODOT=godot" && goto :run

REM 2) Check repo root
if exist "%REPO%Godot.exe" set "GODOT=%REPO%Godot.exe" && goto :run

REM 3) Check Downloads (nested folder from zip extraction)
set "DL=%USERPROFILE%\Downloads"
if exist "%DL%\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64.exe" (
    set "GODOT=%DL%\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64.exe"
    goto :run
)

REM 4) Wildcard search in Downloads
for /f "delims=" %%D in ('dir /b /ad "%DL%\Godot_v*" 2^>nul') do (
    if exist "%DL%\%%D\%%D" (
        set "GODOT=%DL%\%%D\%%D"
        goto :run
    )
)

echo.
echo  ERROR: Godot not found.
echo  Copy Godot_v4.x.exe into %REPO% and rename to Godot.exe
echo  Or add Godot to your PATH.
echo.
pause
goto :eof

:run
echo Starting QUE Guild...
echo Godot: %GODOT%
echo Project: %PROJECT%
"%GODOT%" --path "%PROJECT%"

:eof
endlocal
