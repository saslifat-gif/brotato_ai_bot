@echo off
cd /d %~dp0
C:\ml\brotato-native-tools\ziglang\zig.exe c++ -shared -O2 separation.cpp -o separation.dll
