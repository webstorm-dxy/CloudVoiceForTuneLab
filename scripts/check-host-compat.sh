#!/usr/bin/env bash
# Verify the plugin only uses APIs that exist in the *released host SDK* installed on this machine.
#
# Background: TuneLab's SDK evolves faster than the released app. Compiling against the latest
# TuneLab source can silently add references to new SDK members (e.g. ButtonConfig), and the
# released app then refuses to load the extension with:
#   "built against an incompatible SDK (type ... no longer exists)"
# This script compiles the same plugin sources against the SDK dlls shipped inside the installed
# TuneLab app, so any such API use fails here instead of at runtime.
#
#   ./scripts/check-host-compat.sh
#   ./scripts/check-host-compat.sh "/Applications/TuneLab.app/Contents/MacOS"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HOST_DIR=""
if [ "$#" -ge 1 ]; then
    HOST_DIR="$1"
else
    for candidate in \
        "/Applications/TuneLab.app/Contents/MacOS" \
        "$HOME/Applications/TuneLab.app/Contents/MacOS" \
        "$HOME/Documents/DSH/TuneLab/publish/TuneLab.app/Contents/MacOS"; do
        if [ -f "$candidate/TuneLab.SDK.dll" ]; then HOST_DIR="$candidate"; break; fi
    done
fi
if [ -z "$HOST_DIR" ] || [ ! -f "$HOST_DIR/TuneLab.SDK.dll" ] || [ ! -f "$HOST_DIR/TuneLab.Foundation.dll" ]; then
    echo "error: host TuneLab SDK not found." >&2
    echo "usage: $0 /path/to/TuneLab.app/Contents/MacOS" >&2
    exit 1
fi

echo "==> checking plugin API compatibility against: $HOST_DIR"
WORK="$ROOT/.sandbox/hostcompat"
mkdir -p "$WORK"
cat > "$WORK/hostcompat.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <AssemblyName>TuneLab.CloudVoice</AssemblyName>
    <RootNamespace>TuneLab.CloudVoice</RootNamespace>
    <LangVersion>latest</LangVersion>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="$ROOT/src/TuneLab.CloudVoice/**/*.cs" Exclude="$ROOT/src/TuneLab.CloudVoice/obj/**;$ROOT/src/TuneLab.CloudVoice/bin/**" />
    <Reference Include="TuneLab.SDK"><HintPath>$HOST_DIR/TuneLab.SDK.dll</HintPath><Private>false</Private></Reference>
    <Reference Include="TuneLab.Foundation"><HintPath>$HOST_DIR/TuneLab.Foundation.dll</HintPath><Private>false</Private></Reference>
  </ItemGroup>
</Project>
EOF

export NUGET_PACKAGES="${NUGET_PACKAGES:-$ROOT/.nuget/packages}"
export DOTNET_CLI_HOME="${DOTNET_CLI_HOME:-$ROOT/.dotnet}"
export DOTNET_NOLOGO=1
if dotnet build "$WORK/hostcompat.csproj" -c Debug > "$WORK/build.log" 2>&1; then
    echo "OK: plugin compiles against the host SDK (no incompatible API usage)"
else
    echo "FAILED: plugin uses APIs missing from the installed host SDK:" >&2
    grep -E "error CS|error RS" "$WORK/build.log" | head -20 >&2 || tail -20 "$WORK/build.log" >&2
    exit 1
fi
