#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$root_dir"
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
