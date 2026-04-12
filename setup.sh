#!/bin/bash
# setup.sh — VisualOdo 開発環境セットアップ
# 別マシンでクローン後に実行することで、すぐにビルドできる状態にする

set -e

echo "==> VisualOdo セットアップ開始"

# ── 1. Xcode チェック ──────────────────────────────────────────────
if ! xcode-select -p &>/dev/null; then
  echo "ERROR: Xcode が見つかりません。App Store から Xcode をインストールしてください。"
  exit 1
fi
echo "  [OK] Xcode: $(xcode-select -p)"

# ── 2. XcodeGen チェック・インストール ─────────────────────────────
if ! command -v xcodegen &>/dev/null; then
  echo "==> XcodeGen が未インストールのため Homebrew でインストールします..."
  if ! command -v brew &>/dev/null; then
    echo "ERROR: Homebrew が見つかりません。https://brew.sh からインストールしてください。"
    exit 1
  fi
  brew install xcodegen
fi
echo "  [OK] XcodeGen: $(xcodegen --version)"

# ── 3. .xcodeproj 生成 ─────────────────────────────────────────────
echo "==> Xcode プロジェクトを生成中..."
xcodegen generate
echo "  [OK] VisualOdo.xcodeproj を生成しました"

# ── 4. 完了メッセージ ──────────────────────────────────────────────
echo ""
echo "セットアップ完了！"
echo ""
echo "次の手順:"
echo "  1. open VisualOdo.xcodeproj"
echo "  2. Xcode の Signing & Capabilities でチーム（Apple ID）を設定"
echo "  3. iPad を接続して Run (Cmd+R)"
echo ""
echo "依存ライブラリ（リポジトリに同梱）:"
echo "  - opencv2.xcframework  (Sources/OpenCV/)"
echo "  - Ceres.xcframework    (Sources/Ceres/)"
echo "  - XFeat.mlpackage      (Sources/XFeat/)"
echo "  - ALIKED.mlpackage     (Sources/ALIKED/)"
echo "  - SuperPoint.mlpackage (Sources/SuperPoint/)"
