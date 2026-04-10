# XFeat 3D — iPad向けビジュアル3D復元アプリ

階層的特徴点トラッキング（ORB → XFeat → ALIKED）とOpenCVによるEssential Matrix推定を用いたモノキュラー3D点群復元アプリ。MeshLab・CloudCompare・Blenderで開けるカラー付き `.ply` ファイルを出力します。

## 機能

- **階層的トラッキング** — ORB（高速）→ XFeat（64次元）→ ALIKED（128次元）、xfeat-ipad-demoと同パイプライン
- **階層別色分けオーバーレイ** — 赤 = ORB、青 = XFeat、緑 = ALIKED
- **キーフレームベース3D復元** — Essential Matrix（RANSAC）+ Nフレームごとの三角測量
- **IMU姿勢** — CoreMotionによる回転推定（スケール曖昧性の解消補助）
- **SceneKitプレビュー** — デバイス上でリアルタイム回転する3D点群表示
- **PLY出力** — iOSの共有シートを介してカラー付きASCII PLYを書き出し

## 動作環境

- iPad（iOS 17.0以上）
- Xcode 15以上
- [XcodeGen](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`
- `xfeat-ipad-demo` からコピーしたCoreMLモデル（既設置済み）

## ビルド手順

```bash
xcodegen generate
open XFeat3D.xcodeproj
```

## プロジェクト構成

<!-- AUTO-GENERATED from Sources/ -->

```
Sources/
├── App/
│   └── XFeat3DApp.swift             アプリエントリポイント (@main)
├── Camera/
│   ├── CameraManager.swift          AVFoundationキャプチャ + 向き制御
│   └── CameraPreviewView.swift      プレビューレイヤー SwiftUIラッパー
├── Tracking/                        (xfeat-ipad-demoと共有)
│   ├── PointTracker.swift           階層的ORB→XFeat→ALIKEDトラッカー
│   ├── TrackedPoint.swift           id、位置、ディスクリプタ、階層
│   ├── TrackTier.swift              orb < xfeat < aliked
│   └── TrackingExtractor.swift      抽出器ストラテジー
├── Reconstruction/
│   ├── ReconstructionEngine.swift   キーフレーム選択、Essential Matrix、三角測量
│   ├── MotionManager.swift          CoreMotion姿勢ラッパー
│   ├── PointCloud.swift             マージ半径による重複除去付き3D点群蓄積
│   └── PLYExporter.swift            RGB色 + 信頼度付きASCII PLY出力
├── Views/
│   ├── ContentView.swift            カメラオーバーレイ + コントロール + 設定シート
│   └── PointCloudView.swift         SCNGeometryポイントプリミティブレンダラー
├── XFeat/
│   ├── XFeatMatcher.swift
│   └── XFeat.mlpackage
├── ALIKED/
│   ├── ALIKEDMatcher.swift
│   └── ALIKED.mlpackage
├── SuperPoint/
│   ├── SuperPointMatcher.swift
│   └── SuperPoint.mlpackage
└── OpenCV/
    ├── OpenCVBridge.h/mm            LKフロー、ORB検出、recoverPose、triangulatePoints
    ├── XFeat3D-Bridging-Header.h
    └── opencv2.xcframework
```

<!-- END AUTO-GENERATED -->

## 3D復元パイプライン

<!-- AUTO-GENERATED from ReconstructionEngine.swift -->

```
フレーム N
  │
  ├─► PointTracker.update(frame)
  │       LKオプティカルフロー → 全階層
  │       定期的な再検出（ORB/XFeat/ALIKED）
  │
  └─► ReconstructionEngine.processFrame(alikedPoints, frame)
          │
          ├─ keyframeInterval フレームごと:
          │     十分な視差を持つALIKED階層点をフィルタリング
          │     pts1（前キーフレーム）/ pts2（現在）を構築
          │     OpenCV.recoverPose → R, t（スケール不定）
          │     OpenCV.triangulatePoints → ローカル3D
          │     グローバル座標系へ変換（累積 R, t）
          │     フィルタリング: depth ∈ (0.01, 50)
          │     PointCloud.add → マージ半径による重複除去
          │
          └─ PointCloud.points → SceneKit + PLY出力
```

<!-- END AUTO-GENERATED -->

## 主要パラメータ

<!-- AUTO-GENERATED from ReconstructionEngine.swift defaults -->

| パラメータ | デフォルト | 説明 |
|-----------|---------|-------------|
| `keyframeInterval` | 15フレーム | 3D復元がトリガーされる間隔 |
| `minParallax` | 8px | トリガーに必要な平均点移動量の最小値 |
| `minAlikedPoints` | 20 | 必要なALIKED階層点の最小数 |
| `mergeRadius` | 0.02単位 | 点群内の重複抑制半径 |
| `maxPoints` | 50,000 | 点群サイズの上限 |
| `fx / fy` | 自動 | 焦点距離（`videoFieldOfView` から推定） |
| `cx / cy` | 480 / 360 | 主点（処理解像度 960×720） |

<!-- END AUTO-GENERATED -->

## 使用フレームワーク

<!-- AUTO-GENERATED from project.yml -->

| フレームワーク | 用途 |
|-----------|---------|
| `Accelerate` | ディスクリプタのコサイン類似度計算（BLAS GEMM） |
| `CoreMotion` | IMU姿勢（ジャイロ + 加速度センサー融合） |
| `SceneKit` | デバイス上3D点群プレビュー |
| `opencv2.xcframework` | LKフロー、ORB検出、Essential Matrix、三角測量 |

<!-- END AUTO-GENERATED -->

## PLY出力フォーマット

```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float confidence
end_header
x0 y0 z0 r g b conf
...
```

開き方: **MeshLab**、**CloudCompare**、**Blender**（ファイル → インポート → Stanford PLY）、またはその他PLYビューア。

## 制限事項

- **スケール不定の復元** — ステレオや既知ベースラインなしではメートル単位のスケールは得られません。構造の形状は正しいですが、絶対距離は不正確です。
- **ループ閉鎖なし** — 長いシーケンスで誤差が蓄積します。
- **CPU専用CoreML** — XFeat/ALIKEDの出力テンソルのGPUストライドパディング問題を回避するためCPUのみ使用。
- **iPad専用** — カメラ権限と向き処理はiPad向けに調整済み。
