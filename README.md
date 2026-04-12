# VisualOdo — ArUco初期化 VO + 多キーフレームロスト復帰 (iPad)

ArUco マーカーで座標系を確立し、ALIKED/XFeat 記述子ベースの PnP で自己位置推定を行う iPad 向け Visual Odometry アプリ。
ロスト時には最大 20 枚のキーフレームデータベースを照合してポーズを復帰する。

## 主な機能

- **ArUco ブートストラップ** — 5×5 ArUco マーカーでメートルスケールの世界座標系を確立。ターゲット ID 指定可能。ブートストラップ完了後はマーカーを通常特徴点として扱う
- **階層的トラッキング** — ORB(10f) → XFeat(30f) → ALIKED(45f) の非同期パイプライン + LK オプティカルフロー。上位ティアが下位ティアの点をインプレースでプロモート
- **PnP 自己位置推定** — ALIKED 128次元記述子で VisualMap と照合 → solvePnP RANSAC
- **マップ自動成長** — `triInterval`(20f) ごとに三角測量で新規 3D 点を追加
- **多キーフレームロスト復帰** — 最大 20 枚の KF を保持。アクティブ KF は全件試行、休眠 KF はサムネイル類似度でゲーティング
- **定期ドリフト補正** — `correctionInterval`(60f) ごとに最新 KF と照合して姿勢をサイレント補正

## 動作環境

- iPad (iOS 17.0 以上)
- Xcode 15 以上
- [XcodeGen](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`

## ビルド手順

```bash
xcodegen generate
open VisualOdo.xcodeproj
# Signing & Capabilities でチームを設定してビルド
```

## VO パイプライン

```
ARKit フレーム (毎フレーム)
  │
  ▼ ARSessionManager
  procImage: UIImage (960×720)
  intrinsics: (fx, fy, cx, cy)
  │
  ▼ PointTracker.update(frame)
  LK フロー → 全点追跡
  非同期再検出: ORB/XFeat/ALIKED
  │ [TrackedPoint]
  ▼ VisualOdometryEngine.processFrame(...)
  │
  ├─[Searching]── ArUco 検出 → bootstrapPose 蓄積
  ├─[Bootstrapping]── 視差確認 → triangulate → VisualMap 初期化
  ├─[Tracking]──── findMatches → solvePnP → growMap
  └─[Lost]──────── KFデータベース照合 → 復帰
```

## VO ステートマシン

| Phase | 状態 | 遷移条件 |
|---|---|---|
| Searching | ArUco を探す | ArUco(ID一致) + ALIKED≥12点 → Bootstrapping |
| Bootstrapping | 視差待機 → 三角測量 | マップ点≥8 → Tracking / 共通点不足 → Searching |
| Tracking | PnP 自己位置推定 | PnP失敗/マッチ<6 → Lost |
| Lost | KF照合で復帰試行 | PnP成功 → Tracking / bootstrapComplete==false → Searching |

## キーフレームデータベース

| 設定 | デフォルト | 説明 |
|---|---|---|
| `maxKeyframes` | 20枚 | DB上限。超過時は最古を削除 |
| `keyframeInterval` | 30f | 保存間隔 (Tracking中) |
| `dormancyDelay` | 150f | 保存から何フレームでアクティブ→休眠 |
| `dormantCheckInterval` | 5f | ロスト中に休眠KFを試行する間隔 |

**アクティブ KF** (age ≤ 150f): ロスト時は毎回全件試行  
**休眠 KF** (age > 150f): 5フレームごとに 64×48 サムネイルのコサイン類似度で上位2件をウェイクアップして試行

## プロジェクト構成

```
Sources/
├── App/
│   └── VisualOdoApp.swift             @main エントリポイント
├── Camera/
│   ├── ARSessionManager.swift        ARKit セッション + procImage 配信
│   └── CameraPreviewView.swift       ARSCNView SwiftUI ラッパー
├── Tracking/
│   ├── PointTracker.swift            階層的 ORB→XFeat→ALIKED トラッカー
│   ├── TrackedPoint.swift            id / position / descriptor / tier
│   ├── TrackTier.swift               orb < xfeat < aliked
│   └── TrackingExtractor.swift       抽出器ストラテジー enum
├── Localization/                     ← VO コアモジュール
│   ├── VisualOdometryEngine.swift    状態機械 / PnP / KFデータベース / 復帰
│   └── VisualMap.swift               インメモリ 3D マップ (BLAS GEMM マッチング)
├── OpenCV/
│   ├── OpenCVBridge.h/mm             LKフロー / ORB / ArUco / PnP / 三角測量
│   ├── VisualOdo-Bridging-Header.h
│   └── opencv2.xcframework
├── ALIKED/
│   ├── ALIKEDMatcher.swift           CoreML 128次元記述子抽出
│   └── ALIKED.mlpackage
├── XFeat/
│   ├── XFeatMatcher.swift            CoreML 64次元記述子抽出
│   └── XFeat.mlpackage
├── SuperPoint/
│   ├── SuperPointMatcher.swift       CoreML (single-extractor モード)
│   └── SuperPoint.mlpackage
├── Views/
│   ├── ContentView.swift             ルートビュー
│   ├── VOModeView.swift              VO UI / オーバーレイ / 設定シート
│   └── PointCloudView.swift          SceneKit 3D点群ビューア
└── Reconstruction/                   (レガシー復元モード)
    ├── ReconstructionEngine.swift
    ├── PointCloud.swift
    ├── PLYExporter.swift
    └── MotionManager.swift
```

## 主要パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `targetMarkerID` | -1 (任意) | ブートストラップに使う ArUco ID |
| `markerSizeMeters` | 0.15m | マーカー物理サイズ |
| `minBootstrapParallax` | 20px | ブートストラップ最小視差 |
| `minPnPInliers` | 8 | PnP 成功最小インライア数 |
| `alikedInterval` | 45f | ALIKED 再検出間隔 |
| `alikedBudget` | 300 | ALIKED 点数上限 |
| `triInterval` | 20f | 三角測量間隔 |
| `correctionInterval` | 60f | ドリフト補正間隔 |

## ドキュメント

| ファイル | 説明 |
|---|---|
| `docs/workflow.drawio` | ステートマシン / フレーム処理 / KFデータベース フローチャート |
| `docs/CODEMAPS.md` | ファイル別コードマップ・API・データフロー |

## 使用フレームワーク

| フレームワーク | 用途 |
|---|---|
| ARKit | カメラフレーム取得・内部パラメータ |
| Accelerate (BLAS) | 記述子コサイン類似度行列計算 (cblas_sgemm / vDSP) |
| CoreML | ALIKED / XFeat / SuperPoint 推論 (CPU のみ) |
| CoreMotion | ジャイロ・加速度 (ReconstructionEngine) |
| SceneKit | 3D 点群プレビュー |
| opencv2 | LK フロー / ORB / ArUco / Essential Matrix / 三角測量 / PnP |

## 制限事項

- **ループ閉鎖なし** — 長時間走行で誤差が蓄積する
- **地図はインメモリのみ** — アプリ終了で消滅 (セッション内は保持)
- **CPU 専用 CoreML** — GPU のストライドパディング問題を回避するため
- **iPad 専用** — カメラ権限と向き処理が iPad 向け設定
