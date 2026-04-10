# CODEMAPS — XFeat3D

最終更新: 2026-04-10

## アーキテクチャ概要

```
ARKit (ARSessionManager)
    │ procImage (960×720 UIImage)
    │ intrinsics (fx, fy, cx, cy)
    ▼
PointTracker                    ← 特徴点追跡 (LK + 階層検出)
    │ [TrackedPoint]
    ▼
VisualOdometryEngine            ← VO状態機械 + PnP自己位置推定
    │ cameraPose (simd_float4x4)
    │ mapPointCount
    ▼
VOModeView (SwiftUI)            ← UI表示・設定
    │
    ├── VOTrackingOverlay        ← 特徴点オーバーレイ
    ├── ARAxesOverlay            ← 3D座標軸表示
    └── PointCloudView           ← 3D点群ビューア (SceneKit)

VisualMap (in-memory)           ← 3D地図点 + 記述子
    └── [VisualMapEntry]         ← position + descriptor

KeyframeDB (in-memory)          ← 復帰用キーフレームDB
    └── [StoredKeyframe]         ← image + thumbnail + map2D + cached features
```

---

## Sources/ ディレクトリ構成

### App/

| ファイル | 役割 |
|---|---|
| `XFeat3DApp.swift` | `@main` エントリポイント、`ContentView` を表示 |

---

### Camera/

| ファイル | 役割 |
|---|---|
| `ARSessionManager.swift` | ARKit `ARWorldTrackingConfiguration` セッション管理。各フレームを 960×720 `UIImage` に変換して `procImage` として配信。`scaledIntrinsics(for:)` で proc解像度に換算した内部パラメータを提供 |
| `CameraPreviewView.swift` | `ARSCNView` を SwiftUI でラップ (カメラ映像表示用) |

**主要 API:**
```swift
arManager.procImage          // @Published — VOModeView が subscribe
arManager.latestFrame        // 生 ARFrame (intrinsics 取得用)
arManager.scaledIntrinsics(for: ARFrame) -> (fx, fy, cx, cy)
```

---

### Tracking/

| ファイル | 役割 |
|---|---|
| `PointTracker.swift` | 階層的トラッカー (singleton `PointTracker.shared`) |
| `TrackedPoint.swift` | `id: Int, position: SIMD2<Float>, descriptor: [Float], tier: TrackTier` |
| `TrackTier.swift` | `.orb < .xfeat < .aliked` — Comparable |
| `TrackingExtractor.swift` | 抽出器ストラテジー enum (`.hierarchical` / `.xfeat` / `.aliked` / ...) |

**PointTracker 動作 (hierarchical モード):**
```
毎フレーム update(frame:)
  1. pendingALIKED / pendingXFeat / pendingORB を統合 (mergePending)
  2. LKオプティカルフロー → 全点を現在フレームへ追跡
  3. 非同期スケジュール:
     - ORB    毎 orbInterval(10)f     → 位置のみ、記述子なし
     - XFeat  毎 xfeatInterval(30)f   → 64次元記述子
     - ALIKED 毎 alikedInterval(45)f  → 128次元記述子 (最高品質)
  4. mergePending: 記述子マッチで既存点をプロモート / 新規点を追加
```

**主要パラメータ:**
| パラメータ | デフォルト | 説明 |
|---|---|---|
| `alikedInterval` | 45f | ALIKED再検出間隔 |
| `alikedBudget` | 300 | ALIKED点の上限 |
| `targetCount` | 500 | 全体点数上限 |

---

### Localization/ ← **メインモジュール**

#### VisualOdometryEngine.swift

`@MainActor` の ObservableObject。VO全体を管理する。

**状態機械:**
```
.searching → .bootstrapping → .tracking ↔ .lost
```

| Phase | 説明 |
|---|---|
| `.searching` | ArUco 5×5 検出、bootstrapPose と bootstrapDescriptors を蓄積 |
| `.bootstrapping` | 視差チェック → triangulate → VisualMap 初期化 |
| `.tracking` | PnP RANSAC で毎フレーム自己位置推定。定期的に growMap / addKeyframe |
| `.lost` | KFデータベース照合で復帰試行 |

**主要メソッド:**

| メソッド | 説明 |
|---|---|
| `processFrame(procImage:trackedPoints:intrinsics:)` | 毎フレーム呼ぶ唯一の公開 API |
| `tryMarkerInit(...)` | ArUco 検出 + bootstrapPose/Descriptors 更新 |
| `tryBootstrap(...)` | 記述子マッチ → 視差 → triangulate → .tracking 移行 |
| `trackingStep(...)` | visualMap.findMatches → solvePnP → growMap |
| `tryDescriptorRecovery(...)` | 複数KFを照合してロスト復帰 |
| `tryRecoverFromKeyframe(idx:curKPs:curDescs:intrinsics:)` | 1枚のKFに対するPnP試行 |
| `periodicCorrection(...)` | 最新KFを使った定期ドリフト補正 |
| `addKeyframe(from:aliked:procImage:descDim:)` | KFをデータベースに追加 |
| `mutualNNRaw(desc1:count1:desc2:count2:descDim:threshold:)` | BLAS GEMM + 相互最近傍 |
| `makeThumbnail(from:)` | 64×48 L2正規化グレースケールベクトル生成 |
| `thumbnailSimilarity(_:_:)` | vDSP_dotpr によるコサイン類似度 |
| `growMap(...)` | 三角測量で新規マップ点を追加 |

**@Published プロパティ (VOModeView が observe):**
```swift
phase: Phase
cameraPose: simd_float4x4?
mapPointCount: Int
pnpInlierCount: Int
matchCount: Int
activeKeyframeCount: Int
dormantKeyframeCount: Int
recoveryKeyframeImage: UIImage?
lastMatchedIDs: Set<Int>
lastTriangulatedIDs: Set<Int>
statusMessage: String
```

**キーフレームDB (StoredKeyframe):**
```swift
struct StoredKeyframe {
    image: UIImage                         // 復帰時の特徴抽出用
    thumbnail: [Float]                     // 64×48 L2正規化 (休眠ゲーティング用)
    map2D: [(mapIdx: Int, pos2D: SIMD2<Float>)]  // KF点 → 地図インデックス
    descDim: Int
    savedAtFrame: Int                      // 休眠判定基準
    cachedALIKED: ALIKEDFeatures?          // 遅延キャッシュ
    cachedXFeat: XFeatFeatures?
}
```

**チューニングパラメータ:**
| パラメータ | デフォルト | 説明 |
|---|---|---|
| `targetMarkerID` | -1 (任意) | ブートストラップに使うArUco ID |
| `markerSizeMeters` | 0.15m | マーカー物理サイズ |
| `minBootstrapParallax` | 20px | ブートストラップ最小視差 |
| `minBootstrapCommon` | 12点 | ブートストラップ最小共通点 |
| `minPnPInliers` | 8 | PnP成功最小インライア数 |
| `keyframeInterval` | 30f | KF保存間隔 |
| `correctionInterval` | 60f | ドリフト補正間隔 |
| `triInterval` | 20f | growMap 間隔 |
| `maxKeyframes` | 20枚 | KFデータベース上限 |
| `dormancyDelay` | 150f | KFが休眠に移行するフレーム数 |
| `dormantCheckInterval` | 5f | 休眠KFを照合する間隔 |
| `kfProximityPx` | 20px | map2D近傍探索半径 |

#### VisualMap.swift

```swift
final class VisualMap {
    var entries: [VisualMapEntry]   // 3D位置 + 128次元記述子
    func add(_ newEntries: [VisualMapEntry])
    func findMatches(queryDescriptors:queryCount:descDim:minScore:) -> [(entryIdx, queryIdx)]
    func clear()
}
```

`findMatches` は BLAS GEMM でコサイン類似度行列を計算し相互最近傍を返す。
**完全インメモリ** — アプリ終了で消滅。

---

### OpenCV/

| ファイル | 役割 |
|---|---|
| `OpenCVBridge.h` | Objective-C ブリッジヘッダ |
| `OpenCVBridge.mm` | OpenCV 4.x 実装 |
| `XFeat3D-Bridging-Header.h` | Swift ブリッジ |
| `opencv2.xcframework` | OpenCV iOS フレームワーク |

**提供 API:**

| メソッド | 説明 |
|---|---|
| `toGray(_:width:height:)` | UIImage → グレースケール `NSData` |
| `trackLK(_:count:prevGray:currGray:width:height:)` | スパース LK オプティカルフロー |
| `detectORB(from:topK:procWidth:procHeight:)` | ORB キーポイント検出 |
| `recoverPose(pts1:pts2:count:fx:fy:cx:cy:)` | Essential Matrix → R, t |
| `triangulatePoints(_:proj2:pts1:pts2:count:)` | 三角測量 → 3D点 |
| `detectArUco5x5(_:markerSize:procWidth:procHeight:fx:fy:cx:cy:)` | ArUco 5×5 検出 + 姿勢推定 |
| `solvePnP(points3D:points2D:count:fx:fy:cx:cy:)` | PnP RANSAC → R, t |

---

### ALIKED/ · XFeat/ · SuperPoint/

| ファイル | 役割 |
|---|---|
| `ALIKEDMatcher.swift` | CoreML推論、`extractFeatures(from:) -> ALIKEDFeatures?`、128次元記述子 |
| `XFeatMatcher.swift` | CoreML推論、`extractFeatures(from:) -> XFeatFeatures?`、64次元記述子 |
| `SuperPointMatcher.swift` | CoreML推論 (single-extractor モード用) |

各モデルは `.mlpackage` 形式、CPU推論のみ (GPUストライドパディング問題のため)。

---

### Views/

| ファイル | 役割 |
|---|---|
| `ContentView.swift` | ルートビュー。`ARSessionManager` を生成し `VOModeView` に渡す |
| `VOModeView.swift` | VO モードのメインUI |
| `PointCloudView.swift` | SceneKit ベースの3D点群ビューア |

**VOModeView の構成:**
```
ZStack
├── ARCameraView (カメラ映像)
├── VOTrackingOverlay (特徴点ドット — ORB赤/XFeat青/ALIKED緑/PnP黄/三角測量紫)
├── ARAxesOverlay (3D座標軸)
├── VStack
│   ├── phaseBar (フェーズ + ステータスメッセージ)
│   └── bottomPanel (座標/統計/凡例/リセット)
├── recoveryKeyframeImage サムネイル (右上、KF枚数表示)
└── pointCloudOverlay (モーダル)
```

**@AppStorage 設定キー (VOModeView):**
```
vo_markerSizeCM, vo_targetMarkerID, vo_minBootstrapParallax,
vo_minBootstrapCommon, vo_minPnPInliers, vo_triInterval,
vo_correctionInterval, vo_keyframeInterval, vo_minTriParallax,
vo_alikedBudget, vo_maxTierRaw, vo_showOnlyTopTier,
vo_orbSize/Color, vo_xfeatSize/Color, vo_alikedSize/Color,
vo_matchedSize, vo_triangulatedSize
```

---

### Reconstruction/ (レガシー)

VO モードとは独立した旧来の3D復元モード。現在のアプリでは主要フローから外れている。

| ファイル | 役割 |
|---|---|
| `ReconstructionEngine.swift` | Essential Matrix ベースの逐次復元 |
| `PointCloud.swift` | 3D点蓄積・マージ半径による重複除去 |
| `PLYExporter.swift` | カラー付き ASCII PLY 出力 |
| `MotionManager.swift` | CoreMotion ジャイロ/加速度ラッパー |

---

## データフロー図 (テキスト)

```
ARFrame
  │ pixel buffer → CIImage → UIImage (960×720)
  ▼
PointTracker.update(frame)
  │ LKフロー + 非同期 ALIKED/XFeat/ORB 再検出
  ▼
[TrackedPoint] (id, pos2D, descriptor[128or64], tier)
  │
  ▼
VisualOdometryEngine.processFrame(procImage, trackedPoints, intrinsics)
  │
  ├─[searching/lost(boot前)]─► tryMarkerInit
  │                               OpenCV ArUco 検出
  │                               → bootstrapPose, bootstrapDescriptors
  │
  ├─[bootstrapping]────────────► tryBootstrap
  │                               記述子マッチ + 視差チェック
  │                               → OpenCV triangulatePoints
  │                               → VisualMap.add (初期マップ点)
  │                               → addKeyframe (初回KF保存)
  │                               → phase = .tracking
  │
  ├─[tracking]─────────────────► trackingStep
  │                               VisualMap.findMatches (BLAS GEMM)
  │                               → solvePnP RANSAC → cameraPose
  │                               定期:
  │                                 addKeyframe → KeyframeDB.append
  │                                 periodicCorrection → 補正PnP
  │                                 growMap → 追加三角測量
  │
  └─[lost(boot後)]─────────────► tryDescriptorRecovery
                                  サムネイル類似度でKF候補選別
                                  ALIKED/XFeat抽出 (1回、全KFで共有)
                                  各KF: mutualNNRaw → map2D近傍 → solvePnP
                                  成功: cameraPose更新 → phase = .tracking
```

---

## ビルド方法

```bash
# Xcodeプロジェクト再生成
xcodegen generate

# ビルド (実機のみ)
open XFeat3D.xcodeproj
# Signing & Capabilities でチームを設定してからビルド
```

**依存関係:**
- Xcode 15+, iOS 17.0+, iPad
- [XcodeGen](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`
- `opencv2.xcframework` (Sources/OpenCV/ に配置済み)
- CoreML モデル: `XFeat.mlpackage`, `ALIKED.mlpackage`, `SuperPoint.mlpackage`

---

## 関連ドキュメント

- `docs/workflow.drawio` — ステートマシン / フレーム処理パイプライン / KFデータベース フローチャート (Draw.io形式)
- `README.md` — セットアップ・機能概要
