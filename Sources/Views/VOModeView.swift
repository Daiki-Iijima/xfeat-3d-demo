import SwiftUI
import ARKit
import simd

// MARK: - Tier display configuration

struct TierDisplayConfig {
    var orbSize:          CGFloat
    var xfeatSize:        CGFloat
    var alikedSize:       CGFloat
    var matchedSize:      CGFloat  // PnP対応点（黄）の半径
    var triangulatedSize: CGFloat  // 三角測量点（紫）の半径
    var orbColor:         Color
    var xfeatColor:       Color
    var alikedColor:      Color
    var showOnlyTopTier:  Bool
}

// Color presets (index stored in @AppStorage)
let kTierColors: [Color] = [.red, .orange, .yellow, .green, .cyan, .blue, .purple, .white, .gray]
let kTierColorNames: [String] = ["赤", "橙", "黄", "緑", "シアン", "青", "紫", "白", "灰"]

/// Visual Odometry mode view.
///
/// Uses ARSessionManager for camera access + intrinsics.
/// Pose is estimated entirely from marker-initialized PnP (no ARKit VIO).
struct VOModeView: View {
    @ObservedObject var arManager: ARSessionManager

    @StateObject private var voEngine = VisualOdometryEngine()
    @State private var trackedPoints: [TrackedPoint] = []
    @State private var currentIntrinsics: (fx: Float, fy: Float, cx: Float, cy: Float) =
        (fx: 800, fy: 800, cx: 480, cy: 360)
    @State private var showPointCloud = false
    @State private var showSettings = false
    /// Full-screen size measured by an ignoresSafeArea GeometryReader.
    /// Overlays use this so they match the camera feed's extent exactly.
    @State private var fullScreenSize: CGSize = UIScreen.main.bounds.size

    // MARK: - Persisted Settings (UserDefaults via @AppStorage)
    @AppStorage("vo_markerSizeCM")         private var markerSizeCM:          Double = 15.0
    @AppStorage("vo_minBootstrapParallax") private var minBootstrapParallax:  Double = 20.0
    @AppStorage("vo_minBootstrapCommon")   private var minBootstrapCommon:    Int    = 12
    @AppStorage("vo_minPnPInliers")        private var minPnPInliers:         Int    = 8
    @AppStorage("vo_triInterval")          private var triInterval:           Int    = 20
    @AppStorage("vo_minTriParallax")       private var minTriParallax:        Double = 8.0
    @AppStorage("vo_alikedBudget")         private var alikedBudget:          Int    = 150

    // Feature tier / display settings
    @AppStorage("vo_maxTierRaw")      private var maxTierRaw:      Int    = 2      // ALIKED
    @AppStorage("vo_showOnlyTopTier") private var showOnlyTopTier: Bool   = true
    @AppStorage("vo_orbSize")         private var orbSize:         Double = 3
    @AppStorage("vo_xfeatSize")       private var xfeatSize:       Double = 4
    @AppStorage("vo_alikedSize")      private var alikedSize:       Double = 5
    @AppStorage("vo_matchedSize")     private var matchedSize:     Double = 10
    @AppStorage("vo_triangulatedSize")private var triangulatedSize:Double = 9
    @AppStorage("vo_orbColorIdx")     private var orbColorIdx:     Int    = 0  // red
    @AppStorage("vo_xfeatColorIdx")   private var xfeatColorIdx:   Int    = 5  // blue
    @AppStorage("vo_alikedColorIdx")  private var alikedColorIdx:  Int    = 3  // green

    private let procSize = CGSize(width: 960, height: 720)

    var body: some View {
        ZStack {
            // ── Full-screen size measurement (invisible) ─────────────────
            GeometryReader { geo in
                Color.clear
                    .onAppear     { fullScreenSize = geo.size }
                    .onChange(of: geo.size) { fullScreenSize = $0 }
            }
            .ignoresSafeArea()

            // ── Camera feed ──────────────────────────────────────────────
            ARCameraView(arSession: arManager.arSession)
                .ignoresSafeArea()

            // ── Feature points: tier color + PnP/tri highlights ──────────
            VOTrackingOverlay(
                points: trackedPoints,
                matchedIDs: voEngine.lastMatchedIDs,
                triangulatedIDs: voEngine.lastTriangulatedIDs,
                procSize: procSize,
                displaySize: fullScreenSize,
                config: currentDisplayConfig
            )
            .ignoresSafeArea()

            // ── AR object (axes + cube anchored at marker origin) ────────
            if let pose = voEngine.cameraPose {
                ARAxesOverlay(
                    cameraPose: pose,
                    intrinsics: currentIntrinsics,
                    procSize: procSize,
                    displaySize: fullScreenSize,
                    isMarkerInView: voEngine.isMarkerInView
                )
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }

            // ── Status bar + pose panel (respect safe areas) ─────────────
            VStack(spacing: 0) {
                phaseBar
                Spacer()
                bottomPanel
            }
            .ignoresSafeArea(edges: .bottom)

            // ── Point cloud viewer (modal) ───────────────────────────────
            if showPointCloud {
                pointCloudOverlay
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .navigationTitle("VO モード")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                HStack {
                    Button { showSettings.toggle() } label: {
                        Image(systemName: "gearshape")
                    }
                    Button { showPointCloud.toggle() } label: {
                        Image(systemName: "cube.transparent")
                    }
                }
            }
        }
        .sheet(isPresented: $showSettings) { settingsSheet }
        .onAppear { applySettings() }
        .onReceive(arManager.$procImage.compactMap { $0 }) { frame in
            guard let arFrame = arManager.latestFrame else { return }
            let tracked = PointTracker.shared.update(frame: frame)
            trackedPoints = tracked
            let intr = arManager.scaledIntrinsics(for: arFrame)
            currentIntrinsics = intr
            voEngine.processFrame(procImage: frame, trackedPoints: tracked, intrinsics: intr)
        }
        .onDisappear { voEngine.reset() }
    }

    // MARK: - Phase Bar

    private var phaseBar: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(phaseColor)
                .frame(width: 10, height: 10)
            Text(voEngine.phase.description)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)

            // Marker / PnP source badge
            if voEngine.phase == .tracking {
                if voEngine.isMarkerInView {
                    Label("ArUco", systemImage: "viewfinder")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.green)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.green.opacity(0.2), in: Capsule())
                } else {
                    Label("PnP自己位置推定", systemImage: "location.fill")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.cyan)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.cyan.opacity(0.2), in: Capsule())
                }
            }

            Spacer()
            Text(voEngine.statusMessage)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.8))
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.black.opacity(0.55))
    }

    private var phaseColor: Color {
        switch voEngine.phase {
        case .searching:     return .yellow
        case .bootstrapping: return .orange
        case .tracking:      return .green
        case .lost:          return .red
        }
    }

    // MARK: - Bottom Panel

    private var bottomPanel: some View {
        VStack(spacing: 0) {
            poseReadout
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, 4)

            // Legend row
            HStack(spacing: 14) {
                legendDot(color: .green,   label: "ALIKED")
                legendDot(color: .yellow,  label: "PnP対応点", ring: true)
                legendDot(color: .purple, label: "新規三角測量", ring: true)
                legendDot(color: .blue,    label: "XFeat")
                legendDot(color: .red,     label: "ORB")
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 4)

            // Stats row
            HStack(spacing: 14) {
                statBadge(icon: "mappin.and.ellipse",         value: "\(voEngine.mapPointCount)",  label: "マップ点")
                statBadge(icon: "arrow.triangle.2.circlepath", value: "\(voEngine.pnpInlierCount)", label: "PnP")
                statBadge(icon: "dot.radiowaves.left.and.right", value: "\(voEngine.matchCount)",  label: "マッチ")
                statBadge(icon: "scope",
                          value: "\(trackedPoints.filter { $0.tier == .aliked }.count)",
                          label: "ALIKED")
                Spacer()
                Button(role: .destructive) {
                    voEngine.reset()
                    PointTracker.shared.reset()
                } label: {
                    Label("リセット", systemImage: "arrow.counterclockwise")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.red.opacity(0.8), in: Capsule())
                        .foregroundStyle(.white)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .background(.black.opacity(0.65))
    }

    // MARK: - Pose Readout (always visible)

    private var poseReadout: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "location.north.line.fill")
                    .font(.caption2)
                    .foregroundStyle(voEngine.cameraPose != nil ? Color.cyan : Color.gray)
                Text(voEngine.cameraPose != nil ? "マーカー相対 カメラ位置" : "マーカー未検出 — 位置不明")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(voEngine.cameraPose != nil ? Color.cyan : Color.gray)
                Spacer()
                if let dist = distanceFromMarker {
                    Text(String(format: "距離 %.3f m", dist))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Color.cyan.opacity(0.8))
                }
            }
            HStack(spacing: 0) {
                let p = voEngine.cameraPose?.columns.3
                let yaw: Float? = voEngine.cameraPose.map {
                    atan2($0.columns.0.z, $0.columns.2.z) * 180 / .pi
                }
                coordCell(label: "X",   value: p.map { $0.x }, unit: "m")
                coordCell(label: "Y",   value: p.map { $0.y }, unit: "m")
                coordCell(label: "Z",   value: p.map { $0.z }, unit: "m")
                Divider()
                    .frame(height: 20)
                    .background(Color.white.opacity(0.3))
                    .padding(.horizontal, 8)
                coordCell(label: "Yaw", value: yaw, unit: "°", decimals: 1)
            }
        }
    }

    private var distanceFromMarker: Float? {
        guard let p = voEngine.cameraPose?.columns.3 else { return nil }
        return sqrt(p.x*p.x + p.y*p.y + p.z*p.z)
    }

    @ViewBuilder
    private func coordCell(label: String, value: Float?, unit: String, decimals: Int = 3) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(Color.white.opacity(0.5))
            if let v = value {
                Text(decimals == 1
                     ? String(format: "%+.1f\(unit)", v)
                     : String(format: "%+.3f\(unit)", v))
                    .font(.system(size: 13, design: .monospaced).weight(.medium))
                    .foregroundStyle(Color.cyan)
            } else {
                Text("---")
                    .font(.system(size: 13, design: .monospaced).weight(.medium))
                    .foregroundStyle(Color.gray)
            }
        }
        .frame(minWidth: 72, alignment: .leading)
    }

    @ViewBuilder
    private func statBadge(icon: String, value: String, label: String) -> some View {
        VStack(spacing: 2) {
            HStack(spacing: 3) {
                Image(systemName: icon).font(.caption2)
                Text(value).font(.caption.weight(.bold))
            }
            .foregroundStyle(.white)
            Text(label).font(.system(size: 9)).foregroundStyle(.white.opacity(0.6))
        }
    }

    @ViewBuilder
    private func legendDot(color: Color, label: String, ring: Bool = false) -> some View {
        HStack(spacing: 4) {
            ZStack {
                if ring {
                    Circle()
                        .stroke(color, lineWidth: 1.5)
                        .frame(width: 11, height: 11)
                }
                Circle()
                    .fill(color)
                    .frame(width: 6, height: 6)
            }
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(Color.white.opacity(0.7))
        }
    }

    // MARK: - Point Cloud Overlay

    private var pointCloudOverlay: some View {
        ZStack(alignment: .topTrailing) {
            PointCloudView(points: voEngine.pointCloud.points)
                .ignoresSafeArea()
                .background(Color.black)
            Button { showPointCloud = false } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2).foregroundStyle(.white).padding(16)
            }
        }
    }

    // MARK: - Display Config

    private var currentDisplayConfig: TierDisplayConfig {
        TierDisplayConfig(
            orbSize:          CGFloat(orbSize),
            xfeatSize:        CGFloat(xfeatSize),
            alikedSize:       CGFloat(alikedSize),
            matchedSize:      CGFloat(matchedSize),
            triangulatedSize: CGFloat(triangulatedSize),
            orbColor:         kTierColors[min(orbColorIdx,    kTierColors.count - 1)],
            xfeatColor:       kTierColors[min(xfeatColorIdx,  kTierColors.count - 1)],
            alikedColor:      kTierColors[min(alikedColorIdx, kTierColors.count - 1)],
            showOnlyTopTier:  showOnlyTopTier
        )
    }

    // MARK: - Apply Settings → Engine

    private func applySettings() {
        voEngine.markerSizeMeters      = Float(markerSizeCM) / 100.0
        voEngine.minBootstrapParallax  = Float(minBootstrapParallax)
        voEngine.minBootstrapCommon    = minBootstrapCommon
        voEngine.minPnPInliers         = minPnPInliers
        voEngine.triInterval           = triInterval
        voEngine.minTriParallax        = Float(minTriParallax)
        PointTracker.shared.alikedBudget = alikedBudget
        PointTracker.shared.maxTier      = TrackTier(rawValue: maxTierRaw) ?? .aliked
    }

    // MARK: - Settings Sheet

    private var settingsSheet: some View {
        NavigationStack {
            Form {
                Section("マーカー") {
                    HStack {
                        Text("サイズ")
                        Spacer()
                        Stepper(
                            String(format: "%.1f cm", markerSizeCM),
                            value: $markerSizeCM,
                            in: 1...100, step: 0.5
                        )
                    }
                }
                Section("ブートストラップ") {
                    paramSliderD(label: "最小視差",   value: $minBootstrapParallax, range: 5...50,  format: "%.0fpx")
                    paramSliderI(label: "最小共通点", value: $minBootstrapCommon,   range: 4...30,  format: "%d点")
                }
                Section("トラッキング") {
                    paramSliderI(label: "PnP最小インライア", value: $minPnPInliers,  range: 4...20,  format: "%d点")
                    paramSliderI(label: "三角測量間隔",      value: $triInterval,    range: 5...60,  format: "%dフレーム")
                    paramSliderD(label: "三角測量最小視差",  value: $minTriParallax, range: 2...30,  format: "%.0fpx")
                }
                Section("ALIKED点数") {
                    paramSliderI(label: "上限", value: $alikedBudget, range: 50...500, format: "%d点", step: 25)
                }

                Section(header: Text("特徴点 — 使用レベル")) {
                    Picker("最高ティア", selection: $maxTierRaw) {
                        Text("ORB のみ (最速)").tag(0)
                        Text("ORB + XFeat").tag(1)
                        Text("ALIKED まで (高精度)").tag(2)
                    }
                    .pickerStyle(.segmented)
                    Toggle("最高レベルのみ表示", isOn: $showOnlyTopTier)
                }

                Section(header: Text("特徴点 — サイズ")) {
                    paramSliderD(label: "ORB",          value: $orbSize,           range: 1...16, format: "%.0fpx")
                    paramSliderD(label: "XFeat",        value: $xfeatSize,         range: 1...16, format: "%.0fpx")
                    paramSliderD(label: "ALIKED",       value: $alikedSize,        range: 1...16, format: "%.0fpx")
                    paramSliderD(label: "PnP対応 ◇",    value: $matchedSize,      range: 1...24, format: "%.0fpx")
                    paramSliderD(label: "三角測量 ✕",   value: $triangulatedSize, range: 1...24, format: "%.0fpx")
                }

                Section(header: Text("特徴点 — カラー")) {
                    colorPicker(label: "ORB",    selection: $orbColorIdx)
                    colorPicker(label: "XFeat",  selection: $xfeatColorIdx)
                    colorPicker(label: "ALIKED",  selection: $alikedColorIdx)
                }
                Section {
                    Button(role: .destructive) {
                        voEngine.reset(); PointTracker.shared.reset()
                    } label: {
                        Label("VO エンジンをリセット", systemImage: "arrow.counterclockwise")
                    }
                }
            }
            .navigationTitle("VO 設定")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("完了") {
                        applySettings()
                        showSettings = false
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func paramSliderD(label: String, value: Binding<Double>,
                               range: ClosedRange<Double>, format: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .foregroundStyle(.secondary).font(.caption)
            }
            Slider(value: value, in: range)
        }
    }

    @ViewBuilder
    private func paramSliderI(label: String, value: Binding<Int>,
                               range: ClosedRange<Int>, format: String, step: Int = 1) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(String(format: format, value.wrappedValue))
                .foregroundStyle(.secondary).font(.caption)
            Stepper("", value: value, in: range, step: step).labelsHidden()
        }
    }

    @ViewBuilder
    private func colorPicker(label: String, selection: Binding<Int>) -> some View {
        HStack {
            Text(label)
            Spacer()
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(0..<kTierColors.count, id: \.self) { idx in
                        Button {
                            selection.wrappedValue = idx
                        } label: {
                            ZStack {
                                Circle().fill(kTierColors[idx]).frame(width: 26, height: 26)
                                if selection.wrappedValue == idx {
                                    Circle().stroke(.white, lineWidth: 2.5).frame(width: 26, height: 26)
                                    Circle().stroke(.black.opacity(0.4), lineWidth: 0.5).frame(width: 29, height: 29)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }
}

// MARK: - Proc→Display coordinate transform

/// Maps a point from proc space (960×720, camera-native landscape-right orientation)
/// to display space, applying:
///   1. Interface-orientation rotation (so the point matches ARSCNView's rotated feed)
///   2. AspectFill scaling (ARSCNView scales the camera background to fill its bounds)
///
/// ARSCNView always renders the capturedImage in the correct display orientation with
/// aspectFill, so this function replicates that transform for overlay alignment.
func procToDisplay(_ pt: CGPoint,
                   procSize: CGSize,
                   displaySize: CGSize,
                   orientation: UIInterfaceOrientation) -> CGPoint {

    // Step 1 — rotate proc point to match the current interface orientation.
    // Camera native = landscapeRight (iPad rear camera natural axis).
    var rotPt = pt
    var effectiveW = procSize.width   // proc width after rotation
    var effectiveH = procSize.height  // proc height after rotation

    switch orientation {
    case .landscapeRight:
        break  // no change

    case .landscapeLeft:
        // 180°: flip around centre
        rotPt = CGPoint(x: procSize.width  - pt.x,
                        y: procSize.height - pt.y)

    case .portrait:
        // 90° CCW rotation: (x,y) → (y, W−x), dimensions swap
        rotPt    = CGPoint(x: pt.y, y: procSize.width - pt.x)
        effectiveW = procSize.height   // 720
        effectiveH = procSize.width    // 960

    case .portraitUpsideDown:
        // 90° CW rotation: (x,y) → (H−y, x), dimensions swap
        rotPt    = CGPoint(x: procSize.height - pt.y, y: pt.x)
        effectiveW = procSize.height   // 720
        effectiveH = procSize.width    // 960

    default:
        break
    }

    // Step 2 — aspectFill: scale so the larger dimension fills the display,
    // then centre (the other dimension is cropped symmetrically).
    let scale = max(displaySize.width / effectiveW, displaySize.height / effectiveH)
    let sx = scale * (rotPt.x - effectiveW / 2) + displaySize.width  / 2
    let sy = scale * (rotPt.y - effectiveH / 2) + displaySize.height / 2
    return CGPoint(x: sx, y: sy)
}

/// Returns the current interface orientation of the key window, defaulting to landscapeRight.
func currentInterfaceOrientation() -> UIInterfaceOrientation {
    UIApplication.shared.connectedScenes
        .compactMap { $0 as? UIWindowScene }
        .first?.interfaceOrientation ?? .landscapeRight
}

// MARK: - VO Tracking Overlay

/// Feature point overlay with color coding by tier and role.
///
/// | Color   | Meaning                                           |
/// |---------|---------------------------------------------------|
/// | Red     | ORB — position only                               |
/// | Blue    | XFeat — 64-dim descriptor                         |
/// | Green   | ALIKED — tracked, not yet matched to map          |
/// | Yellow ◇| ALIKED + matched to 3D map (PnP input this frame) |
/// | Purple ✕| ALIKED + just triangulated into the map           |
struct VOTrackingOverlay: View {
    let points: [TrackedPoint]
    let matchedIDs: Set<Int>
    let triangulatedIDs: Set<Int>
    let procSize: CGSize
    let displaySize: CGSize
    let config: TierDisplayConfig

    var body: some View {
        Canvas { ctx, _ in
            let ori = currentInterfaceOrientation()

            // Determine which tiers to render.
            // PnP-matched and triangulated points are always shown regardless of the filter
            // so the user can verify localization quality even in "top tier only" mode.
            let topTier = points.map { $0.tier }.max() ?? .orb
            let visiblePoints: [TrackedPoint] = config.showOnlyTopTier
                ? points.filter {
                    $0.tier == topTier
                    || matchedIDs.contains($0.id)
                    || triangulatedIDs.contains($0.id)
                  }
                : points

            for pt in visiblePoints {
                let sp = procToDisplay(
                    CGPoint(x: CGFloat(pt.position.x), y: CGFloat(pt.position.y)),
                    procSize: procSize, displaySize: displaySize, orientation: ori
                )
                let x = sp.x, y = sp.y
                let isTri     = triangulatedIDs.contains(pt.id)
                let isMatched = matchedIDs.contains(pt.id)

                // Base style from config
                let (baseColor, baseR): (Color, CGFloat) = switch pt.tier {
                    case .orb:    (config.orbColor,    config.orbSize)
                    case .xfeat:  (config.xfeatColor,  config.xfeatSize)
                    case .aliked: (config.alikedColor,  config.alikedSize)
                }

                // PnP-matched and triangulated decorations apply to any tier
                if isTri {
                    // Purple ring + dot + cross
                    let r = config.triangulatedSize
                    ctx.stroke(Path(ellipseIn: CGRect(x: x-r, y: y-r, width: r*2, height: r*2)),
                               with: .color(.purple.opacity(0.9)), lineWidth: 2)
                    ctx.fill(Path(ellipseIn: CGRect(x: x-baseR, y: y-baseR, width: baseR*2, height: baseR*2)),
                             with: .color(.purple))
                    let arm = r * 0.6
                    var cross = Path()
                    cross.move(to: CGPoint(x: x-arm, y: y)); cross.addLine(to: CGPoint(x: x+arm, y: y))
                    cross.move(to: CGPoint(x: x, y: y-arm)); cross.addLine(to: CGPoint(x: x, y: y+arm))
                    ctx.stroke(cross, with: .color(.white.opacity(0.9)), lineWidth: 1.5)

                } else if isMatched {
                    // Yellow ring + dot + diamond
                    let r = config.matchedSize
                    ctx.stroke(Path(ellipseIn: CGRect(x: x-r, y: y-r, width: r*2, height: r*2)),
                               with: .color(.yellow.opacity(0.85)), lineWidth: 2.5)
                    ctx.fill(Path(ellipseIn: CGRect(x: x-baseR, y: y-baseR, width: baseR*2, height: baseR*2)),
                             with: .color(.yellow))
                    let d = r * 0.8
                    var diamond = Path()
                    diamond.move(to: CGPoint(x: x,        y: y-d))
                    diamond.addLine(to: CGPoint(x: x+d*0.7, y: y))
                    diamond.addLine(to: CGPoint(x: x,        y: y+d))
                    diamond.addLine(to: CGPoint(x: x-d*0.7, y: y))
                    diamond.closeSubpath()
                    ctx.stroke(diamond, with: .color(.white.opacity(0.6)), lineWidth: 1)

                } else {
                    // Plain dot — style varies by tier
                    switch pt.tier {
                    case .orb:
                        ctx.fill(Path(ellipseIn: CGRect(x: x-baseR, y: y-baseR, width: baseR*2, height: baseR*2)),
                                 with: .color(baseColor.opacity(0.7)))
                    case .xfeat:
                        ctx.fill(Path(ellipseIn: CGRect(x: x-baseR, y: y-baseR, width: baseR*2, height: baseR*2)),
                                 with: .color(baseColor.opacity(0.8)))
                    case .aliked:
                        let ringR = baseR + 3
                        ctx.stroke(Path(ellipseIn: CGRect(x: x-ringR, y: y-ringR, width: ringR*2, height: ringR*2)),
                                   with: .color(baseColor.opacity(0.45)), lineWidth: 1)
                        ctx.fill(Path(ellipseIn: CGRect(x: x-baseR, y: y-baseR, width: baseR*2, height: baseR*2)),
                                 with: .color(baseColor.opacity(0.85)))
                    }
                }
            }
        }
    }
}

// MARK: - AR Axes + Cube Overlay

/// Projects coordinate axes, a wireframe cube, and a floor grid at the world origin (ArUco marker).
///
/// - When the marker **is** in view: white/standard colour scheme.
/// - When tracking **without** the marker (PnP only): cyan accent + "PnP" badge
///   to make it obvious that marker-free self-localisation is running.
struct ARAxesOverlay: View {
    let cameraPose:    simd_float4x4
    let intrinsics:    (fx: Float, fy: Float, cx: Float, cy: Float)
    let procSize:      CGSize
    let displaySize:   CGSize
    /// True when ArUco was seen recently; false = pure-PnP tracking.
    let isMarkerInView: Bool

    // ── Object dimensions ────────────────────────────────────────────────────
    private let axisLen:  Float = 4.0    // 4 m axes
    private let cubeHalf: Float = 0.20   // 40 cm origin cube (near-field ref)
    private let gridSize: Float = 5.0    // ±5 m → 10 m total floor grid
    private let gridStep: Float = 1.0    // 1 m cells

    var body: some View {
        Canvas { ctx, _ in
            let vm  = cameraPose.inverse
            let ori = currentInterfaceOrientation()

            // Accent colour switches with marker visibility
            let accent: Color = isMarkerInView ? .white : .cyan

            // ── Surface grid (Z = 0 plane = marker surface, 10 m × 10 m) ──
            // OpenCV ArUco: X=right, Y=down-in-image, Z=toward camera (front face).
            // The marker's flat surface is Z=0; the grid lives here and extends in X,Y.
            let baseAlpha: Double = isMarkerInView ? 0.22 : 0.38
            let gridColor = Color(isMarkerInView ? .white : .cyan)
            let steps = Int((gridSize / gridStep).rounded()) // 5
            for i in -steps...steps {
                let f = Float(i) * gridStep
                let isBoundary = (i == -steps || i == steps)
                let lw: CGFloat = isBoundary ? 2.5 : 1.0
                let alpha = isBoundary ? min(baseAlpha * 1.6, 1.0) : baseAlpha

                // Lines along Y at X = f  (Z = 0)
                if let p1 = project(SIMD3<Float>(f, -gridSize, 0), vm: vm, ori: ori),
                   let p2 = project(SIMD3<Float>(f,  gridSize, 0), vm: vm, ori: ori) {
                    var l = Path(); l.move(to: p1); l.addLine(to: p2)
                    ctx.stroke(l, with: .color(gridColor.opacity(alpha)), lineWidth: lw)
                }
                // Lines along X at Y = f  (Z = 0)
                if let p1 = project(SIMD3<Float>(-gridSize, f, 0), vm: vm, ori: ori),
                   let p2 = project(SIMD3<Float>( gridSize, f, 0), vm: vm, ori: ori) {
                    var l = Path(); l.move(to: p1); l.addLine(to: p2)
                    ctx.stroke(l, with: .color(gridColor.opacity(alpha)), lineWidth: lw)
                }
            }

            // Distance labels along +X axis on the surface (Z=0, Y=0)
            for i in 1...steps {
                let f = Float(i) * gridStep
                if let p = project(SIMD3<Float>(f, 0, 0), vm: vm, ori: ori) {
                    ctx.draw(
                        Text("\(i)m").font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(gridColor.opacity(0.55)),
                        at: p
                    )
                }
            }

            // ── Axes ─────────────────────────────────────────────────────
            let o  = project(.zero,                         vm: vm, ori: ori)
            let xT = project(SIMD3<Float>(axisLen, 0, 0),  vm: vm, ori: ori)
            let yT = project(SIMD3<Float>(0, axisLen, 0),  vm: vm, ori: ori)
            let zT = project(SIMD3<Float>(0, 0, axisLen),  vm: vm, ori: ori)

            if let o, let x = xT { drawAxis(ctx: ctx, from: o, to: x, color: .red,   label: "X", vm: vm, ori: ori, dir: SIMD3<Float>(1,0,0)) }
            if let o, let y = yT { drawAxis(ctx: ctx, from: o, to: y, color: .green, label: "Y", vm: vm, ori: ori, dir: SIMD3<Float>(0,1,0)) }
            if let o, let z = zT { drawAxis(ctx: ctx, from: o, to: z, color: .blue,  label: "Z", vm: vm, ori: ori, dir: SIMD3<Float>(0,0,1)) }

            // Origin sphere
            if let o {
                ctx.fill(Path(ellipseIn: CGRect(x: o.x-8, y: o.y-8, width: 16, height: 16)),
                         with: .color(accent))
                ctx.stroke(Path(ellipseIn: CGRect(x: o.x-8, y: o.y-8, width: 16, height: 16)),
                           with: .color(.black.opacity(0.4)), lineWidth: 1)
            }

            // ── Wireframe cube (sits on Z=0 marker surface, top at Z=2h) ──
            // Corner layout:
            //   0-3: bottom face (Z=0, on marker surface)  →  base / back toward viewer
            //   4-7: top face   (Z=2h, Z+ = front face)    →  front / toward Z+
            let h = cubeHalf
            let corners3D: [SIMD3<Float>] = [
                .init(-h, -h, 0),    // 0
                .init( h, -h, 0),    // 1
                .init( h,  h, 0),    // 2
                .init(-h,  h, 0),    // 3
                .init(-h, -h, 2*h),  // 4  ← Z+
                .init( h, -h, 2*h),  // 5
                .init( h,  h, 2*h),  // 6
                .init(-h,  h, 2*h),  // 7
            ]
            let edges: [(Int, Int)] = [
                (0,1),(1,2),(2,3),(3,0),  // bottom (on surface)
                (4,5),(5,6),(6,7),(7,4),  // top (Z+ front face)
                (0,4),(1,5),(2,6),(3,7),  // pillars
            ]
            let pts2D = corners3D.map { project($0, vm: vm, ori: ori) }

            let edgeAlpha: Double = isMarkerInView ? 0.85 : 1.0
            let edgeWidth: CGFloat = isMarkerInView ? 2.0  : 3.0
            for (a, b) in edges {
                guard let p1 = pts2D[a], let p2 = pts2D[b] else { continue }
                var e = Path(); e.move(to: p1); e.addLine(to: p2)
                ctx.stroke(e, with: .color(accent.opacity(edgeAlpha)),
                           style: StrokeStyle(lineWidth: edgeWidth, lineCap: .round))
            }

            // Bottom face filled (on marker surface, Z=0)
            let base = [0,1,2,3].compactMap { pts2D[$0] }
            if base.count == 4 {
                var face = Path()
                face.move(to: base[0]); base.dropFirst().forEach { face.addLine(to: $0) }
                face.closeSubpath()
                ctx.fill(face, with: .color(accent.opacity(0.08)))
            }

            // Front face (Z+ top) filled — brighter to emphasize Z+ direction
            let front = [4,5,6,7].compactMap { pts2D[$0] }
            if front.count == 4 {
                var face = Path()
                face.move(to: front[0]); front.dropFirst().forEach { face.addLine(to: $0) }
                face.closeSubpath()
                ctx.fill(face, with: .color(accent.opacity(0.22)))
                ctx.stroke(face, with: .color(accent), style: StrokeStyle(lineWidth: edgeWidth + 1))
            }

            // ── PnP badge when marker is not in view (above the front face) ─
            if !isMarkerInView, let topPt = [4,5,6,7].compactMap({ pts2D[$0] }).min(by: { $0.y < $1.y }) {
                let badgeCenter = CGPoint(x: topPt.x, y: topPt.y - 24)
                let label = Text("📍 PnP 自己位置推定中")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .foregroundStyle(Color.cyan)
                ctx.draw(label, at: badgeCenter)
            }
        }
    }

    private func project(_ w: SIMD3<Float>, vm: simd_float4x4,
                         ori: UIInterfaceOrientation) -> CGPoint? {
        let (fx, fy, cx, cy) = intrinsics
        let c = vm * SIMD4<Float>(w, 1)
        guard c.z > 0.001 else { return nil }
        let u = (c.x / c.z) * fx + cx
        let v = (c.y / c.z) * fy + cy
        return procToDisplay(CGPoint(x: CGFloat(u), y: CGFloat(v)),
                             procSize: procSize, displaySize: displaySize, orientation: ori)
    }

    private func drawAxis(ctx: GraphicsContext, from: CGPoint, to: CGPoint,
                          color: Color, label: String,
                          vm: simd_float4x4, ori: UIInterfaceOrientation,
                          dir: SIMD3<Float>) {
        var line = Path(); line.move(to: from); line.addLine(to: to)
        ctx.stroke(line, with: .color(color),
                   style: StrokeStyle(lineWidth: 6, lineCap: .round))

        // Arrow head
        let dx = to.x - from.x, dy = to.y - from.y
        let len = sqrt(dx*dx + dy*dy); guard len > 1 else { return }
        let ux = dx/len, uy = dy/len
        let al: CGFloat = 18
        var arr = Path()
        arr.move(to: to)
        arr.addLine(to: CGPoint(x: to.x - al*(ux + uy*0.5), y: to.y - al*(uy - ux*0.5)))
        arr.move(to: to)
        arr.addLine(to: CGPoint(x: to.x - al*(ux - uy*0.5), y: to.y - al*(uy + ux*0.5)))
        ctx.stroke(arr, with: .color(color),
                   style: StrokeStyle(lineWidth: 4, lineCap: .round))

        // Axis label
        ctx.draw(
            Text(label).font(.system(size: 16, weight: .black)).foregroundStyle(color),
            at: CGPoint(x: to.x + ux*20, y: to.y + uy*20)
        )

        // 1 m tick marks along the axis
        let tickCount = Int(axisLen.rounded())
        for m in 1..<tickCount {
            let wp = dir * Float(m)
            guard let tp = project(wp, vm: vm, ori: ori) else { continue }
            // Perpendicular direction for tick
            let px: CGFloat = -uy * 8, py: CGFloat = ux * 8
            var tick = Path()
            tick.move(to: CGPoint(x: tp.x - px, y: tp.y - py))
            tick.addLine(to: CGPoint(x: tp.x + px, y: tp.y + py))
            ctx.stroke(tick, with: .color(color.opacity(0.7)), lineWidth: 2)
            ctx.draw(
                Text("\(m)m").font(.system(size: 9, weight: .medium)).foregroundStyle(color.opacity(0.7)),
                at: CGPoint(x: tp.x + px * 2.2, y: tp.y + py * 2.2)
            )
        }
    }
}
