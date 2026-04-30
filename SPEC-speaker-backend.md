# Speaker Attribution Backend 抽象 + VM 大瘦身 SPEC

**目标**：Kit 提供统一的"谁在说话"接口，宿主 app 只见接口不见后端。VoiceViewModel 完全照搬 demo 第四 tab 的简洁哲学，删除所有 reconstructor / segment-level gate / 时序补丁。

## 背景

- 当前 Kit 同时跑两套归因：旧 `UserSentenceReconstructor` (u+ Schmitt-trigger) + 新 `GazeSpeakerEngine` (jaw+gaze+speaker embedding)
- VM (`SmileOS/ViewModels/VoiceViewModel.swift`) 依赖**旧 u+**：订阅 `userSentenceReconstructor.$userSentence`、`sttManager.$segments`，在外面叠 attention gate / merge queue / short-word defer / TTS tail suppression / userSentence.isEmpty 二次检查
- demo 第四 tab `DiarizationTestView` 用新 GazeSpeakerEngine，完全没有这些补丁——直接读 `transcriptSegments.filter { $0.isUserSpeaker }` 和 `currentTokens.filter { $0.isUserSpeaker }`
- kenefe 决定：完全切到新引擎的哲学，干干净净，时序状态明明白白

## 阶段 1：Kit 协议层 (`SpeakerAttributionBackend`)

### 1.1 新增类型 `Sources/HumanSenseKit/Layer1/SpeakerAttribution.swift`

```swift
import Foundation
import Combine

public enum SpeakerSource: String, Sendable, Codable {
    case user       // 本机用户（embedding/jaw/gaze 综合判定）
    case ambient    // 环境人声（不是用户，不是 TTS）
    case tts        // AI 自己的声音回漏（TTS 窗口内）
    case unknown    // 标定未完成 / 数据不足
}

public struct SpeakerAttributedToken: Identifiable, Sendable {
    public let id: UUID
    public let text: String
    public let audioTime: Double      // SpeechToken.startTime
    public let endTime: Double        // SpeechToken.endTime
    public let source: SpeakerSource
    public let score: Float           // 0-1 置信度
    public let metadata: SpeakerTokenMetadata
    
    public init(id: UUID = UUID(), text: String, audioTime: Double, endTime: Double,
                source: SpeakerSource, score: Float, metadata: SpeakerTokenMetadata = .init()) { ... }
}

public struct SpeakerTokenMetadata: Sendable {
    public var jawDelta: Float = 0
    public var jawVelocity: Float = 0
    public var gazeOnScreen: Float = 0
    public var headYaw: Float = 0
    public var headPitch: Float = 0
    public var faceDistance: Float = 0
    public var speakerSimilarity: Float = 0    // 余弦相似度 to user embedding
    public init() {}
}

public struct SpeakerAttributedSegment: Identifiable, Sendable {
    public let id: UUID
    public let tokens: [SpeakerAttributedToken]
    public let isFinal: Bool
    public let timestamp: Date
    
    public var text: String { tokens.map(\.text).joined() }
    public var userText: String { tokens.filter { $0.source == .user }.map(\.text).joined() }
    public var hasUserSpeech: Bool { tokens.contains { $0.source == .user } }
}

public enum SpeakerEnginePhase: Sendable, Equatable {
    case unconfigured                                                      // 没标定
    case calibrating(progress: Float, currentSentence: String?, sentenceIndex: Int, totalSentences: Int)
    case ready                                                             // 可以 live 识别
}

public enum SpeakerEvent: Sendable {
    case phaseChanged(SpeakerEnginePhase)
    case streamingTokens([SpeakerAttributedToken])    // volatile 更新
    case finalSegment(SpeakerAttributedSegment)       // 一段 final
    case userSpeech(text: String, segment: SpeakerAttributedSegment)
    // ↑ final 且 userText.isEmpty == false 时同时发出，方便消费方
}

@MainActor
public protocol SpeakerAttributionBackend: AnyObject {
    // === 状态 (Observable) ===
    var phase: SpeakerEnginePhase { get }
    var transcriptSegments: [SpeakerAttributedSegment] { get }
    var currentTokens: [SpeakerAttributedToken] { get }
    var debugInfo: SpeakerDebugInfo { get }
    
    // === 单一事件流 ===
    var events: AnyPublisher<SpeakerEvent, Never> { get }
    
    // === Calibration ===
    var calibrationSentences: [String] { get }
    func startCalibration()
    func cancelCalibration()
    func resetCalibration()
    
    // === TTS 窗口（见阶段 3）===
    func markTTSStart(expectedText: String?)
    func markTTSEnd()
    
    // === 控制 ===
    func reset()  // 清 transcriptSegments + currentTokens（切场景时）
}

public struct SpeakerDebugInfo: Sendable {
    public var userEmbeddingStatus: String = "未标定"
    public var embeddingCount: Int = 0
    public var ttsWindowActive: Bool = false
    public var lastUserSimilarity: Float = 0
    public var lastJawDelta: Float = 0
    public init() {}
}
```

### 1.2 改造 `GazeSpeakerEngine` 实现 `SpeakerAttributionBackend`

- 重命名外部类型 `TokenSegment` → 内部用 `SpeakerAttributedToken`（旧名保留为 `typealias` 兼容旧调用方一段时间）
- `Phase` enum → 重新映射成 `SpeakerEnginePhase`
- 新增 `events: PassthroughSubject<SpeakerEvent, Never>`，在 `setupSTTListener` 里 emit
- 现有 `onUserSpeech: ((String) -> Void)?` 保留作 deprecated，事件流是新接口
- 加 `markTTSStart(expectedText:)` / `markTTSEnd()` 实现（见阶段 3）
- 暴露 `calibrationSentences` 已经有

### 1.3 顶层 `HumanSenseKit` 暴露 backend

```swift
public class HumanSenseKit {
    // 已有...
    public let speaker: SpeakerAttributionBackend
    
    public init(
        enableHandGestures: Bool = false,
        enableSTT: Bool = true,
        speakerBackend: SpeakerAttributionBackend? = nil   // 注入点
    ) {
        // 已有 init 逻辑
        self.speaker = speakerBackend ?? GazeSpeakerEngine(engine: engine)
    }
}
```

宿主 app 默认拿到 `GazeSpeakerEngine`，未来可注入其他后端。

## 阶段 2：VoiceViewModel 大瘦身

**文件**: `SmileOS/ViewModels/VoiceViewModel.swift`（当前 1275 行）

### 2.1 删除清单（不能再有任何引用）

| 字段/方法 | 行号区间（粗略） |
|----------|----------------|
| `reconstructorCancellable` | ~40 + 264-272 |
| `segmentsCancellable` 改为只用于 ambient logging（如保留） | 232-249 |
| `checkForNewSpeech()` 整个方法 | 392-610 |
| `commitCandidate()` 整个方法 | 612-720 |
| `publishLivePartial()` 整个方法 | 282-340 |
| `trackSpeechRhythm()` + 所有 rhythm 状态 | ~340-390 + `speechPauseDetected`/`volatileCharCounts`/`speechSpeed` |
| `candidateSegmentId`/`candidateText`/`candidateTimer`/`lastProcessedSegmentId` | 状态字段 |
| `pendingMergeTexts`/`mergeTimer`/`flushMergedInterrupt`/`isWaiting` 合并相关 | merge queue |
| `ambientFilterStrict` setting | 不再有 gate |
| `livePartialDisplayEnabled` setting | 总是显示，不需要开关 |
| `ttsTailSuppressionSeconds`/`lastTTSActiveTime` | 改用 Kit `markTTSStart/End` |
| `loggedAmbientSegmentIds` | ambient 走 `events` 自然过滤 |
| `mergeWindow`/相关 timer 常量 | merge queue 删 |

**预期**：1275 行 → ~600 行

### 2.2 新订阅（只这一个）

```swift
private var speakerEventCancellable: AnyCancellable?

private func subscribeToSpeaker() {
    speakerEventCancellable = humanSense.kit.speaker.events
        .receive(on: DispatchQueue.main)
        .sink { [weak self] event in
            guard let self else { return }
            switch event {
            case .streamingTokens(let tokens):
                self.handleStreamingTokens(tokens)
            case .finalSegment(let segment):
                self.handleFinalSegment(segment)
            case .userSpeech(let text, let segment):
                self.handleUserSpeech(text: text, segment: segment)
            case .phaseChanged(let phase):
                self.handleSpeakerPhase(phase)
            }
        }
}

private func handleStreamingTokens(_ tokens: [SpeakerAttributedToken]) {
    // UI streaming：只取 user 的拼接
    let userText = tokens.filter { $0.source == .user }.map(\.text).joined()
    if userText != livePartialText {
        livePartialText = userText
    }
    // 业务态：streaming 开始时进 listening
    if !userText.isEmpty && conversationState == .idle {
        conversationState = .listening
        // 抓帧时机：第一次有 user token 出现
        if capturedCameraImage == nil {
            capturedCameraImage = humanSense.captureCurrentFrame()
            capturedScreenImage = screenshotProvider?()
        }
    }
}

private func handleFinalSegment(_ segment: SpeakerAttributedSegment) {
    // segment 自然进了 transcriptSegments，UI 直接读，不用 VM 维护
    livePartialText = ""    // streaming 结束
}

private func handleUserSpeech(text: String, segment: SpeakerAttributedSegment) {
    // ★ 唯一的 LLM 触发点
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    
    // TTS barge-in：如果 AI 还在说，停掉
    if ttsService.isSpeaking {
        if partialResponseOnInterrupt.isEmpty && !voiceText.isEmpty {
            partialResponseOnInterrupt = String(voiceText.prefix(600))
        }
        ttsService.stop()
    }
    
    timeline.mark("⬆️")
    processUserSpeech(trimmed)    // 已存在，进 ChatQueue
    capturedCameraImage = nil     // 重置下次抓帧
    capturedScreenImage = nil
}

private func handleSpeakerPhase(_ phase: SpeakerEnginePhase) {
    speakerPhase = phase    // 暴露给 UI 层，用于 calibration 引导
}
```

### 2.3 完全删除的逻辑（明确确认）

- ✗ short-word defer 500ms：Kit 的 `isFinal` 已经是真 final（Apple SpeechTranscriber 的 final 事件 + GazeSpeakerEngine 的 segment 边界判定共同保证）
- ✗ stability timer 1s：同上
- ✗ merge queue / barge-in 合并：每次 `userSpeech` 事件就是独立一轮，由 ChatQueue 处理串行
- ✗ attention gate `!looking || !headFwd`：Kit 已用 gaze 在 token 级判定
- ✗ final-stage userSentence.isEmpty 检查：`userSpeech` 事件本身就保证 userText 非空
- ✗ TTS tail suppression 500ms：改成 Kit 的 `markTTSEnd()` + 内部 200ms 抖动窗口

### 2.4 保留

- `chatQueue`/`processUserSpeech`/`processUserMessage`/LLM pipeline
- `ttsService`/`capturedCameraImage`/`capturedScreenImage`/`screenshotProvider`
- `conversationState` enum
- `partialResponseOnInterrupt` 逻辑（TTS 打断时记录）
- `MemoryStore.appendLog`：但 `.user` 角色在 `handleUserSpeech` 里写，`.ambient` 改成订阅 `events` 里 final segment 中 `tokens.filter { $0.source == .ambient }`

## 阶段 3：TTS 窗口集成

### 3.1 GazeSpeakerEngine 内部状态

```swift
private var ttsWindowActive: Bool = false
private var ttsWindowEndedAt: Date? = nil
private let ttsTailGraceSeconds: TimeInterval = 0.2     // AEC 延迟消抖
private var ttsExpectedText: String? = nil

public func markTTSStart(expectedText: String?) {
    ttsWindowActive = true
    ttsExpectedText = expectedText
    ttsWindowEndedAt = nil
    debugInfo.ttsWindowActive = true
}

public func markTTSEnd() {
    ttsWindowActive = false
    ttsWindowEndedAt = Date()
    // expectedText 保留 ttsTailGraceSeconds 后清空
}
```

### 3.2 在 `processTokens` 内的判定增强

每个 token 经 `attributor.processTokens` 拿到原始 `isUserSpeaker`，再经过 TTS 窗口判定：

```swift
private func resolveSource(for token: SpeakerToken, baseIsUser: Bool, similarity: Float) -> SpeakerSource {
    let inWindow = ttsWindowActive ||
        (ttsWindowEndedAt.map { Date().timeIntervalSince($0) < ttsTailGraceSeconds } ?? false)
    
    if !inWindow {
        return baseIsUser ? .user : .ambient
    }
    
    // TTS 窗口内：拉高用户判定阈值，真 barge-in 才认
    if baseIsUser && similarity > strongUserSimilarity {   // 比正常阈值高 0.1
        return .user
    }
    
    // expectedText 文本对比兜底
    if let expected = ttsExpectedText, !expected.isEmpty {
        if textSimilarity(token.text, contains: expected) > 0.7 {
            return .tts
        }
    }
    
    return .tts
}
```

### 3.3 VM 侧调用

```swift
// TTSService 监听
ttsService.onPlaybackStart = { [weak self] expectedText in
    self?.humanSense.kit.speaker.markTTSStart(expectedText: expectedText)
}
ttsService.onPlaybackEnd = { [weak self] in
    self?.humanSense.kit.speaker.markTTSEnd()
}
```

如果 `TTSService` 现在没有 onPlaybackStart/End 钩子，**新增**这两个 callback。检查 `SmileOS/Services/TTSService.swift` 现有结构，最小改动加。

## 阶段 4：Calibration UI

### 4.1 SmileOS 启动时检测 phase

在 SmileOS app 启动流程（`SmileOSApp.swift` 或主 ContentView）：

```swift
@State private var speakerPhase: SpeakerEnginePhase = .unconfigured

.onAppear {
    speakerPhase = humanSenseService.kit.speaker.phase
}

.onReceive(humanSenseService.kit.speaker.events) { event in
    if case .phaseChanged(let phase) = event {
        speakerPhase = phase
    }
}

ZStack {
    if case .ready = speakerPhase {
        VoiceMainView()    // 现有主界面
    } else {
        CalibrationView(speaker: humanSenseService.kit.speaker)
    }
}
```

### 4.2 `CalibrationView`（新文件 `SmileOS/Views/CalibrationView.swift`）

参考 demo `DiarizationTestView.calibrationView`（见 `human-sense-demo/Sources/Views/DiarizationTestView.swift:62-128`），但用 SmileOS 设计语言：

- 黑色背景
- 标题"声纹标定 / 让我学会你的声音"
- 当前要念的句子（大字）
- 进度条 + "N/M"
- "开始"按钮 / 标定中显示状态
- 标定完成后自动 transition（GazeSpeakerEngine 内部已经会切到 `.ready`）

UI 极简，先能跑通；视觉精修留 kenefe 后续。

## 阶段 5：DebugOverlayView 适配

`SmileOS/DebugOverlayView.swift` 里：
- Tokens tab 已经在用 `speakerEngine.transcriptSegments` / `currentTokens`，**保持不变**
- 其他 tab 如果有引用 `reconstructor.userSentence` 之类，全部改成读 `kit.speaker.transcriptSegments`/`currentTokens`
- 显示 `source: .tts` 的 token 用灰色或带 ⚠️ 图标，方便调试

## 阶段 6：测试

### 6.1 编译

```bash
cd ~/Dropbox/CODE/Agents/HumanSenseKit
swift build --target HumanSenseKit
```

### 6.2 装机回归测试（依次验证）

1. **首次启动**：SmileOS 进 calibration 流程，念 N 句话，看到 phase 切到 `.ready`
2. **基础识别**：说一句话 → 灰色 streaming 立即出 → final 后 LLM 触发
3. **TTS 防回漏**：触发 LLM 让 AI 说一段长话，AI 说话期间不要发新一轮 LLM（embedding + TTS 窗口双层防护）
4. **Barge-in**：AI 说话时打断它，验证：
   - `ttsService.stop()` 被调
   - 新一轮 user 触发
   - `partialResponseOnInterrupt` 有内容
5. **切前后台**：切桌面 3 秒回来说话，jaw/V 不卡（4.9.85 + 4.9.89 修复保留）
6. **环境人声**：找个第二个人说话，validate `source = .ambient`，不触发 LLM

### 6.3 性能

- ARFrame retaining 警告应该只在启动 race 那 33 秒内（已修，不应回归）
- speaker.events 频率应该跟 token 频率一致（约 5-10 Hz），不应该爆 main thread

## 边界条件

- **空 expectedText**：TTS 窗口内 expectedText 为 nil 时只用 similarity 判定
- **Calibration 失败**：用户标定到一半 background app，重启应当能继续（`GazeSpeakerEngine` 检查 `userEmbeddings.isEmpty` 决定 phase）
- **多人同时说话**：TTS 窗口外，`isUserSpeaker` 由 embedding 决定，跟用户最像的认作 user，其他认作 ambient
- **未标定状态收到 token**：phase = unconfigured，所有 token source = unknown，VM 收到 streamingTokens 但 userText 永远空，不触发 LLM

## 不允许做的事

1. ✗ 不要为了"最小改动"保留 reconstructor 的 VM 订阅"作为 fallback"——彻底清理
2. ✗ 不要重新引入任何 attention gate / TTS suppression / merge queue
3. ✗ 不要修改 `GazeSpeakerAttributor.swift` 内部算法（embedding cosine、判定阈值），这是另一个研究项目
4. ✗ 不要删除旧 `UserSentenceReconstructor`——demo TokenTableView 还在用，保留
5. ✗ 不要修改 `human-sense-demo` 项目，只动 HumanSenseKit + SmileOS

## 交付清单

完成后必须有：

1. [ ] HumanSenseKit `swift build` 通过
2. [ ] HumanSenseKit 新 tag (4.9.90)，push GitHub
3. [ ] visual-talk-ios `project.yml` bump 到 4.9.90，push
4. [ ] visual-talk-ios xcodebuild SmileOS scheme `BUILD SUCCEEDED`
5. [ ] VoiceViewModel.swift 不再 import / 引用 `userSentenceReconstructor`、`reconstructor`
6. [ ] grep 验证：`grep -n "userSentenceReconstructor\|reconstructor\|publishLivePartial\|checkForNewSpeech\|commitCandidate\|pendingMergeTexts\|mergeTimer\|ambientFilterStrict\|livePartialDisplayEnabled\|ttsTailSuppressionSeconds" SmileOS/ViewModels/VoiceViewModel.swift` 全空
7. [ ] CalibrationView.swift 新文件存在
8. [ ] SmileOSApp / 主 ContentView 有 phase 路由
9. [ ] TTSService 有 onPlaybackStart/End hook 接到 `markTTSStart/End`
10. [ ] 提交两个 PR-style commit（一个 Kit，一个 SmileOS），message 含 SPEC 链接
