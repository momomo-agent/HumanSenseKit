# HumanSenseKit

Multi-modal human sensing for iOS using ARKit, Vision, and AVFoundation.

## Features

- **Face tracking**: 52 blend shapes, gaze direction, head orientation, distance
- **Speech detection**: Audio volume, speaking state
- **Hand gestures**: 5 gestures (👍 ✌️ 🖐 ✊ ☝️), left/right hand distinction
- **Head gestures**: Nodding, shaking
- **Gaze tracking**: Screen-space coordinates with smoothing

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/momomo-agent/HumanSenseKit.git", from: "1.0.0")
]
```

## Usage

### Layer 1: Raw Data

```swift
let kit = HumanSenseKit()
kit.start()

// Access raw state
let faceState = kit.state.rawState.face
let audioState = kit.state.rawState.audio
let handState = kit.state.rawState.hand
```

### Layer 2: High-level Queries

```swift
if kit.state.isLookingAtScreen {
    print("User is looking at screen")
}

if kit.state.isSpeakingToDevice {
    print("User is speaking to the device")
}

if kit.state.isNodding {
    print("User is nodding")
}
```

### Layer 3: Semantic/Fluent API

```swift
// Simple condition
kit.observer.whenLookingAtScreen().then {
    print("User started looking at screen")
}

// Combined conditions
kit.observer
    .whenLookingAtScreen()
    .andSpeaking()
    .then {
        print("User is speaking to the device")
    }

// Custom condition
kit.observer
    .when { $0.distanceFromCamera < 0.5 }
    .andLookingAtScreen()
    .onChange { isTrue in
        print("User is close and looking: \(isTrue)")
    }

// Hand gesture
kit.observer.whenHandGesture(.thumbsUp).then {
    print("User gave thumbs up!")
}
```

## Debug UI

```swift
#if DEBUG
struct ContentView: View {
    let kit = HumanSenseKit()
    
    var body: some View {
        kit.debugView()
            .onAppear { kit.start() }
    }
}
#endif
```

## Requirements

- iOS 17.0+
- iPhone X or later (TrueDepth camera required)
- Xcode 15.0+

## License

MIT
