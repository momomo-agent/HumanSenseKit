// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "HumanSenseKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)  // For Combine support, but ARKit only works on iOS
    ],
    products: [
        .library(
            name: "HumanSenseKit",
            targets: ["HumanSenseKit"]
        ),
    ],
    targets: [
        .target(
            name: "HumanSenseKit",
            path: "Sources/HumanSenseKit"
        ),
    ]
)
