// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "HumanSenseKit",
    platforms: [
        .iOS(.v26),
        .macOS(.v26)
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
            path: "Sources/HumanSenseKit",
            resources: [
                .copy("Resources/FBank.mlmodelc"),
                .copy("Resources/Embedding.mlmodelc")
            ]
        ),
    ]
)
