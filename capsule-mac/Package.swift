// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "CaseCapsules",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "CapsuleCore", targets: ["CapsuleCore"]),
        .executable(name: "CaseCapsules", targets: ["CaseCapsules"])
    ],
    targets: [
        .target(name: "CapsuleCore"),
        .executableTarget(name: "CaseCapsules", dependencies: ["CapsuleCore"]),
        .testTarget(name: "CapsuleCoreTests", dependencies: ["CapsuleCore"])
    ],
    swiftLanguageVersions: [.v5]
)
