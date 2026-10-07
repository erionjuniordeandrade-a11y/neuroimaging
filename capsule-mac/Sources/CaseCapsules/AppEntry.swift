import AppKit
import CapsuleCore
import Darwin
import Foundation
import SwiftUI

@main
struct CaseCapsulesEntry {
    @MainActor
    static func main() {
        switch AppCommand.parse(Array(CommandLine.arguments.dropFirst())) {
        case .app:
            CaseCapsulesApp.main()
        case .scanSummary:
            runScanSummary()
        case let .selfTest(capsuleURL, outputPath, downloads):
            let application = NSApplication.shared
            application.setActivationPolicy(.prohibited)
            let runner = SelfTestRunner(capsuleURL: capsuleURL, outputPath: outputPath, downloads: downloads)
            runner.start()
            application.run()
        case let .invalid(reason):
            SelfTestOutput.failure(reason)
            exit(1)
        }
    }

    @MainActor
    private static func runScanSummary() {
        do {
            let result = try CapsuleScanner.scan(roots: AppModel.defaultRoots())
            for kind in CapsuleKind.allCases {
                let count = result.capsules.filter { $0.kind == kind }.count
                Swift.print("kind \(kind.rawValue)=\(count)")
            }
            for root in result.roots {
                Swift.print("root \(root.root.label)=\(root.capsuleCount)")
            }
            // The 3-versions pick, masked: root label and mtime only, never a name.
            let picks = CapsuleLibrary.threeVersions(in: result.capsules)
            for kind in [CapsuleKind.ct, .mri, .mriTracts] {
                if let pick = picks[kind] {
                    let stamp = Int(pick.modificationDate?.timeIntervalSince1970 ?? 0)
                    Swift.print("pick \(kind.rawValue) root=\(pick.root.label) mtime=\(stamp) v=\(pick.version ?? 0)")
                } else {
                    Swift.print("pick \(kind.rawValue) none")
                }
            }
        } catch {
            for kind in CapsuleKind.allCases {
                Swift.print("kind \(kind.rawValue)=0")
            }
            Swift.print("scan unavailable")
            exit(1)
        }
    }
}

@MainActor
struct CaseCapsulesApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        Window("Biblioteca", id: "library") {
            LibraryView(model: model)
                .frame(minWidth: 900, minHeight: 560)
        }
        .defaultSize(width: 1180, height: 760)

        MenuBarExtra("Case Capsules", systemImage: "cube.transparent") {
            MenuBarView(model: model)
        }

        Settings {
            SettingsView(model: model)
                .frame(width: 620, height: 430)
        }
    }
}

private enum AppCommand {
    case app
    case scanSummary
    case selfTest(URL, String, Bool)
    case invalid(String)

    static func parse(_ arguments: [String]) -> AppCommand {
        guard let first = arguments.first else { return .app }
        if first == "--scan-summary" {
            return arguments.count == 1 ? .scanSummary : .invalid("invalid scan-summary arguments")
        }
        guard first == "--selftest" else { return .invalid("unknown command") }
        guard (arguments.count == 4 || arguments.count == 5),
              arguments[2] == "--out",
              arguments.count == 4 || arguments[4] == "--selftest-download" else {
            return .invalid("usage: --selftest <capsule.html> --out <png> [--selftest-download]")
        }

        return .selfTest(
            URL(fileURLWithPath: arguments[1]),
            arguments[3],
            arguments.count == 5
        )
    }
}
