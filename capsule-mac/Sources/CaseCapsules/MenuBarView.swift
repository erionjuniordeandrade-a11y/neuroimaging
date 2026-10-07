import CapsuleCore
import SwiftUI

struct MenuBarView: View {
    @ObservedObject var model: AppModel
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings

    private var threeVersions: ThreeVersions {
        CapsuleLibrary.threeVersions(in: model.capsules)
    }

    var body: some View {
        if model.isScanning {
            Label("Escaneando pastas…", systemImage: "hourglass")
                .disabled(true)
        } else if case let .failed(message) = model.scanState {
            Label("Erro ao escanear", systemImage: "exclamationmark.triangle")
                .help(message)
                .disabled(true)
        } else if model.capsules.isEmpty {
            Text("Nenhuma cápsula encontrada")
                .disabled(true)
        }

        Button("Abrir as 3 versões") { model.openThreeVersions() }
            .keyboardShortcut("3", modifiers: [.command, .option])

        kindButton(.ct)
        kindButton(.mri)
        kindButton(.mriTracts)

        Menu("Recentes") {
            if model.recentCapsules.isEmpty {
                Text("Nenhum arquivo aberto")
                    .disabled(true)
            } else {
                ForEach(model.recentCapsules) { recent in
                    Button(recent.title) { model.openRecent(recent) }
                }
            }
        }

        Divider()
        Button("Biblioteca…") { openWindow(id: "library") }
        Button("Reescanear") { model.requestRescan() }
        Button("Ajustes…") { openSettings() }
        Divider()
        Button("Sair") { NSApplication.shared.terminate(nil) }
    }

    @ViewBuilder
    private func kindButton(_ kind: CapsuleKind) -> some View {
        if let capsule = threeVersions[kind] {
            Button("\(kind.displayName): \(capsule.caseLabel)") { model.open(capsule) }
        } else {
            Button("\(kind.displayName): nenhuma cápsula de \(kind.displayName)") {}
                .disabled(true)
        }
    }
}
