import AppKit
import CapsuleCore
import SwiftUI

struct SettingsView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Pastas monitoradas")
                .font(.title2.weight(.semibold))

            scanStatus

            if model.roots.isEmpty {
                ContentUnavailableView(
                    "Nenhuma pasta configurada",
                    systemImage: "folder.badge.plus",
                    description: Text("Adicione uma pasta para encontrar cápsulas.")
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(model.roots) { root in
                        HStack(spacing: 10) {
                            Image(systemName: "folder")
                                .foregroundStyle(.secondary)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(root.label)
                                Text(rootState(for: root))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Remover", systemImage: "minus.circle") {
                                model.removeRoot(root)
                            }
                            .labelStyle(.iconOnly)
                            .help("Remover pasta")
                        }
                    }
                }
                .listStyle(.inset)
            }

            HStack {
                Button("Adicionar pasta…", systemImage: "plus", action: addRootPanel)
                Spacer()
                Button("Restaurar padrões") { model.resetRoots() }
            }
        }
        .padding(18)
    }

    @ViewBuilder
    private var scanStatus: some View {
        switch model.scanState {
        case .scanning:
            Label("Escaneando as pastas…", systemImage: "hourglass")
                .foregroundStyle(.secondary)
        case .loaded:
            Label("Leitura concluída", systemImage: "checkmark.circle")
                .foregroundStyle(.secondary)
        case let .failed(message):
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        }
    }

    private func rootState(for root: CapsuleRoot) -> String {
        guard let report = model.scanResult?.roots.first(where: { $0.root.id == root.id }) else {
            return model.isScanning ? "Escaneando…" : "Aguardando leitura"
        }
        switch report.status {
        case .scanned: return "\(report.capsuleCount) cápsulas"
        case .missing: return "Pasta não encontrada"
        case .unavailable: return "Sem acesso à pasta"
        }
    }

    private func addRootPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = "Adicionar"
        panel.begin { response in
            guard response == .OK else { return }
            for url in panel.urls { model.addRoot(url) }
        }
    }
}
