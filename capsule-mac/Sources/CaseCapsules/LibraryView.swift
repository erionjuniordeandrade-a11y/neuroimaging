import AppKit
import CapsuleCore
import SwiftUI
import UniformTypeIdentifiers

struct LibraryView: View {
    @ObservedObject var model: AppModel
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        NavigationSplitView {
            List(selection: $model.selectedSection) {
                Section("Biblioteca") {
                    ForEach(LibrarySection.allCases) { section in
                        Label(section.title, systemImage: icon(for: section))
                            .tag(section)
                    }
                }
            }
            .navigationSplitViewColumnWidth(min: 190, ideal: 220)
        } detail: {
            detail
                .toolbar {
                    ToolbarItemGroup {
                        Button("Reescanear", systemImage: "arrow.clockwise") {
                            model.requestRescan()
                        }
                        Button("Abrir arquivo…", systemImage: "doc.badge.plus", action: openFilePanel)
                        Button("Mostrar no Finder", systemImage: "folder") {
                            if let capsule = model.selectedCapsule {
                                NSWorkspace.shared.activateFileViewerSelecting([capsule.url])
                            }
                        }
                        .disabled(model.selectedCapsule == nil)
                        Button("Abrir", systemImage: "arrow.up.right.square") {
                            model.openSelectedCapsule()
                        }
                        .disabled(model.selectedCapsule == nil)
                    }
                    ToolbarItem(placement: .automatic) {
                        TextField("Buscar por rótulo ou arquivo", text: $model.searchText)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 250)
                    }
                }
        }
        .navigationTitle("Biblioteca")
        .alert(item: $model.notice) { notice in
            Alert(
                title: Text(notice.title),
                message: Text(notice.message),
                dismissButton: .default(Text("OK"))
            )
        }
    }

    @ViewBuilder
    private var detail: some View {
        VStack(spacing: 0) {
            if model.isScanning {
                ProgressView("Escaneando pastas…")
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)
                    .padding(.vertical, 8)
            }

            rootStatus

            if case let .failed(message) = model.scanState {
                errorState(message)
            } else if model.scanResult == nil {
                ProgressView("Preparando a biblioteca…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if model.capsules.isEmpty {
                emptyLibrary
            } else {
                let families = model.families(for: model.selectedSection)
                if families.isEmpty {
                    emptySection
                } else {
                    List(selection: $model.selectedCapsuleID) {
                        ForEach(families) { family in
                            if family.older.isEmpty {
                                CapsuleRow(capsule: family.newest, model: model)
                                    .tag(family.newest.id)
                            } else {
                                DisclosureGroup {
                                    ForEach(family.older) { capsule in
                                        CapsuleRow(capsule: capsule, model: model)
                                            .tag(capsule.id)
                                    }
                                } label: {
                                    CapsuleRow(capsule: family.newest, model: model)
                                        .tag(family.newest.id)
                                }
                            }
                        }
                    }
                    .listStyle(.inset)
                    .onSubmit { model.openSelectedCapsule() }
                }
            }
        }
    }

    @ViewBuilder
    private var rootStatus: some View {
        if let result = model.scanResult {
            let missing = result.roots.filter { $0.status == .missing }
            let unavailable = result.roots.filter { $0.status == .unavailable }
            if !missing.isEmpty || !unavailable.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(missing) { report in
                        Label("Pasta não encontrada: \(report.root.label)", systemImage: "folder.badge.questionmark")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(unavailable) { report in
                        Label("Sem acesso à pasta: \(report.root.label)", systemImage: "exclamationmark.folder")
                            .foregroundStyle(.orange)
                    }
                }
                .font(.caption)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
        }
    }

    private var emptyLibrary: some View {
        VStack(spacing: 12) {
            Image(systemName: "shippingbox")
                .font(.system(size: 34))
                .foregroundStyle(.secondary)
            Text("Nenhuma cápsula encontrada")
                .font(.title3.weight(.semibold))
            Text("Confira as pastas monitoradas e adicione ou restaure pastas em Ajustes.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
            Button("Abrir Ajustes…", action: openSettings.callAsFunction)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptySection: some View {
        VStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 30))
                .foregroundStyle(.secondary)
            Text(model.searchText.isEmpty ? "Nenhuma cápsula nesta seção" : "Nenhum resultado")
                .font(.title3.weight(.semibold))
            Text(model.searchText.isEmpty
                 ? "Nenhuma cápsula de \(model.selectedSection.title) está disponível."
                 : "Tente outro rótulo ou nome de arquivo.")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func errorState(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 30))
                .foregroundStyle(.orange)
            Text("Não foi possível ler as pastas")
                .font(.title3.weight(.semibold))
            Text(message)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Tentar novamente") { model.requestRescan() }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func openFilePanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.html]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            model.openFile(at: url)
        }
    }

    private func icon(for section: LibrarySection) -> String {
        switch section {
        case .threeVersions: "square.stack.3d.up"
        case .ct: "viewfinder"
        case .mri: "circle.dotted"
        case .mriTracts: "point.3.connected.trianglepath.dotted"
        case .all: "square.grid.2x2"
        case .drafts: "pencil.and.outline"
        }
    }
}

private struct CapsuleRow: View {
    let capsule: CapsuleInfo
    @ObservedObject var model: AppModel

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 7) {
                    Text(capsule.caseLabel)
                        .font(.headline)
                        .lineLimit(1)
                    Text(capsule.kind.displayName)
                        .font(.caption)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(.quaternary, in: Capsule())
                    Text(capsule.root.label)
                        .font(.caption)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(.quaternary, in: Capsule())
                }
                Text(capsule.filename)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                if let manifest = capsule.manifest {
                    HStack(spacing: 10) {
                        Text(capsule.modificationDate?.formatted(date: .abbreviated, time: .shortened) ?? "Data desconhecida")
                        Text(ByteCountFormatter.string(fromByteCount: capsule.sizeBytes ?? 0, countStyle: .file))
                        Text("v\(manifest.version)")
                        Text("\(manifest.maskCount) máscaras · \(manifest.tractCount) tratos · \(manifest.tourStepCount) passos do tour")
                            .lineLimit(1)
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                } else if case let .unreadable(issue) = capsule.contents {
                    Text(issue.message)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            Spacer(minLength: 8)
            Button("Abrir") { model.open(capsule) }
                .disabled(capsule.manifest == nil)
        }
        .padding(.vertical, 5)
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { model.open(capsule) }
    }
}
