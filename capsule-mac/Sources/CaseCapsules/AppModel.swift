import AppKit
import CapsuleCore
import Foundation
import SwiftUI
import WebKit

enum LibrarySection: String, CaseIterable, Hashable, Identifiable {
    case threeVersions
    case ct
    case mri
    case mriTracts
    case all
    case drafts

    var id: String { rawValue }

    var title: String {
        switch self {
        case .threeVersions: "As 3 versões"
        case .ct: CapsuleKind.ct.displayName
        case .mri: CapsuleKind.mri.displayName
        case .mriTracts: CapsuleKind.mriTracts.displayName
        case .all: "Todas"
        case .drafts: "Rascunhos"
        }
    }

    var scope: LibraryScope {
        switch self {
        case .threeVersions: .threeVersions
        case .ct: .kind(.ct)
        case .mri: .kind(.mri)
        case .mriTracts: .kind(.mriTracts)
        case .all: .all
        case .drafts: .drafts
        }
    }
}

enum LibraryScanState {
    case scanning(previous: ScanResult?)
    case loaded(ScanResult)
    case failed(String)

    var result: ScanResult? {
        switch self {
        case let .scanning(previous): previous
        case let .loaded(result): result
        case .failed: nil
        }
    }

    var isScanning: Bool {
        if case .scanning = self { return true }
        return false
    }
}

enum OfflineRuleState {
    case compiling
    case ready(WKContentRuleList)
    case failed(String)
}

struct AppNotice: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

struct RecentCapsule: Identifiable {
    let path: String
    let title: String

    var id: String { path }
}

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var scanState: LibraryScanState = .scanning(previous: nil)
    @Published private(set) var ruleState: OfflineRuleState = .compiling
    @Published private(set) var roots: [CapsuleRoot]
    @Published var searchText = ""
    @Published var selectedSection: LibrarySection = .threeVersions
    @Published var selectedCapsuleID: String?
    @Published var notice: AppNotice?

    private let defaults: UserDefaults
    private let viewerWindows = ViewerWindowManager()
    private var scanTask: Task<Void, Never>?
    private var scanTimeoutTask: Task<Void, Never>?
    private var scanGeneration = UUID()
    private var rescanQueued = false
    private var openThreeAfterScan = false

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.roots = Self.loadRoots(from: defaults)
        viewerWindows.onDownloadFinished = { [weak self] in
            self?.requestRescan()
        }
        compileOfflineRules()
        rescan()
    }

    var scanResult: ScanResult? { scanState.result }
    var isScanning: Bool { scanState.isScanning }
    var capsules: [CapsuleInfo] { scanResult?.capsules ?? [] }

    var selectedCapsule: CapsuleInfo? {
        guard let selectedCapsuleID else { return nil }
        return capsules.first { $0.id == selectedCapsuleID }
    }

    var recentCapsules: [RecentCapsule] {
        let paths = defaults.stringArray(forKey: Self.recentPathsKey) ?? []
        return paths.prefix(8).map { path in
            let match = capsules.first { $0.url.standardizedFileURL.path == path }
            return RecentCapsule(
                path: path,
                title: match?.caseLabel ?? URL(fileURLWithPath: path).lastPathComponent
            )
        }
    }

    func families(for section: LibrarySection) -> [CapsuleFamily] {
        CapsuleLibrary.families(in: capsules, scope: section.scope, searchText: searchText)
    }

    func rescan() {
        guard scanTask == nil else {
            rescanQueued = true
            return
        }

        let generation = UUID()
        scanGeneration = generation
        let rootsToScan = roots
        let previous = scanResult
        scanState = .scanning(previous: previous)
        scanTimeoutTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 120_000_000_000)
            guard !Task.isCancelled else { return }
            self?.finishTimedOutScan(generation: generation)
        }
        scanTask = Task.detached(priority: .utility) { [weak self] in
            do {
                let result = try CapsuleScanner.scan(
                    roots: rootsToScan,
                    isCancelled: { Task.isCancelled }
                )
                await self?.finishScan(generation: generation, result: result)
            } catch is CancellationError {
                await self?.finishCancelledScan(generation: generation)
            } catch {
                await self?.finishFailedScan(generation: generation)
            }
        }
    }

    func requestRescan() {
        if scanTask != nil {
            rescanQueued = true
        } else {
            rescan()
        }
    }

    func replaceRoots(_ roots: [CapsuleRoot]) {
        var seen: Set<String> = []
        self.roots = roots.filter { seen.insert($0.id).inserted }
        persistRoots()

        scanTask?.cancel()
        scanTimeoutTask?.cancel()
        scanTimeoutTask = nil
        scanTask = nil
        rescanQueued = false
        rescan()
    }

    func addRoot(_ url: URL) {
        let root = CapsuleRoot(url: url, label: url.lastPathComponent)
        guard !roots.contains(where: { $0.id == root.id }) else { return }
        replaceRoots(roots + [root])
    }

    func removeRoot(_ root: CapsuleRoot) {
        replaceRoots(roots.filter { $0.id != root.id })
    }

    func resetRoots() {
        replaceRoots(Self.defaultRoots())
    }

    func open(_ capsule: CapsuleInfo) {
        guard case .readable = capsule.contents else {
            let issue: ManifestIssue
            if case let .unreadable(reason) = capsule.contents { issue = reason }
            else { issue = .invalidShape }
            notice = AppNotice(title: "Cápsula indisponível", message: issue.message)
            return
        }

        guard FileManager.default.fileExists(atPath: capsule.url.path) else {
            notice = AppNotice(title: "Arquivo indisponível", message: "O arquivo não foi encontrado.")
            return
        }

        guard case let .ready(ruleList) = ruleState else {
            switch ruleState {
            case .compiling:
                notice = AppNotice(title: "Preparando o visualizador", message: "A proteção offline ainda está sendo preparada.")
            case let .failed(message):
                notice = AppNotice(title: "Visualizador indisponível", message: message)
            case .ready:
                break
            }
            return
        }

        viewerWindows.open(
            capsule: capsule,
            ruleList: ruleList,
            onOpened: { [weak self] in self?.recordRecent(capsule.url) }
        )
    }

    func openSelectedCapsule() {
        guard let selectedCapsule else { return }
        open(selectedCapsule)
    }

    func openThreeVersions() {
        guard let result = scanResult else {
            openThreeAfterScan = true
            requestRescan()
            return
        }

        let selected = CapsuleLibrary.threeVersions(in: result.capsules)
        let available = CapsuleKind.threeVersionOrder.compactMap { selected[$0] }
        guard !available.isEmpty else {
            notice = AppNotice(title: "As 3 versões", message: "Nenhuma cápsula legível e atual está disponível.")
            return
        }

        guard case let .ready(ruleList) = ruleState else {
            open(available[0])
            return
        }

        let missing = CapsuleKind.threeVersionOrder.filter { selected[$0] == nil }
        viewerWindows.openAsTabs(
            capsules: available,
            ruleList: ruleList,
            onOpened: { [weak self] capsule in self?.recordRecent(capsule.url) }
        )
        if let firstMissing = missing.first {
            notice = AppNotice(
                title: "As 3 versões",
                message: "Nenhuma cápsula de \(firstMissing.displayName) está disponível."
            )
        }
    }

    func openRecent(_ recent: RecentCapsule) {
        if let capsule = capsules.first(where: { $0.url.standardizedFileURL.path == recent.path }) {
            open(capsule)
        } else {
            openFile(at: URL(fileURLWithPath: recent.path))
        }
    }

    func openFile(at url: URL) {
        let fileURL = url.standardizedFileURL
        let root = CapsuleRoot(url: fileURL.deletingLastPathComponent(), label: "arquivo")
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let manifest = try ManifestReader.read(from: fileURL)
                let values = try? fileURL.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
                let capsule = CapsuleInfo(
                    url: fileURL,
                    root: root,
                    modificationDate: values?.contentModificationDate,
                    sizeBytes: values?.fileSize.map(Int64.init),
                    contents: .readable(manifest)
                )
                await self?.open(capsule)
            } catch let issue as ManifestIssue {
                await self?.showUnreadableFile(issue)
            } catch {
                await self?.showUnreadableFile(.fileUnavailable)
            }
        }
    }

    func present(_ notice: AppNotice) {
        self.notice = notice
    }

    private func finishScan(generation: UUID, result: ScanResult) {
        guard generation == scanGeneration else { return }
        scanTask = nil
        scanTimeoutTask?.cancel()
        scanTimeoutTask = nil
        scanState = .loaded(result)

        if rescanQueued {
            rescanQueued = false
            rescan()
        } else if openThreeAfterScan {
            openThreeAfterScan = false
            openThreeVersions()
        }
    }

    private func finishCancelledScan(generation: UUID) {
        guard generation == scanGeneration else { return }
        scanTask = nil
        scanTimeoutTask?.cancel()
        scanTimeoutTask = nil
        if rescanQueued {
            rescanQueued = false
            rescan()
        } else {
            scanState = scanResult.map(LibraryScanState.loaded) ?? .failed("A leitura foi cancelada.")
        }
    }

    private func finishFailedScan(generation: UUID) {
        guard generation == scanGeneration else { return }
        scanTask = nil
        scanTimeoutTask?.cancel()
        scanTimeoutTask = nil
        scanState = .failed("Não foi possível concluir a leitura das pastas.")
    }

    private func finishTimedOutScan(generation: UUID) {
        guard generation == scanGeneration else { return }
        scanTask?.cancel()
        scanTask = nil
        scanTimeoutTask = nil
        scanState = .failed("A leitura das pastas excedeu o tempo limite.")
    }

    private func showUnreadableFile(_ issue: ManifestIssue) {
        notice = AppNotice(title: "Arquivo indisponível", message: issue.message)
    }

    private func compileOfflineRules() {
        OfflineRuleCompiler.compile { [weak self] result in
            switch result {
            case let .success(ruleList): self?.ruleState = .ready(ruleList)
            case .failure: self?.ruleState = .failed("Não foi possível ativar o bloqueio de rede.")
            }
        }
    }

    private func recordRecent(_ url: URL) {
        var paths = defaults.stringArray(forKey: Self.recentPathsKey) ?? []
        let path = url.standardizedFileURL.path
        paths.removeAll { $0 == path }
        paths.insert(path, at: 0)
        defaults.set(Array(paths.prefix(8)), forKey: Self.recentPathsKey)
    }

    private func persistRoots() {
        let stored = roots.map { StoredRoot(path: $0.url.path, label: $0.label) }
        if let data = try? JSONEncoder().encode(stored) {
            defaults.set(data, forKey: Self.rootsKey)
        }
    }

    private static let rootsKey = "caseCapsules.roots.v1"
    private static let recentPathsKey = "caseCapsules.recentPaths.v1"

    private static func loadRoots(from defaults: UserDefaults) -> [CapsuleRoot] {
        guard let data = defaults.data(forKey: rootsKey),
              let stored = try? JSONDecoder().decode([StoredRoot].self, from: data) else {
            return defaultRoots()
        }
        return stored.map { CapsuleRoot(url: URL(fileURLWithPath: $0.path), label: $0.label) }
    }

    static func defaultRoots() -> [CapsuleRoot] {
        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        return [
            CapsuleRoot(url: home.appendingPathComponent("case-capsule/out", isDirectory: true), label: "público"),
            CapsuleRoot(url: home.appendingPathComponent("case-capsule-private", isDirectory: true), label: "local · paciente"),
            CapsuleRoot(url: home.appendingPathComponent("case-capsule-r8/out", isDirectory: true), label: "demo r8")
        ]
    }
}

private struct StoredRoot: Codable {
    let path: String
    let label: String
}
