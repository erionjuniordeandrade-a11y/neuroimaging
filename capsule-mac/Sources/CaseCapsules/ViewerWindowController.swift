import AppKit
import CapsuleCore
import WebKit

@MainActor
final class ViewerWindowController: NSWindowController, NSWindowDelegate {
    let capsule: CapsuleInfo
    let webContent: CapsuleWebViewController
    var onClosed: ((String) -> Void)?

    init(
        capsule: CapsuleInfo,
        ruleList: WKContentRuleList,
        downloadCoordinator: DownloadDestinationCoordinator,
        onDownloadFinished: @escaping (URL) -> Void,
        onDownloadFailed: @escaping (String) -> Void,
        onClosed: @escaping (String) -> Void
    ) {
        self.capsule = capsule
        self.webContent = CapsuleWebViewController(
            capsule: capsule,
            ruleList: ruleList,
            downloadCoordinator: downloadCoordinator
        )
        self.onClosed = onClosed

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1440, height: 900),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.tabbingMode = .preferred
        window.tabbingIdentifier = "com.deandrade.casecapsules.viewer"
        window.contentViewController = webContent
        window.setContentSize(NSSize(width: 1440, height: 900))
        window.title = "\(capsule.caseLabel) — v\(capsule.version ?? 0)"
        window.setFrameAutosaveName("CaseCapsulesViewer")
        if !window.setFrameUsingName("CaseCapsulesViewer") {
            window.center()
        }
        super.init(window: window)

        self.webContent.onDownloadFinished = onDownloadFinished
        self.webContent.onDownloadFailed = onDownloadFailed
        window.delegate = self
    }

    required init?(coder: NSCoder) {
        return nil
    }

    func windowWillClose(_ notification: Notification) {
        onClosed?(capsule.id)
    }
}

@MainActor
final class ViewerWindowManager {
    var onDownloadFinished: (() -> Void)?
    private var windows: [String: ViewerWindowController] = [:]
    private let downloadCoordinator = DownloadDestinationCoordinator()

    func open(
        capsule: CapsuleInfo,
        ruleList: WKContentRuleList,
        onOpened: @escaping () -> Void
    ) {
        let controller = controller(for: capsule, ruleList: ruleList)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
        onOpened()
    }

    func openAsTabs(
        capsules: [CapsuleInfo],
        ruleList: WKContentRuleList,
        onOpened: @escaping (CapsuleInfo) -> Void
    ) {
        guard let first = capsules.first else { return }
        let base = controller(for: first, ruleList: ruleList)
        base.showWindow(nil)
        base.window?.makeKeyAndOrderFront(nil)
        onOpened(first)

        for capsule in capsules.dropFirst() {
            let controller = controller(for: capsule, ruleList: ruleList)
            if let baseWindow = base.window, let window = controller.window, controller !== base {
                baseWindow.addTabbedWindow(window, ordered: .above)
                window.makeKeyAndOrderFront(nil)
            }
            onOpened(capsule)
        }
    }

    private func controller(for capsule: CapsuleInfo, ruleList: WKContentRuleList) -> ViewerWindowController {
        if let existing = windows[capsule.id] { return existing }

        let controller = ViewerWindowController(
            capsule: capsule,
            ruleList: ruleList,
            downloadCoordinator: downloadCoordinator,
            onDownloadFinished: { [weak self] _ in self?.onDownloadFinished?() },
            onDownloadFailed: { _ in },
            onClosed: { [weak self] id in self?.windows[id] = nil }
        )
        windows[capsule.id] = controller
        return controller
    }
}
