import AppKit
import CapsuleCore
import Foundation
import UniformTypeIdentifiers
import WebKit

extension Notification.Name {
    static let capsuleDownloadFinished = Notification.Name("CaseCapsulesDownloadFinished")
}

@MainActor
final class DownloadDestinationCoordinator {
    private var reservedPaths: Set<URL> = []

    func reserve(source: URL, suggestion: String) -> URL {
        let destination = SaveNaming.destination(
            source: source,
            suggestedFilename: suggestion,
            reserved: reservedPaths
        )
        reservedPaths.insert(destination.standardizedFileURL)
        return destination
    }

    func release(_ destination: URL) {
        reservedPaths.remove(destination.standardizedFileURL)
    }
}

enum OfflineRuleCompiler {
    private static let identifier = "com.deandrade.casecapsules.offline-v1"
    private static let encodedRules = #"[{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},{"trigger":{"url-filter":"^wss?://"},"action":{"type":"block"}}]"#

    static func compile(completion: @escaping (Result<WKContentRuleList, Error>) -> Void) {
        guard let store = WKContentRuleListStore.default() else {
            DispatchQueue.main.async {
                completion(.failure(OfflineRuleError.storeUnavailable))
            }
            return
        }

        store.compileContentRuleList(
            forIdentifier: identifier,
            encodedContentRuleList: encodedRules
        ) { ruleList, error in
            DispatchQueue.main.async {
                if let ruleList {
                    completion(.success(ruleList))
                } else {
                    completion(.failure(error ?? OfflineRuleError.compilationFailed))
                }
            }
        }
    }
}

private enum OfflineRuleError: Error {
    case storeUnavailable
    case compilationFailed
}

@MainActor
final class CapsuleWebViewController: NSViewController, WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate {
    let capsule: CapsuleInfo
    let webView: WKWebView
    var onDownloadFinished: ((URL) -> Void)?
    var onDownloadFailed: ((String) -> Void)?
    var onViewerFailure: ((String) -> Void)?

    private let showsBanner: Bool
    private let downloadCoordinator: DownloadDestinationCoordinator
    private var activeDownloads: [ObjectIdentifier: URL] = [:]
    private let banner = NSStackView()
    private let bannerLabel = NSTextField(labelWithString: "")
    private let bannerButton = NSButton(title: "", target: nil, action: nil)
    private var bannerAction: BannerAction = .none

    private enum BannerAction {
        case none
        case reload
        case reveal(URL)
    }

    init(
        capsule: CapsuleInfo,
        ruleList: WKContentRuleList,
        showsBanner: Bool = true,
        downloadCoordinator: DownloadDestinationCoordinator? = nil
    ) {
        self.capsule = capsule
        self.showsBanner = showsBanner
        self.downloadCoordinator = downloadCoordinator ?? DownloadDestinationCoordinator()

        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(ruleList)
        self.webView = WKWebView(frame: .zero, configuration: configuration)
        super.init(nibName: nil, bundle: nil)

        if #available(macOS 13.0, *) {
            webView.isInspectable = ProcessInfo.processInfo.environment["CAPSULES_DEBUG"] == "1"
        }
    }

    required init?(coder: NSCoder) {
        return nil
    }

    override func loadView() {
        let container = NSView(frame: NSRect(x: 0, y: 0, width: 1440, height: 900))
        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .width
        stack.distribution = .fill
        stack.spacing = 0
        stack.translatesAutoresizingMaskIntoConstraints = false

        banner.orientation = .horizontal
        banner.alignment = .centerY
        banner.spacing = 12
        banner.edgeInsets = NSEdgeInsets(top: 8, left: 12, bottom: 8, right: 12)
        bannerLabel.lineBreakMode = .byTruncatingTail
        banner.addArrangedSubview(bannerLabel)
        bannerButton.target = self
        bannerButton.action = #selector(performBannerAction)
        banner.addArrangedSubview(bannerButton)
        banner.isHidden = true

        webView.translatesAutoresizingMaskIntoConstraints = false
        stack.addArrangedSubview(banner)
        stack.addArrangedSubview(webView)
        container.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            stack.topAnchor.constraint(equalTo: container.topAnchor),
            stack.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            webView.heightAnchor.constraint(greaterThanOrEqualToConstant: 200)
        ])
        view = container
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.loadFileURL(
            capsule.url,
            allowingReadAccessTo: capsule.url.deletingLastPathComponent()
        )
    }

    func evaluateJavaScript(_ source: String, completion: @escaping (Any?, Error?) -> Void) {
        webView.evaluateJavaScript(source, completionHandler: completion)
    }

    func takeSnapshot(completion: @escaping (NSImage?, Error?) -> Void) {
        webView.takeSnapshot(with: nil, completionHandler: completion)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        if navigationAction.shouldPerformDownload {
            decisionHandler(.download)
            return
        }

        guard let scheme = navigationAction.request.url?.scheme?.lowercased(),
              ["file", "blob", "data", "about"].contains(scheme) else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        let disposition = (navigationResponse.response as? HTTPURLResponse)?
            .value(forHTTPHeaderField: "Content-Disposition")
        let isAttachment = disposition?.localizedCaseInsensitiveContains("attachment") == true
        if !navigationResponse.canShowMIMEType || isAttachment {
            decisionHandler(.download)
        } else {
            decisionHandler(.allow)
        }
    }

    func webView(
        _ webView: WKWebView,
        navigationAction: WKNavigationAction,
        didBecome download: WKDownload
    ) {
        download.delegate = self
    }

    func webView(
        _ webView: WKWebView,
        navigationResponse: WKNavigationResponse,
        didBecome download: WKDownload
    ) {
        download.delegate = self
    }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping ([URL]?) -> Void
    ) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.begin { response in
            completionHandler(response == .OK ? panel.urls : nil)
        }
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        onViewerFailure?("web content process terminated")
        showBanner("O visualizador parou (memória?)", buttonTitle: "Recarregar", action: .reload)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        showNavigationError(error)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        showNavigationError(error)
    }

    func download(
        _ download: WKDownload,
        decideDestinationUsing response: URLResponse,
        suggestedFilename: String,
        completionHandler: @escaping (URL?) -> Void
    ) {
        let destination = downloadCoordinator.reserve(source: capsule.url, suggestion: suggestedFilename)
        activeDownloads[ObjectIdentifier(download)] = destination
        completionHandler(destination)
    }

    func downloadDidFinish(_ download: WKDownload) {
        guard let destination = finishDownload(download) else { return }
        if showsBanner {
            showBanner("Salvo: \(destination.lastPathComponent)", buttonTitle: "Mostrar no Finder", action: .reveal(destination))
        }
        onDownloadFinished?(destination)
        NotificationCenter.default.post(name: .capsuleDownloadFinished, object: nil)
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        let destination = activeDownloads[ObjectIdentifier(download)]
        if let destination { downloadCoordinator.release(destination) }
        activeDownloads[ObjectIdentifier(download)] = nil
        let message = "Não foi possível salvar o arquivo. \(error.localizedDescription)"
        if showsBanner { showBanner(message, buttonTitle: "Fechar", action: .none) }
        onDownloadFailed?(message)
    }

    private func finishDownload(_ download: WKDownload) -> URL? {
        let id = ObjectIdentifier(download)
        guard let destination = activeDownloads.removeValue(forKey: id) else { return nil }
        downloadCoordinator.release(destination)
        return destination
    }

    private func showNavigationError(_ error: Error) {
        onViewerFailure?("navigation failed")
        showBanner("Falha ao abrir o visualizador. \(error.localizedDescription)", buttonTitle: "Recarregar", action: .reload)
    }

    private func showBanner(_ message: String, buttonTitle: String, action: BannerAction) {
        guard showsBanner else { return }
        bannerLabel.stringValue = message
        bannerButton.title = buttonTitle
        bannerAction = action
        banner.isHidden = false
    }

    @objc private func performBannerAction() {
        switch bannerAction {
        case .none:
            banner.isHidden = true
        case .reload:
            banner.isHidden = true
            webView.reload()
        case let .reveal(url):
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
    }
}
