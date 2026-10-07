import AppKit
import Foundation
import WebKit

@main
struct TractLabMain {
    static func main() {
        let application = NSApplication.shared
        let delegate = AppDelegate()
        application.delegate = delegate
        application.run()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var pickerHandler: FolderPickerHandler!
    private var serverProcess: Process?
    private(set) var serverPort: Int?
    private var homeURL: URL?
    private var startupResolved = false
    private var missingTools: [String] = []

    private var repoURL: URL {
        let env = ProcessInfo.processInfo.environment
        let path = env["TRACTLAB_REPO"] ?? "\(NSHomeDirectory())/tractlab"
        let expandedPath = (path as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expandedPath, isDirectory: true).standardizedFileURL
    }

    private var pythonPath: String { "\(NSHomeDirectory())/fsl/bin/python" }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildMenu()
        buildWindow()

        missingTools = findMissingTools()
        if missingTools.contains("Python (~/fsl/bin/python)") {
            showStartupPage(
                title: "TractLab could not start",
                detail: "The local Python interpreter is required to open the cases page.",
                items: missingTools,
                casesURL: nil
            )
            return
        }
        startServer()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        guard let process = serverProcess, process.isRunning else { return }
        process.terminate()
        process.waitUntilExit()
    }

    private func buildWindow() {
        let configuration = WKWebViewConfiguration()
        let contentController = WKUserContentController()
        pickerHandler = FolderPickerHandler(owner: self)
        contentController.add(pickerHandler, name: "pickFolder")
        configuration.userContentController = contentController

        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.setValue(false, forKey: "drawsBackground")

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1060, height: 760),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "TractLab"
        window.minSize = NSSize(width: 680, height: 500)
        window.contentView = webView
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func buildMenu() {
        let menuBar = NSMenu()

        let applicationItem = NSMenuItem()
        let applicationMenu = NSMenu(title: "TractLab")
        applicationMenu.addItem(
            NSMenuItem(title: "About TractLab", action: #selector(showAbout), keyEquivalent: "")
        )
        applicationMenu.addItem(NSMenuItem.separator())
        applicationMenu.addItem(
            NSMenuItem(title: "Quit TractLab", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        )
        applicationItem.submenu = applicationMenu
        menuBar.addItem(applicationItem)

        let casesItem = NSMenuItem()
        let casesMenu = NSMenu(title: "Cases")
        let showCasesItem = NSMenuItem(title: "Show Cases", action: #selector(showCases), keyEquivalent: "1")
        showCasesItem.target = self
        casesMenu.addItem(showCasesItem)
        casesItem.submenu = casesMenu
        menuBar.addItem(casesItem)

        NSApp.mainMenu = menuBar
    }

    @objc private func showAbout() {
        let alert = NSAlert()
        alert.messageText = "TractLab"
        alert.informativeText = "Research/preview only — not navigation."
        alert.alertStyle = .informational
        alert.runModal()
    }

    @objc private func showCases() {
        guard let homeURL else {
            showStartupPage(
                title: "TractLab is not ready",
                detail: "The local case server has not started.",
                items: missingTools,
                casesURL: nil
            )
            return
        }
        webView.load(URLRequest(url: homeURL))
    }

    private func findMissingTools() -> [String] {
        let env = ProcessInfo.processInfo.environment
        let home = NSHomeDirectory()
        let fslDir = env["FSLDIR"] ?? "\(home)/fsl"
        let antsPath = env["ANTSPATH"] ?? "\(home)/ants-2.6.5/bin"
        let fsHome = env["FREESURFER_HOME"] ?? "\(home)/freesurfer"
        let path = housePath(
            home: home,
            fslDir: fslDir,
            antsPath: antsPath,
            fsHome: fsHome,
            existing: env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        )

        var missing: [String] = []
        if !FileManager.default.isExecutableFile(atPath: pythonPath) {
            missing.append("Python (~/fsl/bin/python)")
        }
        let dcm2niix = env["DCM2NIIX"] ?? "\(home)/fsl/bin/dcm2niix"
        let dcm2niixAvailable = dcm2niix.contains("/")
            ? FileManager.default.isExecutableFile(atPath: dcm2niix)
            : hasExecutable(dcm2niix, in: path)
        if !dcm2niixAvailable {
            missing.append("dcm2niix")
        }
        if !hasExecutable("dwifslpreproc", in: path) || !hasExecutable("mrconvert", in: path) {
            missing.append("MRtrix3 (dwifslpreproc, mrconvert)")
        }
        if !hasExecutable("recon-all", in: path) {
            missing.append("FreeSurfer (recon-all)")
        }
        if !hasExecutable("antsRegistration", in: path) {
            missing.append("ANTs (antsRegistration)")
        }
        if !hasExecutable("ss3t_csd_beta1", in: path) {
            missing.append("ss3t_csd_beta1")
        }
        return missing
    }

    private func housePath(home: String, fslDir: String, antsPath: String, fsHome: String, existing: String) -> String {
        let parts = [
            "\(home)/mrtrix3tissue/pyshim",
            "\(home)/mrtrix3tissue/bin",
            antsPath,
            "\(fslDir)/share/fsl/bin",
            "\(fslDir)/bin",
            "\(home)/mrtrix3/bin",
            "\(fsHome)/bin",
        ] + existing.split(separator: ":").map(String.init)
        var seen = Set<String>()
        return parts.filter { seen.insert($0).inserted }.joined(separator: ":")
    }

    private func hasExecutable(_ name: String, in path: String) -> Bool {
        for directory in path.split(separator: ":").map(String.init) {
            if FileManager.default.isExecutableFile(atPath: "\(directory)/\(name)") {
                return true
            }
        }
        return false
    }

    private func startServer() {
        let process = Process()
        let stdout = Pipe()
        var environment = ProcessInfo.processInfo.environment
        let home = NSHomeDirectory()
        let fslDir = environment["FSLDIR"] ?? "\(home)/fsl"
        let antsPath = environment["ANTSPATH"] ?? "\(home)/ants-2.6.5/bin"
        let fsHome = environment["FREESURFER_HOME"] ?? "\(home)/freesurfer"
        environment["TRACTLAB_REPO"] = repoURL.path
        environment["PYTHONPATH"] = repoURL.appendingPathComponent("src").path
        environment["FSLDIR"] = fslDir
        environment["ANTSPATH"] = antsPath
        environment["FREESURFER_HOME"] = fsHome
        environment["PATH"] = housePath(
            home: home,
            fslDir: fslDir,
            antsPath: antsPath,
            fsHome: fsHome,
            existing: environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        )

        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["-m", "tractlab.app_server", "--port", "0"]
        process.currentDirectoryURL = repoURL
        process.environment = environment
        process.standardOutput = stdout
        process.standardError = FileHandle.standardError

        do {
            try process.run()
        } catch {
            showStartupPage(
                title: "TractLab could not start",
                detail: "The local server could not be launched. Check the repository path and Python installation.",
                items: missingTools,
                casesURL: nil
            )
            return
        }
        serverProcess = process

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let handle = stdout.fileHandleForReading
            var line = Data()
            var foundLine: Data?
            while foundLine == nil {
                let chunk = handle.availableData
                if chunk.isEmpty { break }
                if let newline = chunk.firstIndex(of: 0x0A) {
                    line.append(contentsOf: chunk[..<newline])
                    foundLine = line
                } else {
                    line.append(chunk)
                }
            }
            let text = foundLine.flatMap { String(data: $0, encoding: .utf8) }
            DispatchQueue.main.async {
                self?.handleServerLine(text)
            }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 20) { [weak self, weak process] in
            guard let self, !self.startupResolved else { return }
            self.startupResolved = true
            process?.terminate()
            self.showStartupPage(
                title: "TractLab server timed out",
                detail: "The local server did not report that it was listening.",
                items: self.missingTools,
                casesURL: nil
            )
        }
    }

    private func handleServerLine(_ line: String?) {
        guard !startupResolved else { return }
        guard let line else {
            startupResolved = true
            showStartupPage(
                title: "TractLab could not start",
                detail: "The local server exited before it reported a listening port.",
                items: missingTools,
                casesURL: nil
            )
            return
        }
        let parts = line.split(separator: " ")
        guard parts.count == 2, parts[0] == "LISTENING", let port = Int(parts[1]), (1...65535).contains(port) else {
            startupResolved = true
            showStartupPage(
                title: "TractLab could not start",
                detail: "The local server returned an unreadable startup response.",
                items: missingTools,
                casesURL: nil
            )
            return
        }
        startupResolved = true
        serverPort = port
        homeURL = URL(string: "http://127.0.0.1:\(port)/")
        guard let homeURL else { return }
        if missingTools.isEmpty {
            webView.load(URLRequest(url: homeURL))
        } else {
            showStartupPage(
                title: "Some tools are missing",
                detail: "Existing cases are still available. Install the listed tools before starting a new DICOM pipeline.",
                items: missingTools,
                casesURL: homeURL
            )
        }
    }

    private func showStartupPage(title: String, detail: String, items: [String], casesURL: URL?) {
        let escapedTitle = escapeHTML(title)
        let escapedDetail = escapeHTML(detail)
        let list = items.map { "<li>\(escapeHTML($0))</li>" }.joined()
        let continueLink: String
        if let casesURL {
            continueLink = "<p><a href=\"\(escapeHTML(casesURL.absoluteString))\">Open existing cases</a></p>"
        } else {
            continueLink = ""
        }
        let html = """
        <!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
        <title>TractLab setup</title><style>
        :root{--bg:#0d1117;--panel:#141b24;--ink:#edf2f7;--dim:#b7c3d1;--gold:#dfbf70;--line:#3b4b60;--caution:#e0a030}
        *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,sans-serif}
        main{max-width:700px;margin:12vh auto;padding:30px;background:var(--panel);border:1px solid var(--line);border-radius:6px}
        h1{margin:0 0 8px;font-size:22px}p{color:var(--dim)}.disclaimer{padding:9px 11px;border-left:2px solid var(--caution);background:#171b20}
        ul{padding-left:22px;color:var(--dim)}li{margin:5px 0}a{color:var(--gold)}
        </style></head><body><main><h1>\(escapedTitle)</h1><p>\(escapedDetail)</p>
        <ul>\(list)</ul>\(continueLink)<p class="disclaimer">Research/preview only — not navigation.</p></main></body></html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }

    private func escapeHTML(_ value: String) -> String {
        value.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#39;")
    }

    fileprivate func pickFolder() {
        let panel = NSOpenPanel()
        panel.title = "Choose a DICOM folder"
        panel.prompt = "Choose Folder"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        let result = panel.runModal()
        let path = result == .OK ? panel.url?.path ?? "" : ""
        sendPickedFolder(path)
    }

    private func sendPickedFolder(_ path: String) {
        guard let data = try? JSONSerialization.data(withJSONObject: path, options: [.fragmentsAllowed]),
              let literal = String(data: data, encoding: .utf8) else {
            return
        }
        webView.evaluateJavaScript("window.tractlabFolderPicked(\(literal));", completionHandler: nil)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let url = navigationAction.request.url,
              url.scheme == "http",
              url.host == "127.0.0.1",
              url.port != nil else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }
}

private final class FolderPickerHandler: NSObject, WKScriptMessageHandler {
    weak var owner: AppDelegate?

    init(owner: AppDelegate) {
        self.owner = owner
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "pickFolder",
              message.frameInfo.isMainFrame,
              let owner,
              let port = owner.serverPort,
              message.frameInfo.securityOrigin.protocol == "http",
              message.frameInfo.securityOrigin.host == "127.0.0.1",
              message.frameInfo.securityOrigin.port == port else {
            return
        }
        owner.pickFolder()
    }
}
