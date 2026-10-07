import AppKit
import CapsuleCore
import Darwin
import Foundation
import WebKit

@MainActor
final class SelfTestRunner {
    private let capsuleURL: URL
    private let outputURL: URL
    private let reportedOutputPath: String
    private let downloads: Bool
    private let downloadCoordinator = DownloadDestinationCoordinator()
    private var controller: CapsuleWebViewController?
    private var window: NSWindow?
    private var pollTimer: Timer?
    private var elapsedTimer: Timer?
    private var startedAt = 0.0
    private var outputByteCount = 0
    private var ready = false
    private var finished = false
    private let timeoutSeconds: TimeInterval = 120

    init(capsuleURL: URL, outputPath: String, downloads: Bool) {
        self.capsuleURL = capsuleURL.standardizedFileURL
        self.outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        self.reportedOutputPath = outputPath
        self.downloads = downloads
    }

    func start() {
        guard FileManager.default.fileExists(atPath: capsuleURL.path) else {
            finishFailure("capsule file unavailable")
            return
        }

        let manifest: CapsuleManifest
        do {
            manifest = try ManifestReader.read(from: capsuleURL)
        } catch {
            finishFailure("manifest unavailable")
            return
        }

        let values = try? capsuleURL.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        let root = CapsuleRoot(url: capsuleURL.deletingLastPathComponent(), label: "self-test")
        let capsule = CapsuleInfo(
            url: capsuleURL,
            root: root,
            modificationDate: values?.contentModificationDate,
            sizeBytes: values?.fileSize.map(Int64.init),
            contents: .readable(manifest)
        )

        OfflineRuleCompiler.compile { [weak self] result in
            guard let self else { return }
            switch result {
            case let .success(ruleList): self.beginWebView(capsule: capsule, ruleList: ruleList)
            case .failure: self.finishFailure("offline content rules could not compile")
            }
        }
    }

    private func beginWebView(capsule: CapsuleInfo, ruleList: WKContentRuleList) {
        guard !finished else { return }
        let controller = CapsuleWebViewController(
            capsule: capsule,
            ruleList: ruleList,
            showsBanner: false,
            downloadCoordinator: downloadCoordinator
        )
        controller.onDownloadFinished = { [weak self] url in self?.downloadFinished(at: url) }
        controller.onDownloadFailed = { [weak self] _ in self?.finishFailure("self-test download failed") }
        controller.onViewerFailure = { [weak self] reason in self?.finishFailure(reason) }
        self.controller = controller

        let window = NSWindow(
            contentRect: NSRect(x: -1800, y: -1200, width: 1440, height: 900),
            styleMask: .borderless,
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        window.contentViewController = controller
        window.setFrame(NSRect(x: -1800, y: -1200, width: 1440, height: 900), display: false)
        self.window = window
        window.orderFrontRegardless()

        startedAt = ProcessInfo.processInfo.systemUptime
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.pollReadiness() }
        }
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: timeoutSeconds, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, !self.ready else { return }
                self.finishFailure("timeout waiting for manifest and WebGL2")
            }
        }
    }

    private func pollReadiness() {
        guard !finished, !ready, let controller else { return }
        let script = #"(() => { const node = document.getElementById('capsule-manifest'); let manifest = false; try { manifest = !!node && !!JSON.parse(node.textContent || ''); } catch (_) {} const webgl2 = Array.from(document.querySelectorAll('canvas')).some(c => { try { return !!c.getContext('webgl2'); } catch (_) { return false; } }); const visible = e => { const s = getComputedStyle(e); const r = e.getBoundingClientRect(); return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity) !== 0 && r.width > 0 && r.height > 0; }; const errors = Array.from(document.querySelectorAll('[role="alert"], [data-viewer-error], .viewer-error, .error-state, #viewer-error')).some(visible); return JSON.stringify({manifest, webgl2, errors}); })()"#
        controller.evaluateJavaScript(script) { [weak self] result, _ in
            guard let self, !self.finished,
                  let result = result as? String,
                  let data = result.data(using: .utf8),
                  let state = try? JSONSerialization.jsonObject(with: data) as? [String: Bool] else { return }
            if state["errors"] == true {
                self.finishFailure("viewer reported a visible error")
            } else if state["manifest"] == true && state["webgl2"] == true {
                self.ready = true
                self.pollTimer?.invalidate()
                self.elapsedTimer?.invalidate()
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                    self.captureSnapshot()
                }
            }
        }
    }

    private func captureSnapshot() {
        guard !finished, let controller else { return }
        controller.takeSnapshot { [weak self] image, error in
            Task { @MainActor in
                if let error {
                    FileHandle.standardError.write(Data("snapshot error: \((error as NSError).domain) \((error as NSError).code)\n".utf8))
                }
                guard let self, !self.finished, let image,
                      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                    self?.finishFailure("snapshot failed")
                    return
                }
                do {
                    let bitmap = NSBitmapImageRep(cgImage: cgImage)
                    guard let png = bitmap.representation(using: .png, properties: [:]) else {
                        self.finishFailure("snapshot encoding failed")
                        return
                    }
                    try FileManager.default.createDirectory(
                        at: self.outputURL.deletingLastPathComponent(),
                        withIntermediateDirectories: true
                    )
                    try png.write(to: self.outputURL, options: .atomic)
                    self.outputByteCount = png.count
                    if self.downloads {
                        self.triggerDownload()
                    } else {
                        self.finishSuccess(download: false)
                    }
                } catch {
                    self.finishFailure("could not write snapshot")
                }
            }
        }
    }

    private func triggerDownload() {
        guard let controller else { finishFailure("self-test viewer unavailable"); return }
        let script = #"(() => { const blob = new Blob(['capsule-selftest'], {type:'text/plain'}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'selftest.txt'; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000); return 'started'; })()"#
        controller.evaluateJavaScript(script) { [weak self] _, error in
            guard let self else { return }
            if error != nil {
                self.finishFailure("self-test download trigger failed")
                return
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
                if !self.finished { self.finishFailure("self-test download timed out") }
            }
        }
    }

    private func downloadFinished(at url: URL) {
        guard downloads, !finished else { return }
        let expectedDirectory = capsuleURL.deletingLastPathComponent().standardizedFileURL
        let actual = url.standardizedFileURL
        guard actual.deletingLastPathComponent() == expectedDirectory,
              actual != capsuleURL,
              FileManager.default.fileExists(atPath: actual.path) else {
            finishFailure("download did not appear beside the capsule")
            return
        }

        do {
            try FileManager.default.removeItem(at: actual)
            finishSuccess(download: true)
        } catch {
            finishFailure("self-test download cleanup failed")
        }
    }

    private func finishSuccess(download: Bool) {
        guard !finished else { return }
        finished = true
        let milliseconds = Int((ProcessInfo.processInfo.systemUptime - startedAt) * 1000)
        SelfTestOutput.success(
            elapsedMilliseconds: milliseconds,
            outputPath: reportedOutputPath,
            byteCount: outputByteCount,
            download: download
        )
        terminate(status: 0)
    }

    private func finishFailure(_ reason: String) {
        guard !finished else { return }
        finished = true
        SelfTestOutput.failure(reason)
        terminate(status: 1)
    }

    private func terminate(status: Int32) {
        pollTimer?.invalidate()
        elapsedTimer?.invalidate()
        window?.orderOut(nil)
        DispatchQueue.main.async { exit(status) }
    }
}

enum SelfTestOutput {
    private struct Report: Encodable {
        let ok: Bool
        let webgl2: Bool?
        let ms: Int?
        let png: String?
        let bytes: Int?
        let download: Bool?
        let reason: String?
    }

    static func success(elapsedMilliseconds: Int, outputPath: String, byteCount: Int, download: Bool) {
        emit(Report(
            ok: true,
            webgl2: true,
            ms: elapsedMilliseconds,
            png: outputPath,
            bytes: byteCount,
            download: download ? true : nil,
            reason: nil
        ))
    }

    static func failure(_ reason: String) {
        emit(Report(ok: false, webgl2: nil, ms: nil, png: nil, bytes: nil, download: nil, reason: reason))
    }

    private static func emit(_ report: Report) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        if let data = try? encoder.encode(report), let line = String(data: data, encoding: .utf8) {
            Swift.print(line)
        } else {
            Swift.print(#"{"ok":false,"reason":"self-test output encoding failed"}"#)
        }
    }
}
