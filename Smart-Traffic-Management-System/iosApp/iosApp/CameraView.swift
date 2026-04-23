//
//  CameraView.swift
//  iosApp
//
//  Created by Abdul Alim on 7/13/25.
//

import SwiftUI
import AVFoundation

struct CameraView: UIViewRepresentable {
    class CameraCoordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
        private let session = AVCaptureSession()
        private var webSocket: URLSessionWebSocketTask?
        private var finalUrl: String

        private let videoOutput = AVCaptureVideoDataOutput()
        private let queue = DispatchQueue(label: "videoQueue")

        init(finalUrl: String) {
            self.finalUrl = finalUrl
            super.init()

            setupSession()
            setupWebSocket()
        }

        func setupSession() {
            session.beginConfiguration()
            session.sessionPreset = .medium

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else {
                print("❌ Failed to initialize camera")
                return
            }

            if session.canAddInput(input) {
                session.addInput(input)
            }

            videoOutput.setSampleBufferDelegate(self, queue: queue)
            videoOutput.alwaysDiscardsLateVideoFrames = true

            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }

            session.commitConfiguration()
            session.startRunning()
        }

        func setupWebSocket() {
            guard let url = URL(string: finalUrl) else { return }
            let session = URLSession(configuration: .default)
            webSocket = session.webSocketTask(with: url)
            webSocket?.resume()
        }

        func makePreviewLayer() -> AVCaptureVideoPreviewLayer {
            let layer = AVCaptureVideoPreviewLayer(session: session)
            layer.videoGravity = .resizeAspectFill
            return layer
        }

        func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
            guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

            let ciImage = CIImage(cvPixelBuffer: imageBuffer)
            let context = CIContext()

            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }

            let image = UIImage(cgImage: cgImage)
            let resized = image.resized(to: CGSize(width: 720, height: 720)) // Optional resize

            guard let jpegData = resized.jpegData(compressionQuality: 0.6) else { return }

            let message = URLSessionWebSocketTask.Message.data(jpegData)
            webSocket?.send(message) { error in
                if let error = error {
                    print("WebSocket send error: \(error)")
                }
            }
        }

        deinit {
            session.stopRunning()
            webSocket?.cancel()
        }
    }

    var finalUrl: String

    func makeCoordinator() -> CameraCoordinator {
        return CameraCoordinator(finalUrl: finalUrl)
    }

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let previewLayer = context.coordinator.makePreviewLayer()
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)

        DispatchQueue.main.async {
            previewLayer.frame = view.bounds
        }

        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}


extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, self.scale)
        self.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized ?? self
    }
}
