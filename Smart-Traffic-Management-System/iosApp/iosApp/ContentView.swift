//import UIKit
//import SwiftUI
////import ComposeApp
//
////struct ComposeView: UIViewControllerRepresentable {
////    func makeUIViewController(context: Context) -> UIViewController {
////        MainViewControllerKt.MainViewController()
////    }
////
////    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}
////}
//
//struct ContentView: View {
//    var body: some View {
////        ComposeView()
////                .ignoresSafeArea(.keyboard) // Compose has own keyboard handler
//    }
//}
//
//
//



import SwiftUI

struct ContentView: View {
    @State private var ip: String = "34.87.172.238"
    @State private var port: String = "8000"
    @State private var finalUrl: String = "ws://34.87.172.238:8000/ws"
    @State private var hasPermission: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            if hasPermission {
                CameraView(finalUrl: finalUrl)
                    .frame(maxHeight: .infinity)
                    .background(Color.black)
            } else {
                VStack {
                    Text("Camera access is required to stream video.")
                        .padding()
                    Button("Grant Camera Access") {
                        requestCameraPermission()
                    }
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .frame(maxHeight: .infinity)
            }

            VStack(spacing: 16) {
                TextField("Enter Server IP", text: $ip)
                    .textFieldStyle(.roundedBorder)
                    .keyboardType(.numbersAndPunctuation)

                TextField("Enter Server Port", text: $port)
                    .textFieldStyle(.roundedBorder)
                    .keyboardType(.numberPad)

                Button("Connect") {
                    finalUrl = "ws://\(ip):\(port)/ws"
                }
                .padding()
                .background(Color.green)
                .foregroundColor(.white)
                .clipShape(RoundedRectangle(cornerRadius: 10))

                Text("Connected To: \(finalUrl)")
                    .font(.caption)
                    .padding(.top)
            }
            .padding()
            .background(Color(UIColor.secondarySystemBackground))
        }
        .onAppear {
            requestCameraPermission()
        }
        .ignoresSafeArea()
    }

    private func requestCameraPermission() {
        CameraPermissionManager.requestCameraAccess { granted in
            hasPermission = granted
        }
    }
}
