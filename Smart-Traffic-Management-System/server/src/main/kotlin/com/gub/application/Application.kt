package com.gub.application

import com.gub.SERVER_PORT
import com.gub.di.dashboardModule
import com.gub.routes.analytics
import com.gub.routes.dashboardRoute
import com.gub.routes.monitoring
import com.gub.routes.settings
import com.gub.routes.system
import com.gub.utils.InstantSerializer
import com.gub.utils.json
import io.ktor.http.ContentType.Application.Json
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.routing.*
import io.ktor.server.websocket.*
import kotlinx.serialization.json.Json
import kotlinx.serialization.modules.SerializersModule
import org.koin.ktor.plugin.Koin
import java.time.Instant

fun main() {
    embeddedServer(
        factory = Netty, port = SERVER_PORT,
        host = "0.0.0.0", module = Application::module
    ).start(wait = true)
}

fun Application.module() {
    // Websocket
    install(WebSockets)

    // Configure Koin
    install(Koin) {
        modules(dashboardModule)
    }

    // Configure JSON serialization
    install(ContentNegotiation) {
        json()
    }

    // Configure routing
    routing {
        system()

        settings()
        analytics()
        monitoring()
        dashboardRoute()
    }
}