package com.gub.routes

import com.gub.domain.usecase.monitoring.UseCaseLiveSignal
import io.ktor.http.HttpStatusCode
import io.ktor.server.response.respond
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.route
import org.koin.ktor.ext.inject
import kotlin.getValue

fun Route.monitoring() {

    val liveSignal by inject<UseCaseLiveSignal>()

    route("/api/monitoring") {

        get("/signal") {
            call.respond(HttpStatusCode.OK, liveSignal())
        }

    }
}