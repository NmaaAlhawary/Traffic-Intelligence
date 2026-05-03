package com.gub.routes

import com.gub.domain.models.analytics.fromValue
import com.gub.domain.usecase.analytics.UseCaseTrafficVolume
import com.gub.domain.usecase.monitoring.UseCaseLiveSignal
import io.ktor.http.HttpStatusCode
import io.ktor.server.response.respond
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.route
import org.koin.ktor.ext.inject
import kotlin.getValue

fun Route.analytics() {

    val trafficVolume by inject<UseCaseTrafficVolume>()

    route("/api/analytics") {

        get("/traffic-volume") {
            val type = call.parameters["type"]
            type?.let {
                call.respond(
                    HttpStatusCode.OK,
                    trafficVolume(it.fromValue())
                )

            } ?: call.respond("Invalid traffic volume type $type")
        }
    }
}