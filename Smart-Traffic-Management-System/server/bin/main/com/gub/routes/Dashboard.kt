package com.gub.routes

import com.gub.domain.usecase.dashboard.UseCaseLiveTraffic
import com.gub.domain.usecase.dashboard.UseCaseSystemOverview
import io.ktor.http.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import org.koin.ktor.ext.inject

fun Route.dashboardRoute() {

    val liveTraffic by inject<UseCaseLiveTraffic>()
    val systemOverview by inject<UseCaseSystemOverview>()

    route("/api/dashboard") {

//        get("/ai-control") {
//            val result = repositoryDashboard.getAiControlSystem()
//            call.respond(HttpStatusCode.OK, result)
//        }
//
        get("/traffic") {
            call.respond(HttpStatusCode.OK, liveTraffic())
        }

        get("/system") {
            call.respond(HttpStatusCode.OK, systemOverview())
        }
    }
}