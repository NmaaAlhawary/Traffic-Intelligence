package com.gub.features.dashboard.data.remote

import com.gub.domain.models.dashboard.ModelLiveTraffic
import com.gub.models.dashboard.overview.ModelSystemOverview
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.request.*
import io.ktor.http.*

interface SystemOverviewApi {

    suspend fun getSystemOverview(): ModelSystemOverview

    suspend fun updateSystemOverview(overview: ModelSystemOverview): Unit

    suspend fun liveTrafficMatrics(): ModelLiveTraffic
}


class SystemOverviewApiImpl(
    private val httpClient: HttpClient
) : SystemOverviewApi {

    override suspend fun getSystemOverview(): ModelSystemOverview {
        return httpClient.get("/api/dashboard/system").body()
    }

    override suspend fun updateSystemOverview(overview: ModelSystemOverview) {
        httpClient.post("/api/dashboard/system") {
            contentType(ContentType.Application.Json)
            setBody(overview)
        }
    }

    override suspend fun liveTrafficMatrics(): ModelLiveTraffic {
        return httpClient.get("/api/dashboard/traffic").body()
    }
}