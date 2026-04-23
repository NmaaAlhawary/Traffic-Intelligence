package com.gub.features.dashboard.domain.repository


import com.gub.core.domain.Response
import com.gub.domain.models.dashboard.ModelLiveTraffic
import com.gub.models.dashboard.overview.ModelSystemOverview
import kotlinx.coroutines.flow.Flow

interface SystemOverviewRepository {

    suspend fun getSystemOverview(): Response<ModelSystemOverview>

    fun getSystemOverviewStream(): Flow<ModelSystemOverview>

    suspend fun getTrafficMatrics(): Response<ModelLiveTraffic>
}