package com.gub.features.dashboard.data.repository

import com.gub.core.domain.Response
import com.gub.domain.models.dashboard.ModelLiveTraffic
import com.gub.features.dashboard.data.remote.SystemOverviewApi
import com.gub.features.dashboard.domain.repository.SystemOverviewRepository
import com.gub.models.dashboard.overview.ModelSystemOverview
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class SystemOverviewRepositoryImpl(
    private val api: SystemOverviewApi
) : SystemOverviewRepository {

    override suspend fun getSystemOverview(): Response<ModelSystemOverview> {
        return try {
            val dto = api.getSystemOverview()
            Response.Success(dto)
        } catch (e: Exception) {
            Response.Error(e.toString())
        }
    }

    override fun getSystemOverviewStream(): Flow<ModelSystemOverview> = flow {
        while (true) {
            try {
                val overview = api.getSystemOverview()
                emit(overview)
                delay(5000) // Refresh every 5 seconds
            } catch (e: Exception) {
                // Handle error or emit default values
                emit(ModelSystemOverview())
                delay(5000) // Refresh every 5 seconds
            }
        }
    }

    override suspend fun getTrafficMatrics(): Response<ModelLiveTraffic> {
        return try {
            Response.Success(api.liveTrafficMatrics())
        } catch (e: Exception) {
            Response.Error(e.toString())
        }
    }
}