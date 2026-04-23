package com.gub.features.dashboard.domain.usecase

import com.gub.core.domain.Response
import com.gub.features.dashboard.domain.repository.SystemOverviewRepository
import com.gub.models.dashboard.overview.ModelSystemOverview
import kotlinx.coroutines.flow.Flow

class GetSystemOverviewUseCase(
    private val repository: SystemOverviewRepository
) {
    suspend operator fun invoke(): Response<ModelSystemOverview> {
        return repository.getSystemOverview()
    }

    fun getStream(): Flow<ModelSystemOverview> {
        return repository.getSystemOverviewStream()
    }
}