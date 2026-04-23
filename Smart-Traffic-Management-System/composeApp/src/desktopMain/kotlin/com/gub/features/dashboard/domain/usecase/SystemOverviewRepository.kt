package com.gub.features.dashboard.domain.usecase

import com.gub.core.domain.Response
import com.gub.domain.models.dashboard.ModelLiveTraffic
import com.gub.features.dashboard.domain.repository.SystemOverviewRepository

class UpdateSystemMetricsUseCase(
    private val repository: SystemOverviewRepository
) {
    suspend operator fun invoke(): Response<ModelLiveTraffic> {
        return repository.getTrafficMatrics()
    }
}