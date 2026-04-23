package com.gub.features.monitoring.domain.usecase

import com.gub.core.domain.Response
import com.gub.domain.models.monitoring.ModelLiveSignal
import com.gub.features.monitoring.domain.repository.RepositoryLiveSignal

class UseCaseLiveSignal(
    private val repository: RepositoryLiveSignal
) {

    suspend operator fun invoke(): Response<ModelLiveSignal> {
        return repository.getLiveSignal()
    }
}